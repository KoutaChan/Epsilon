package com.epsilon.agent.runtime;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtPinnedBuffer;
import ai.djl.pytorch.engine.PtStream;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayDeque;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;

/**
 * 1 つの GPU の上で、ホストからの入力の転送、順伝播、ホストへの出力の転送を、スロットごとにずらして重ねる。
 *
 * <p>モデルは {@link Lease} を通して使う。同じ GPU のモデルは、3 本のスレッド（入力を用意する、計算を出す、完了を見る）と、3 本の
 * ストリーム（入力の転送、計算、出力の転送。{@link NetworkRuntime.Streams}）を共有する。完了は GPU のイベントの状態を見て確かめる
 * （{@link PtEvent#isComplete()}）。ホストが GPU を待つのは、失敗と終了の後始末のときだけ。
 *
 * <p>ホストの容量（符号化を終えてスロットを待てるバッチの数）は、スロットとは別に数える。その空きは {@link InferenceDispatcher.Device}
 * として推論の受付に知らせる。失敗した後でスロットの後始末が済んだと確かめられなければ、スロットとモデルのリソースは解放せずに隔離し、
 * 以後の利用を拒む。
 */
public final class DecisionDevicePipeline implements InferenceDispatcher.Device {

  private static final long POLL_NANOS = TimeUnit.MICROSECONDS.toNanos(50);

  private final Device device;
  private final int slotCount;
  private final int readyCapacity;
  private final PtNDManager manager;
  private final NetworkRuntime.Streams streams;
  private final HostReadyCapacity hostReadyCapacity;
  private final BlockingQueue<Entry> ready;
  private final BlockingQueue<Slot> freeSlots;
  private final BlockingQueue<Slot> submitActions = new LinkedBlockingQueue<>();
  private final BlockingQueue<Slot> submittedSlots = new LinkedBlockingQueue<>();
  private final AtomicLong pendingWork = new AtomicLong();
  private final Counters counters = new Counters();
  private final Thread stagingThread;
  private final Thread submitThread;
  private final Thread completionThread;
  private int references;
  private volatile boolean running = true;
  private volatile Throwable terminalFailure;

  /**
   * 渡したストリームの GPU で共有するパイプラインを借りる。まだ無ければ作る。既にあるパイプラインが止まっているか失敗しているとき、
   * またはスロットの数か容量が違うときは、IllegalStateException を投げる。
   */
  static Lease acquire(NetworkRuntime.Streams streams, int slotCount, int readyCapacity) {
    synchronized (streams) {
      DecisionDevicePipeline pipeline = streams.pipeline;
      if (pipeline == null) {
        pipeline = new DecisionDevicePipeline(streams, slotCount, readyCapacity);
        streams.pipeline = pipeline;
      } else if (!pipeline.running || pipeline.terminalFailure != null) {
        throw new IllegalStateException(
            "The decision pipeline on " + streams.device() + " failed", pipeline.terminalFailure);
      } else if (pipeline.slotCount != slotCount || pipeline.readyCapacity != readyCapacity) {
        throw new IllegalStateException(
            "The decision pipeline on " + streams.device() + " runs " + pipeline.slotCount + '/'
                + pipeline.readyCapacity + " but " + slotCount + '/' + readyCapacity + " was requested");
      }
      pipeline.references++;
      return new Lease(pipeline);
    }
  }

  /**
   * 渡したストリームの GPU に、パイプラインが残っていないことを確かめる。残っていれば IllegalStateException を投げる。使用中の
   * パイプラインや、失敗で隔離したリソースを残したまま次の段へ進まないために呼ぶ。
   */
  static void requireReleased(NetworkRuntime.Streams streams) {
    synchronized (streams) {
      DecisionDevicePipeline pipeline = streams.pipeline;
      if (pipeline != null) {
        throw new IllegalStateException(
            "The runtime still holds a decision pipeline on " + streams.device(), pipeline.terminalFailure);
      }
    }
  }

  /** 渡したストリームの GPU のパイプラインが、失敗でリソースを隔離していれば IllegalStateException を投げる。隔離は解けない。 */
  static void requireNotQuarantined(NetworkRuntime.Streams streams) {
    synchronized (streams) {
      DecisionDevicePipeline pipeline = streams.pipeline;
      if (pipeline != null && pipeline.terminalFailure != null) {
        throw new IllegalStateException(
            "The decision pipeline on " + streams.device() + " failed", pipeline.terminalFailure);
      }
    }
  }

  private DecisionDevicePipeline(NetworkRuntime.Streams streams, int slotCount, int readyCapacity) {
    this.device = streams.device();
    this.slotCount = slotCount;
    this.readyCapacity = readyCapacity;
    this.streams = streams;
    manager = (PtNDManager) NDManager.newBaseManager(device);
    hostReadyCapacity = new HostReadyCapacity(readyCapacity);
    ready = new ArrayBlockingQueue<>(readyCapacity);
    freeSlots = new ArrayBlockingQueue<>(slotCount);
    for (int index = 0; index < slotCount; index++) {
      freeSlots.add(new Slot(index, manager, streams.h2d(), streams.compute(), streams.d2h()));
    }
    String suffix = device.getDeviceId() + "-" + Integer.toHexString(System.identityHashCode(this));
    stagingThread = Thread.ofPlatform().daemon().name("epsilon-decision-stage-" + suffix).start(this::stageLoop);
    submitThread = Thread.ofPlatform().daemon().name("epsilon-decision-submit-" + suffix).start(this::submitLoop);
    completionThread =
        Thread.ofPlatform().daemon().name("epsilon-decision-complete-" + suffix).start(this::completionLoop);
  }

  @Override
  public Object key() {
    return device;
  }

  @Override
  public int slots() {
    return slotCount;
  }

  @Override
  public int hostCapacity() {
    return readyCapacity;
  }

  @Override
  public int suppliedBatches() {
    return slotCount - freeSlots.size() + hostReadyCapacity.inUse();
  }

  @Override
  public long pendingWork() {
    return pendingWork.get();
  }

  @Override
  public CompletableFuture<Void> capacityAvailable() {
    return hostReadyCapacity.availability();
  }

  /** 全てのモデルが積んだバッチが終わるまで（残りの作業量が 0 になるまで）待つ。 */
  void awaitIdle() {
    boolean interrupted = false;
    synchronized (this) {
      while (pendingWork.get() != 0L) {
        try {
          wait();
        } catch (InterruptedException ignored) {
          interrupted = true;
        }
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  Metrics metrics() {
    return counters.snapshot(device.toString(), System.nanoTime());
  }

  void resetMetrics() {
    counters.reset(System.nanoTime());
  }

  private boolean isWorkerThread(Thread thread) {
    return thread == stagingThread || thread == submitThread || thread == completionThread;
  }

  private void requireUsable() {
    Throwable failure = terminalFailure;
    if (failure != null) {
      throw new IllegalStateException("The decision pipeline on " + device + " failed", failure);
    }
    if (!running) {
      throw new IllegalStateException("The decision pipeline on " + device + " is closed");
    }
  }

  private void enqueue(Entry entry) {
    requireUsable();
    pendingWork.addAndGet(entry.work.estimatedWorkUnits());
    counters.accepted(System.nanoTime());
    boolean enqueued = false;
    try {
      ready.add(entry);
      enqueued = true;
    } finally {
      if (!enqueued) {
        pendingWork.addAndGet(-entry.work.estimatedWorkUnits());
        counters.finished(entry, System.nanoTime(), false, 0L);
      }
    }
  }

  private synchronized HostReadyCell acquireHostReadyCell(Lease owner) {
    requireUsable();
    if (!hostReadyCapacity.tryAcquire()) {
      return null;
    }
    counters.hostReadyCells(hostReadyCapacity.inUse());
    return new HostReadyCell(this, owner);
  }

  private void releaseHostReadyCell(Lease owner) {
    CompletableFuture<Void> availability = hostReadyCapacity.release(running && terminalFailure == null);
    owner.finishedHostReadyCell();
    if (availability != null) {
      availability.complete(null);
    }
  }

  private void stageLoop() {
    while (running || !ready.isEmpty()) {
      Entry entry = poll(ready);
      if (entry == null) {
        continue;
      }
      Throwable pipelineFailure = terminalFailure;
      if (pipelineFailure != null) {
        failBeforeSubmit(entry, pipelineFailure);
        continue;
      }
      Slot slot = takeFreeSlot();
      if (slot == null) {
        failBeforeSubmit(
            entry,
            terminalFailure != null
                ? terminalFailure
                : new IllegalStateException("The decision pipeline on " + device + " is closed"));
        continue;
      }
      slot.assign(entry);
      entry.hostReadyCell.releaseAfterMove();
      try {
        entry.stageStartedNanos = System.nanoTime();
        entry.work.stage(slot);
        entry.stageFinishedNanos = System.nanoTime();
        putUninterruptibly(submitActions, slot);
      } catch (Throwable failure) {
        try {
          entry.work.fail(failure);
        } finally {
          finish(entry);
          slot.resetAfterFailure();
          putUninterruptibly(freeSlots, slot);
          entry.work.publish();
        }
      }
    }
  }

  /** 計算を出す処理と、終わったスロットの後始末を、同じスレッドで行う。モデルのリソースを取る側と返す側を、1 本のスレッドにそろえるため。 */
  private void submitLoop() {
    while (running || !submitActions.isEmpty()) {
      Slot slot = poll(submitActions);
      if (slot == null) {
        continue;
      }
      if (slot.state == State.COMPLETING) {
        if (slot.failure == null) {
          reclaim(slot);
        } else {
          abort(slot);
        }
        continue;
      }
      Entry entry = slot.entry;
      try {
        entry.submitStartedNanos = System.nanoTime();
        entry.work.submit(slot);
        entry.submitFinishedNanos = System.nanoTime();
        counters.submitted(entry);
        entry.submitted = true;
        slot.markSubmitted();
        putUninterruptibly(submittedSlots, slot);
      } catch (Throwable failure) {
        boolean reusable = false;
        try {
          reusable = entry.work.abort(slot, failure);
        } finally {
          if (reusable) {
            finish(entry);
            slot.resetAfterFailure();
            putUninterruptibly(freeSlots, slot);
          } else {
            try {
              quarantine(slot, failure);
            } finally {
              finish(entry);
            }
          }
          entry.work.publish();
        }
      }
    }
  }

  private void completionLoop() {
    ArrayDeque<Slot> active = new ArrayDeque<>(slotCount);
    while (running || !submittedSlots.isEmpty() || !active.isEmpty()) {
      submittedSlots.drainTo(active);
      if (active.isEmpty()) {
        Slot first = poll(submittedSlots);
        if (first != null) {
          active.addLast(first);
        }
        continue;
      }
      boolean progressed = false;
      int count = active.size();
      for (int index = 0; index < count; index++) {
        Slot slot = active.removeFirst();
        boolean complete;
        try {
          complete = slot.entry.work.isComplete(slot) && slot.completeOutput();
        } catch (Throwable failure) {
          slot.markCompleting(failure);
          putUninterruptibly(submitActions, slot);
          progressed = true;
          continue;
        }
        if (!complete) {
          active.addLast(slot);
          continue;
        }
        progressed = true;
        slot.entry.completionObservedNanos = System.nanoTime();
        slot.markCompleting(null);
        try {
          slot.entry.work.complete(slot);
        } catch (Throwable failure) {
          slot.entry.work.fail(failure);
        } finally {
          slot.entry.composeFinishedNanos = System.nanoTime();
        }
        putUninterruptibly(submitActions, slot);
      }
      if (!progressed) {
        LockSupport.parkNanos(POLL_NANOS);
      }
    }
  }

  private void reclaim(Slot slot) {
    Entry entry = slot.entry;
    long releaseStartedNanos = System.nanoTime();
    // 使い終えた一時の配列は、計算のストリームの上で解放する。ほかのストリームへの依存を増やさないため
    try (var ignored = slot.computeStream.openScope()) {
      entry.work.release(slot);
    } catch (Throwable failure) {
      entry.work.fail(failure);
    } finally {
      finish(entry, true, releaseStartedNanos);
      slot.reset();
      putUninterruptibly(freeSlots, slot);
      entry.work.publish();
    }
  }

  private void abort(Slot slot) {
    Entry entry = slot.entry;
    boolean reusable = false;
    try {
      if (slot.abortOutput(slot.failure)) {
        reusable = entry.work.abort(slot, slot.failure);
      } else {
        entry.work.fail(slot.failure);
      }
    } finally {
      if (reusable) {
        finish(entry);
        slot.resetAfterFailure();
        putUninterruptibly(freeSlots, slot);
      } else {
        try {
          quarantine(slot, slot.failure);
        } finally {
          finish(entry);
        }
      }
      entry.work.publish();
    }
  }

  private void quarantine(Slot slot, Throwable failure) {
    slot.quarantine();
    terminalFailure = failure;
    hostReadyCapacity.fail(failure);
    Entry queued;
    while ((queued = ready.poll()) != null) {
      failBeforeSubmit(queued, failure);
    }
  }

  private void failBeforeSubmit(Entry entry, Throwable failure) {
    try {
      entry.work.fail(failure);
    } finally {
      try {
        entry.hostReadyCell.releaseAfterMove();
      } finally {
        finish(entry);
        entry.work.publish();
      }
    }
  }

  private void finish(Entry entry) {
    finish(entry, false, 0L);
  }

  private void finish(Entry entry, boolean completionObserved, long releaseStartedNanos) {
    counters.finished(entry, System.nanoTime(), completionObserved, releaseStartedNanos);
    long remaining = pendingWork.addAndGet(-entry.work.estimatedWorkUnits());
    entry.owner.finished();
    if (remaining == 0L) {
      synchronized (this) {
        notifyAll();
      }
    }
  }

  private void releaseReference() {
    synchronized (streams) {
      // 隔離したバッチは、モデルの配列をまだ指していることがある。失敗した後は、どのモデルにも権利を返させず、配列を解放させない
      if (terminalFailure != null) {
        throw new IllegalStateException(
            "The decision pipeline on " + device + " retains quarantined GPU resources", terminalFailure);
      }
      references--;
      if (references != 0) {
        return;
      }
      close();
      streams.pipeline = null;
    }
  }

  private void close() {
    running = false;
    hostReadyCapacity.fail(new IllegalStateException("The decision pipeline on " + device + " is closed"));
    stagingThread.interrupt();
    submitThread.interrupt();
    completionThread.interrupt();
    join(stagingThread);
    join(completionThread);
    join(submitThread);
    if (terminalFailure != null) {
      throw new IllegalStateException(
          "The decision pipeline on " + device + " retains quarantined GPU resources", terminalFailure);
    }
    if (freeSlots.size() != slotCount) {
      throw new IllegalStateException("The decision pipeline on " + device + " closed with active slots");
    }
    if (hostReadyCapacity.inUse() != 0) {
      throw new IllegalStateException("The decision pipeline on " + device + " closed with host-ready cells");
    }
    for (Slot slot : freeSlots) {
      slot.close();
    }
    manager.close();
  }

  private Slot takeFreeSlot() {
    while (running && terminalFailure == null) {
      Slot slot = poll(freeSlots);
      if (slot != null) {
        return slot;
      }
    }
    return null;
  }

  private static void join(Thread thread) {
    boolean interrupted = false;
    while (thread.isAlive()) {
      try {
        thread.join();
      } catch (InterruptedException ignored) {
        interrupted = true;
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  private static <T> T poll(BlockingQueue<T> queue) {
    try {
      return queue.poll(10, TimeUnit.MILLISECONDS);
    } catch (InterruptedException ignored) {
      return queue.poll();
    }
  }

  private static <T> void putUninterruptibly(BlockingQueue<T> queue, T value) {
    boolean interrupted = false;
    while (true) {
      try {
        queue.put(value);
        break;
      } catch (InterruptedException ignored) {
        interrupted = true;
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  /** モデルが共有のパイプラインを使う権利。借りている間は共有のリソースを解放しない。 */
  public static final class Lease implements AutoCloseable {
    private final DecisionDevicePipeline pipeline;
    private int pendingBatches;
    private int pendingHostReadyCells;
    private boolean closed;
    private boolean referenceReleased;
    private Throwable releaseFailure;

    private Lease(DecisionDevicePipeline pipeline) {
      this.pipeline = pipeline;
    }

    /** 符号化を終えたバッチを、スロットへ渡すまで置いておくセルを 1 つ取る。空きが無ければ null を返す。 */
    public synchronized HostReadyCell acquireHostReadyCell() {
      requireOpen();
      HostReadyCell cell = pipeline.acquireHostReadyCell(this);
      if (cell != null) {
        pendingHostReadyCells++;
      }
      return cell;
    }

    /**
     * バッチを積むために、セルの所有権を移す。以後は {@link HostReadyCell#abortIfActive} を呼んでもセルは返らない。バッチを積む前に
     * 一度だけ呼ぶ。ほかの借り手のセルなら IllegalArgumentException を投げる。
     */
    public synchronized void claim(HostReadyCell cell) {
      requireOpen();
      cell.moveTo(this);
    }

    /** 所有権を移したセルのバッチを、スロットを待つキューに積む。積めなければセルを返して例外を投げる。 */
    public void submitClaimed(Work work, HostReadyCell cell) {
      boolean pending = false;
      try {
        synchronized (this) {
          requireOpen();
          cell.requireMovedBy(this);
          pendingBatches++;
          pending = true;
        }
        pipeline.enqueue(new Entry(this, work, cell));
      } catch (RuntimeException | Error failure) {
        if (pending) {
          finished();
        }
        cell.releaseAfterMove();
        throw failure;
      }
    }

    /** 推論の受付がホストの容量を数えるのに使うデバイス。{@link InferenceDispatcher.Backend#device} はこれを返す。 */
    public InferenceDispatcher.Device inferenceDevice() {
      return pipeline;
    }

    /**
     * 積んだバッチが全て終わり、セルが全て空くまで待ってから、権利を返す。最後の借り手なら、パイプラインも閉じる。パイプラインの
     * スレッドから呼ぶと IllegalStateException を投げる。
     */
    @Override
    public void close() {
      if (pipeline.isWorkerThread(Thread.currentThread())) {
        throw new IllegalStateException("A decision pipeline lease must be closed outside its worker threads");
      }
      boolean interrupted = false;
      synchronized (this) {
        if (closed) {
          while (!referenceReleased) {
            try {
              wait();
            } catch (InterruptedException ignored) {
              interrupted = true;
            }
          }
          if (interrupted) {
            Thread.currentThread().interrupt();
          }
          rethrow(releaseFailure);
          return;
        }
        closed = true;
        while (pendingBatches != 0 || pendingHostReadyCells != 0) {
          try {
            wait();
          } catch (InterruptedException ignored) {
            interrupted = true;
          }
        }
      }
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
      Throwable failure = null;
      try {
        pipeline.releaseReference();
      } catch (RuntimeException | Error problem) {
        failure = problem;
      }
      synchronized (this) {
        releaseFailure = failure;
        referenceReleased = true;
        notifyAll();
      }
      rethrow(failure);
    }

    private void requireOpen() {
      if (closed) {
        throw new IllegalStateException("The decision pipeline lease is closed");
      }
    }

    private synchronized void finished() {
      pendingBatches--;
      notifyAll();
    }

    private synchronized void finishedHostReadyCell() {
      pendingHostReadyCells--;
      notifyAll();
    }

    private static void rethrow(Throwable failure) {
      if (failure instanceof RuntimeException runtime) {
        throw runtime;
      }
      if (failure instanceof Error error) {
        throw error;
      }
    }
  }

  /** 符号化中か実行待ちのバッチを 1 つ置くセル。所有権は一度だけ次の段へ移せる。 */
  public static final class HostReadyCell {

    private final DecisionDevicePipeline pipeline;
    private final Lease owner;
    private CellState state = CellState.ACTIVE;
    private Runnable slotHandoff;

    private HostReadyCell(DecisionDevicePipeline pipeline, Lease owner) {
      this.pipeline = pipeline;
      this.owner = owner;
    }

    /** バッチがスロットへ移ったときに呼ぶ処理を登録する。推論の受付は、この処理で予約したホストの容量を返す。 */
    public synchronized void onSlotHandoff(Runnable handoff) {
      require(CellState.ACTIVE);
      slotHandoff = handoff;
    }

    /** まだ所有権を移していなければ、セルを返す。 */
    public void abortIfActive() {
      boolean release;
      synchronized (this) {
        release = state == CellState.ACTIVE;
        if (release) {
          state = CellState.RELEASED;
        }
      }
      if (release) {
        pipeline.releaseHostReadyCell(owner);
      }
    }

    private synchronized void moveTo(Lease expectedOwner) {
      if (owner != expectedOwner) {
        throw new IllegalArgumentException("The host-ready cell belongs to another lease");
      }
      require(CellState.ACTIVE);
      state = CellState.MOVED;
    }

    private synchronized void requireMovedBy(Lease expectedOwner) {
      if (owner != expectedOwner) {
        throw new IllegalArgumentException("The host-ready cell belongs to another lease");
      }
      require(CellState.MOVED);
    }

    private void releaseAfterMove() {
      synchronized (this) {
        require(CellState.MOVED);
        state = CellState.RELEASED;
      }
      pipeline.releaseHostReadyCell(owner);
      if (slotHandoff != null) {
        slotHandoff.run();
      }
    }

    private void require(CellState expected) {
      if (state != expected) {
        throw new IllegalStateException("The host-ready cell is " + state + ", expected " + expected);
      }
    }
  }

  /** 符号化中と実行待ちのバッチの数と、空いたときの通知。スロットとは別に数える。 */
  static final class HostReadyCapacity {
    private final int capacity;
    private int inUse;
    private Throwable failure;
    private CompletableFuture<Void> availability = new CompletableFuture<>();

    HostReadyCapacity(int capacity) {
      if (capacity <= 0) {
        throw new IllegalArgumentException("The host-ready capacity must be positive: " + capacity);
      }
      this.capacity = capacity;
    }

    synchronized boolean tryAcquire() {
      if (failure != null) {
        throw new IllegalStateException("The host-ready capacity is unavailable", failure);
      }
      if (inUse >= capacity) {
        return false;
      }
      inUse++;
      return true;
    }

    synchronized CompletableFuture<Void> availability() {
      return availability;
    }

    /**
     * セルを 1 つ返す。
     *
     * @param clean パイプラインが動いていて、失敗していないなら true
     * @return 待っている側へ知らせる通知（呼び出し側が完了させる）。{@code clean} でないか、失敗した後は null
     */
    synchronized CompletableFuture<Void> release(boolean clean) {
      if (inUse == 0) {
        throw new IllegalStateException("The host-ready capacity was released without an owner");
      }
      inUse--;
      if (!clean || failure != null) {
        return null;
      }
      CompletableFuture<Void> signal = availability;
      availability = new CompletableFuture<>();
      return signal;
    }

    /** 以後の取得を拒み、待っている側へ失敗を知らせる。 */
    void fail(Throwable cause) {
      CompletableFuture<Void> signal;
      synchronized (this) {
        if (failure != null) {
          return;
        }
        failure = cause;
        signal = availability;
      }
      signal.completeExceptionally(cause);
    }

    synchronized int inUse() {
      return inUse;
    }
  }

  /** モデルのバッチ 1 つ分の処理。スロットで入力を用意し、計算を出し、完了を見て、リソースを返す。 */
  public interface Work {
    /** このバッチの作業量の見積もり。単位は {@link InferenceDispatcher.Device#pendingWork} と同じ。 */
    long estimatedWorkUnits();

    /** 計測に使う、このバッチの大きさ。 */
    BatchMetadata batchMetadata();

    /** スロットを受け取った直後に、入力を用意するスレッドで呼ぶ。 */
    void stage(Slot slot);

    /** 計算を GPU へ出す。計算を出すスレッドで呼ぶ。 */
    void submit(Slot slot);

    /** モデルがデバイスに出した計算が終わったなら true。出力をホストへ転送し終えたかは、この後にパイプラインが確かめる。 */
    boolean isComplete(Slot slot);

    /** 完了を見るスレッドで結果を組み立てる。 */
    void complete(Slot slot);

    /** 計算を出すスレッドで、一時のリソースを返す。 */
    void release(Slot slot);

    /**
     * 失敗したバッチのリソースを返す。
     *
     * @return スロットを使い回せるなら true。後始末が済んだと確かめられなければ false（スロットを隔離する）
     */
    boolean abort(Slot slot, Throwable failure);

    void fail(Throwable failure);

    /** スロットとデバイスのリソースを全て返した後で、結果を呼び出し側へ知らせる。 */
    void publish();
  }

  /** 1 つのスロットで使い回す、出力のバッファ・イベント・ストリーム。 */
  public static final class Slot {
    private final int index;
    private final PtStream h2dStream;
    private final PtStream computeStream;
    private final PtStream d2hStream;
    private final PtEvent h2dReady;
    private final PtEvent computeReady;
    private final PtEvent outputReady;
    private final PtNDManager manager;
    private PtPinnedBuffer outputBuffer;
    private PtNDArray outputSource;
    private boolean outputStarted;
    private boolean outputRecorded;
    private Entry entry;
    private Throwable failure;
    private State state = State.FREE;

    private Slot(int index, PtNDManager manager, PtStream h2dStream, PtStream computeStream, PtStream d2hStream) {
      this.index = index;
      this.manager = manager;
      this.h2dStream = h2dStream;
      this.computeStream = computeStream;
      this.d2hStream = d2hStream;
      h2dReady = h2dStream.newEvent();
      computeReady = computeStream.newEvent();
      outputReady = d2hStream.newEvent();
    }

    public PtStream h2dStream() {
      return h2dStream;
    }

    public PtEvent h2dReady() {
      return h2dReady;
    }

    public PtEvent computeReady() {
      return computeReady;
    }

    /** 出力の配列を、ホストへ転送するまで持っておく。転送先にはページ固定の FP32 のバッファを用意し、足りなければ取り直す。 */
    public void prepareOutput(PtNDArray source) {
      int elements = Math.toIntExact(source.size());
      if (outputBuffer == null || outputBuffer.size() < elements) {
        if (outputBuffer != null) {
          outputBuffer.close();
        }
        outputBuffer = manager.allocatePinned(elements, DataType.FLOAT32);
      }
      outputSource = source;
    }

    /** ホストへ転送した出力。次のバッチまで有効。 */
    public FloatBuffer outputFloats() {
      return outputBuffer.getByteBuffer().duplicate().order(ByteOrder.nativeOrder()).asFloatBuffer();
    }

    /** 出力の転送は、計算が終わったのを見てから出す。計算を待つ転送を先に積むと、その後に積む入力の転送まで止まるため。 */
    private boolean completeOutput() {
      if (outputSource == null) {
        return true;
      }
      if (!outputStarted) {
        try (var ignored = d2hStream.openScope()) {
          outputStarted = true;
          computeReady.waitOnStream();
          outputSource.enqueueCopyTo(outputBuffer);
          outputReady.record();
          outputRecorded = true;
        }
      }
      return outputReady.isComplete();
    }

    /** 失敗したときも、出した転送が終わるのを待つ。終わったと確かめられなければ false を返し、リソースは解放せずに残す。 */
    private boolean abortOutput(Throwable failure) {
      if (!outputStarted) {
        return true;
      }
      try {
        if (!outputRecorded) {
          try (var ignored = d2hStream.openScope()) {
            outputReady.record();
            outputRecorded = true;
          }
        }
        outputReady.synchronize();
        return true;
      } catch (Throwable synchronizationFailure) {
        failure.addSuppressed(synchronizationFailure);
        return false;
      }
    }

    private void assign(Entry entry) {
      require(State.FREE);
      this.entry = entry;
      state = State.FILLING;
    }

    private void markSubmitted() {
      require(State.FILLING);
      state = State.SUBMITTED;
    }

    private void markCompleting(Throwable failure) {
      require(State.SUBMITTED);
      state = State.COMPLETING;
      this.failure = failure;
    }

    private void reset() {
      require(State.COMPLETING);
      resetAfterFailure();
    }

    private void resetAfterFailure() {
      entry = null;
      failure = null;
      outputSource = null;
      outputStarted = false;
      outputRecorded = false;
      state = State.FREE;
    }

    private void quarantine() {
      entry = null;
      state = State.QUARANTINED;
    }

    private void require(State expected) {
      if (state != expected) {
        throw new IllegalStateException("Decision slot " + index + " is " + state + ", expected " + expected);
      }
    }

    private void close() {
      outputReady.close();
      computeReady.close();
      h2dReady.close();
      if (outputBuffer != null) {
        outputBuffer.close();
        outputBuffer = null;
      }
    }
  }

  /**
   * バッチの大きさ。
   *
   * @param rows バッチの行数
   * @param capacity バッチに入る行数の上限
   */
  public record BatchMetadata(int rows, int capacity) {}

  /**
   * 1 つの計測の区間の集計。
   *
   * @param pipelineIdleNanos パイプラインに処理するバッチが無かった時間。ホストで見た時間で、GPU が空いていた時間ではない
   * @param maxHostReadyCells 同時に使ったセルの数の最大
   * @param total 区間に出した全てのバッチの集計
   */
  public record Metrics(
      String device,
      long elapsedNanos,
      long pipelineIdleNanos,
      int maxHostReadyCells,
      BatchMetrics total) {}

  /**
   * バッチの数と、時間の和（ナノ秒）。平均を出すときは、計算を出すまでの時間を出したバッチの数で割る。完了を見たバッチにだけ足す時間は、
   * 完了を見たバッチの数で割る。
   *
   * @param batches 出したバッチの数
   * @param observedCompletions 完了を見たバッチの数
   * @param completionWaitNanos 出し終えてから、計算と出力の転送の完了をホストが見るまでの時間。GPU の計算の時間ではない
   */
  public record BatchMetrics(
      long batches,
      long rows,
      long capacity,
      long fullBatches,
      long readyWaitNanos,
      long stageNanos,
      long submitQueueWaitNanos,
      long submitNanos,
      long observedCompletions,
      long completionWaitNanos,
      long composeNanos,
      long reclaimWaitNanos,
      long releaseNanos,
      long roundTripNanos) {}

  private static final class Entry {
    final Lease owner;
    final Work work;
    final HostReadyCell hostReadyCell;
    final BatchMetadata metadata;
    final long enqueuedNanos = System.nanoTime();
    long stageStartedNanos;
    long stageFinishedNanos;
    long submitStartedNanos;
    long submitFinishedNanos;
    long completionObservedNanos;
    long composeFinishedNanos;
    boolean submitted;

    Entry(Lease owner, Work work, HostReadyCell hostReadyCell) {
      this.owner = owner;
      this.work = work;
      this.hostReadyCell = hostReadyCell;
      metadata = work.batchMetadata();
    }
  }

  /** 同期もテンソルの読み出しも足さずに、ホストで見たバッチを数える。 */
  private static final class Counters {
    private Group total = new Group();
    private long started = System.nanoTime();
    private long idleStarted = started;
    private long idleNanos;
    private int pending;
    private int maxHostReadyCells;

    synchronized void accepted(long now) {
      if (pending++ == 0) {
        idleNanos += now - idleStarted;
      }
    }

    synchronized void submitted(Entry entry) {
      total.batches++;
      total.rows += entry.metadata.rows();
      total.capacity += entry.metadata.capacity();
      if (entry.metadata.rows() == entry.metadata.capacity()) {
        total.fullBatches++;
      }
      total.readyWaitNanos += entry.stageStartedNanos - entry.enqueuedNanos;
      total.stageNanos += entry.stageFinishedNanos - entry.stageStartedNanos;
      total.submitQueueWaitNanos += entry.submitStartedNanos - entry.stageFinishedNanos;
      total.submitNanos += entry.submitFinishedNanos - entry.submitStartedNanos;
    }

    synchronized void finished(Entry entry, long now, boolean completionObserved, long releaseStartedNanos) {
      if (entry.submitted) {
        total.roundTripNanos += now - entry.enqueuedNanos;
        if (completionObserved) {
          total.observedCompletions++;
          total.completionWaitNanos += entry.completionObservedNanos - entry.submitFinishedNanos;
          total.composeNanos += entry.composeFinishedNanos - entry.completionObservedNanos;
          total.reclaimWaitNanos += releaseStartedNanos - entry.composeFinishedNanos;
          total.releaseNanos += now - releaseStartedNanos;
        }
      }
      if (--pending == 0) {
        idleStarted = now;
      }
    }

    synchronized void hostReadyCells(int inUse) {
      maxHostReadyCells = Math.max(maxHostReadyCells, inUse);
    }

    synchronized void reset(long now) {
      if (pending != 0) {
        throw new IllegalStateException("Inference metrics need a drained pipeline");
      }
      total = new Group();
      started = now;
      idleStarted = now;
      idleNanos = 0;
      maxHostReadyCells = 0;
    }

    synchronized Metrics snapshot(String device, long now) {
      return new Metrics(
          device,
          now - started,
          idleNanos + (pending == 0 ? now - idleStarted : 0),
          maxHostReadyCells,
          total.metrics());
    }

    private static final class Group {
      long batches;
      long rows;
      long capacity;
      long fullBatches;
      long readyWaitNanos;
      long stageNanos;
      long submitQueueWaitNanos;
      long submitNanos;
      long observedCompletions;
      long completionWaitNanos;
      long composeNanos;
      long reclaimWaitNanos;
      long releaseNanos;
      long roundTripNanos;

      BatchMetrics metrics() {
        return new BatchMetrics(
            batches,
            rows,
            capacity,
            fullBatches,
            readyWaitNanos,
            stageNanos,
            submitQueueWaitNanos,
            submitNanos,
            observedCompletions,
            completionWaitNanos,
            composeNanos,
            reclaimWaitNanos,
            releaseNanos,
            roundTripNanos);
      }
    }
  }

  private enum CellState {
    ACTIVE,
    MOVED,
    RELEASED
  }

  private enum State {
    FREE,
    FILLING,
    SUBMITTED,
    COMPLETING,
    QUARANTINED
  }
}
