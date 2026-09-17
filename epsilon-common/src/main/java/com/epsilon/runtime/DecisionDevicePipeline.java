package com.epsilon.runtime;

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
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * GPUごとに、ホストからの入力転送・順伝播・ホストへの出力転送をまとめて実行する。
 *
 * <p>各モデルの評価器は{@link Lease}を介して利用する。同じGPU上のモデルは、処理を投入するスレッドと役割別の3本のストリームを共有する。 通常の完了判定は{@link
 * PtEvent#isComplete()}を使い、ホストでGPUの完了を待つ処理は終了時と失敗時の回収に限る。
 */
public final class DecisionDevicePipeline implements AutoCloseable, InferenceDispatcher.Device {

  private static final long POLL_NANOS = TimeUnit.MICROSECONDS.toNanos(50);

  private final Device device;
  private final int slotCount;
  private final int readyCapacity;
  private final PtNDManager manager;
  private final DecisionDeviceStreams streams;
  private final PtStream h2dStream;
  private final PtStream computeStream;
  private final PtStream d2hStream;
  private final HostReadyCapacity hostReadyCapacity;
  private final BlockingQueue<Entry> ready;
  private final BlockingQueue<Slot> freeSlots;
  private final BlockingQueue<Slot> submitActions = new LinkedBlockingQueue<>();
  private final BlockingQueue<Slot> submittedSlots = new LinkedBlockingQueue<>();
  private final AtomicLong pendingWork = new AtomicLong();
  private final InferenceBatchMetrics metrics = new InferenceBatchMetrics();
  private final Thread stagingThread;
  private final Thread submitThread;
  private final Thread completionThread;
  private int references;
  private volatile boolean running = true;
  private volatile Throwable terminalFailure;

  /** 指定GPUの共有パイプラインを取得する。同じデバイスに異なる実行容量を指定すると例外を送出する。 */
  public static Lease acquire(
      Device device, int slotCount, int readyCapacity, DecisionDeviceStreams streams) {
    if (!device.equals(streams.device())) {
      throw new IllegalArgumentException("Decision pipeline stream device differs: " + device);
    }
    synchronized (streams) {
      DecisionDevicePipeline pipeline = streams.pipeline;
      if (pipeline == null) {
        pipeline = new DecisionDevicePipeline(device, slotCount, readyCapacity, streams);
        streams.pipeline = pipeline;
      } else if (!pipeline.running || pipeline.terminalFailure != null) {
        throw new IllegalStateException(
            "Decision pipeline is unavailable after a terminal failure: " + device,
            pipeline.terminalFailure);
      } else if (pipeline.slotCount != slotCount || pipeline.readyCapacity != readyCapacity) {
        throw new IllegalStateException(
            "Decision pipeline configuration differs on "
                + device
                + ": active="
                + pipeline.slotCount
                + '/'
                + pipeline.readyCapacity
                + " requested="
                + slotCount
                + '/'
                + readyCapacity);
      }
      pipeline.references++;
      return new Lease(pipeline);
    }
  }

  /** パイプラインの解放を確認し、使用中や失敗後に未回収のストリームを次の処理に渡さない。 */
  static void requireReleased(DecisionDeviceStreams streams) {
    synchronized (streams) {
      DecisionDevicePipeline pipeline = streams.pipeline;
      if (pipeline != null) {
        throw new IllegalStateException(
            "Decision execution context still has an inference pipeline: " + streams.device(),
            pipeline.terminalFailure);
      }
    }
  }

  private DecisionDevicePipeline(
      Device device, int slotCount, int readyCapacity, DecisionDeviceStreams streams) {
    this.device = device;
    this.slotCount = slotCount;
    this.readyCapacity = readyCapacity;
    manager = (PtNDManager) NDManager.newBaseManager(device);
    this.streams = streams;
    h2dStream = streams.h2d();
    computeStream = streams.compute();
    d2hStream = streams.d2h();
    hostReadyCapacity = new HostReadyCapacity(readyCapacity);
    ready = new ArrayBlockingQueue<>(readyCapacity);
    freeSlots = new ArrayBlockingQueue<>(slotCount);
    for (int index = 0; index < slotCount; index++) {
      freeSlots.add(new Slot(index, manager, h2dStream, computeStream, d2hStream));
    }
    String suffix = device.getDeviceId() + "-" + Integer.toHexString(System.identityHashCode(this));
    stagingThread = newThread("epsilon-decision-stage-" + suffix, this::stageLoop);
    submitThread = newThread("epsilon-decision-submit-" + suffix, this::submitLoop);
    completionThread = newThread("epsilon-decision-complete-" + suffix, this::completionLoop);
    stagingThread.start();
    submitThread.start();
    completionThread.start();
  }

  private static Thread newThread(String name, Runnable target) {
    Thread thread = new Thread(target, name);
    thread.setDaemon(true);
    return thread;
  }

  private boolean isWorkerThread(Thread thread) {
    return thread == stagingThread || thread == submitThread || thread == completionThread;
  }

  private void enqueue(Entry entry) {
    Throwable failure = terminalFailure;
    if (failure != null) {
      throw new IllegalStateException("Decision device pipeline failed: " + device, failure);
    }
    if (!running) {
      throw new IllegalStateException("Decision device pipeline is closed: " + device);
    }
    pendingWork.addAndGet(entry.work.estimatedWorkUnits());
    metrics.accepted(System.nanoTime());
    boolean enqueued = false;
    try {
      ready.add(entry);
      enqueued = true;
    } finally {
      if (!enqueued) {
        pendingWork.addAndGet(-entry.work.estimatedWorkUnits());
        metrics.finished(entry.metadata, false, 0, System.nanoTime(), false, 0, 0, 0, 0);
      }
    }
  }

  /** 符号化が完了したバッチを、実行先へ渡すまで保持する領域を1件確保する。 */
  private synchronized HostReadyCell acquireHostReadyCell(Lease owner) {
    Throwable failure = terminalFailure;
    if (failure != null) {
      throw new IllegalStateException("Decision device pipeline failed: " + device, failure);
    }
    if (!running) {
      throw new IllegalStateException("Decision device pipeline is closed: " + device);
    }
    if (!hostReadyCapacity.tryAcquire()) {
      return null;
    }
    metrics.hostReadyCells(hostReadyCapacity.inUse());
    return new HostReadyCell(this, owner);
  }

  private CompletableFuture<Void> hostReadyCellAvailable() {
    return hostReadyCapacity.availability();
  }

  private void releaseHostReadyCell(Lease owner) {
    CompletableFuture<Void> availability =
        hostReadyCapacity.release(running && terminalFailure == null);
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
                : new IllegalStateException("Decision device pipeline is closed: " + device));
        continue;
      }
      slot.assign(entry);
      entry.hostReadyCell.releaseAfterMove();
      try {
        entry.stageStartedNanos = System.nanoTime();
        try (var attached = InferenceProfile.attach(entry.work.profile());
            var profiled = InferenceProfile.section("host.stage")) {
          entry.work.stage(slot);
        }
        entry.stageFinishedNanos = System.nanoTime();
        putUninterruptibly(submitActions, slot);
      } catch (Throwable failure) {
        try {
          entry.work.fail(failure);
        } finally {
          finish(entry);
          slot.resetAfterFailure();
          putUninterruptibly(freeSlots, slot);
          publish(entry);
        }
      }
    }
  }

  /** Fusion 利用権の取得と返却を必ず同じスレッドへ集約します。 */
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
      try {
        slot.entry.submitStartedNanos = System.nanoTime();
        try (var attached = InferenceProfile.attach(slot.entry.work.profile())) {
          slot.entry.work.submit(slot);
        }
        Entry entry = slot.entry;
        entry.submitFinishedNanos = System.nanoTime();
        metrics.submitted(
            entry.metadata,
            entry.stageStartedNanos - entry.enqueuedNanos,
            entry.stageFinishedNanos - entry.stageStartedNanos,
            entry.submitStartedNanos - entry.stageFinishedNanos,
            entry.submitFinishedNanos - entry.submitStartedNanos);
        entry.submitted = true;
        slot.markSubmitted();
        putUninterruptibly(submittedSlots, slot);
      } catch (Throwable failure) {
        Entry entry = slot.entry;
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
          publish(entry);
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
      boolean completed = false;
      int count = active.size();
      for (int index = 0; index < count; index++) {
        Slot slot = active.removeFirst();
        boolean complete;
        try {
          complete = slot.entry.work.isComplete(slot) && slot.completeOutput();
        } catch (Throwable failure) {
          slot.markCompleting(failure);
          putUninterruptibly(submitActions, slot);
          completed = true;
          continue;
        }
        if (!complete) {
          active.addLast(slot);
          continue;
        }
        completed = true;
        slot.entry.completionObservedNanos = System.nanoTime();
        slot.markCompleting();
        try {
          try (var attached = InferenceProfile.attach(slot.entry.work.profile());
              var profiled = InferenceProfile.section("host.compose")) {
            slot.entry.work.complete(slot);
          }
        } catch (Throwable failure) {
          slot.entry.work.fail(failure);
        } finally {
          slot.entry.composeFinishedNanos = System.nanoTime();
        }
        putUninterruptibly(submitActions, slot);
      }
      if (!completed) {
        java.util.concurrent.locks.LockSupport.parkNanos(POLL_NANOS);
      }
    }
  }

  private void reclaim(Slot slot) {
    Entry entry = slot.entry;
    long releaseStartedNanos = System.nanoTime();
    // 完了済みの一時テンソルへ、無関係な既定のストリームの依存を追加しない。
    try (var ignored = slot.computeStream().openScope()) {
      entry.work.release(slot);
    } catch (Throwable failure) {
      entry.work.fail(failure);
    } finally {
      finish(entry, true, releaseStartedNanos);
      slot.reset();
      putUninterruptibly(freeSlots, slot);
      publish(entry);
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
      publish(entry);
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
        publish(entry);
      }
    }
  }

  /** 実行枠とデバイス資源をすべて返した後でだけ、呼び出し側へ結果を公開します。 */
  private static void publish(Entry entry) {
    entry.work.publish();
  }

  long pendingEstimatedWork() {
    return pendingWork.get();
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

  private void finish(Entry entry) {
    finish(entry, false, 0L);
  }

  private void finish(Entry entry, boolean completionObserved, long releaseStartedNanos) {
    long now = System.nanoTime();
    metrics.finished(
        entry.metadata,
        entry.submitted,
        now - entry.enqueuedNanos,
        now,
        completionObserved,
        completionObserved ? entry.completionObservedNanos - entry.submitFinishedNanos : 0L,
        completionObserved ? entry.composeFinishedNanos - entry.completionObservedNanos : 0L,
        completionObserved ? releaseStartedNanos - entry.composeFinishedNanos : 0L,
        completionObserved ? now - releaseStartedNanos : 0L);
    long remaining = pendingWork.addAndGet(-entry.work.estimatedWorkUnits());
    entry.owner.finished();
    if (remaining == 0L) {
      synchronized (this) {
        notifyAll();
      }
    }
  }

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
    return metrics.snapshot(device.toString(), System.nanoTime());
  }

  void resetMetrics() {
    metrics.reset(System.nanoTime());
  }

  private void releaseReference() {
    synchronized (streams) {
      // 解放を保留した処理はモデル所有テンソルを参照し得るため、すべての利用参照が持つモデルの解放を止める。
      if (terminalFailure != null) {
        throw new IllegalStateException(
            "Decision pipeline retained quarantined GPU resources: " + device, terminalFailure);
      }
      references--;
      if (references != 0) {
        return;
      }
      close();
      streams.pipeline = null;
    }
  }

  @Override
  public void close() {
    running = false;
    hostReadyCapacity.fail(
        new IllegalStateException("Decision device pipeline is closed: " + device));
    stagingThread.interrupt();
    submitThread.interrupt();
    completionThread.interrupt();
    join(stagingThread);
    join(completionThread);
    join(submitThread);
    if (terminalFailure != null) {
      throw new IllegalStateException(
          "Decision pipeline retained quarantined GPU resources: " + device, terminalFailure);
    }
    if (freeSlots.size() != slotCount) {
      throw new IllegalStateException("Decision pipeline closed with active slots: " + device);
    }
    if (hostReadyCapacity.inUse() != 0) {
      throw new IllegalStateException("Decision pipeline closed with host-ready cells: " + device);
    }
    for (Slot slot : freeSlots) {
      slot.close();
    }
    manager.close();
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

  private Slot takeFreeSlot() {
    while (running && terminalFailure == null) {
      Slot slot = poll(freeSlots);
      if (slot != null) {
        return slot;
      }
    }
    return null;
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

  /** 評価器から共有パイプラインを利用するための参照。参照数を管理し、利用中のモデルがある間は共有資源を解放しない。 */
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

    public HostReadyCell acquireHostReadyCell() {
      synchronized (this) {
        if (closed) {
          throw new IllegalStateException("Decision pipeline lease is closed");
        }
        HostReadyCell cell = pipeline.acquireHostReadyCell(this);
        if (cell != null) {
          pendingHostReadyCells++;
        }
        return cell;
      }
    }

    public synchronized CompletableFuture<Void> hostReadyCellAvailable() {
      if (closed) {
        throw new IllegalStateException("Decision pipeline lease is closed");
      }
      return pipeline.hostReadyCellAvailable();
    }

    public void claim(HostReadyCell hostReadyCell) {
      synchronized (this) {
        if (closed) {
          throw new IllegalStateException("Decision pipeline lease is closed");
        }
        if (hostReadyCell == null) {
          throw new NullPointerException("hostReadyCell");
        }
        hostReadyCell.moveTo(this);
      }
    }

    public void submitClaimed(Work work, HostReadyCell hostReadyCell) {
      boolean pending = false;
      try {
        synchronized (this) {
          if (closed) {
            throw new IllegalStateException("Decision pipeline lease is closed");
          }
          if (work == null) {
            throw new NullPointerException("work");
          }
          hostReadyCell.requireMovedBy(this);
          pendingBatches++;
          pending = true;
        }
        pipeline.enqueue(new Entry(this, work, hostReadyCell));
      } catch (RuntimeException | Error failure) {
        if (pending) {
          finished();
        }
        hostReadyCell.releaseAfterMove();
        throw failure;
      }
    }

    public void abortClaimed(HostReadyCell hostReadyCell) {
      hostReadyCell.requireMovedBy(this);
      hostReadyCell.releaseAfterMove();
    }

    public synchronized long pendingEstimatedWork() {
      return pipeline.pendingEstimatedWork();
    }

    public InferenceDispatcher.Device inferenceDevice() {
      return pipeline;
    }

    /** 同じ物理GPUへ投入済みの全モデルバッチが枠を返すまで待ちます。 */
    public void awaitDeviceIdle() {
      pipeline.awaitIdle();
    }

    private synchronized void finished() {
      pendingBatches--;
      notifyAll();
    }

    private synchronized void finishedHostReadyCell() {
      pendingHostReadyCells--;
      notifyAll();
    }

    @Override
    public void close() {
      if (pipeline.isWorkerThread(Thread.currentThread())) {
        throw new IllegalStateException(
            "Decision pipeline lease must be closed outside its worker callback");
      }
      boolean interrupted = false;
      synchronized (this) {
        if (referenceReleased) {
          rethrow(releaseFailure);
          return;
        }
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
      }
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
      rethrow(releaseReferenceAfterPending());
    }

    private Throwable releaseReferenceAfterPending() {
      boolean interrupted = false;
      synchronized (this) {
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
      return releaseReference();
    }

    private Throwable releaseReference() {
      Throwable failure = null;
      try {
        pipeline.releaseReference();
      } catch (RuntimeException | Error releaseFailure) {
        failure = releaseFailure;
      }
      synchronized (this) {
        this.releaseFailure = failure;
        referenceReleased = true;
        notifyAll();
      }
      return failure;
    }

    private static void rethrow(Throwable failure) {
      if (failure instanceof RuntimeException runtimeFailure) {
        throw runtimeFailure;
      }
      if (failure instanceof Error error) {
        throw error;
      }
    }
  }

  /** 符号化中または実行待ちのバッチを1件保持する。バッチの所有権は一度だけ次の処理へ移せる。 */
  public static final class HostReadyCell implements AutoCloseable {

    private final DecisionDevicePipeline pipeline;
    private final Lease owner;
    private CellState state = CellState.ACTIVE;
    private Runnable slotHandoff;

    private HostReadyCell(DecisionDevicePipeline pipeline, Lease owner) {
      this.pipeline = pipeline;
      this.owner = owner;
    }

    private synchronized void moveTo(Lease expectedOwner) {
      if (owner != expectedOwner) {
        throw new IllegalArgumentException("Decision host-ready cell belongs to another lease");
      }
      require(CellState.ACTIVE);
      state = CellState.MOVED;
    }

    /** 符号化が完了したバッチの共有ホスト予約を、枠への移譲後に返す。 */
    public synchronized void onSlotHandoff(Runnable handoff) {
      require(CellState.ACTIVE);
      slotHandoff = handoff;
    }

    private synchronized void requireMovedBy(Lease expectedOwner) {
      if (owner != expectedOwner) {
        throw new IllegalArgumentException("Decision host-ready cell belongs to another lease");
      }
      require(CellState.MOVED);
    }

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

    private void releaseAfterMove() {
      synchronized (this) {
        require(CellState.MOVED);
        state = CellState.RELEASED;
      }
      pipeline.releaseHostReadyCell(owner);
      if (slotHandoff != null) slotHandoff.run();
    }

    @Override
    public void close() {
      synchronized (this) {
        require(CellState.ACTIVE);
        state = CellState.RELEASED;
      }
      pipeline.releaseHostReadyCell(owner);
    }

    private void require(CellState expected) {
      if (state != expected) {
        throw new IllegalStateException(
            "Decision host-ready cell is " + state + ", expected " + expected);
      }
    }
  }

  /** ホスト側で構築中または実行待ちのバッチ数と、容量の解放通知を管理する。デバイスの実行枠とは別に数える。 */
  static final class HostReadyCapacity {
    private final int capacity;
    private int inUse;
    private Throwable failure;
    private CompletableFuture<Void> availability = new CompletableFuture<>();

    HostReadyCapacity(int readyCapacity) {
      if (readyCapacity <= 0) {
        throw new IllegalArgumentException("Decision host-ready capacity must be positive");
      }
      capacity = readyCapacity;
    }

    synchronized boolean tryAcquire() {
      if (failure != null) {
        throw new IllegalStateException("Decision host-ready capacity is unavailable", failure);
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

    CompletableFuture<Void> release(boolean clean) {
      CompletableFuture<Void> signal = null;
      synchronized (this) {
        if (inUse == 0) {
          throw new IllegalStateException(
              "Decision host-ready capacity was released without owner");
        }
        inUse--;
        if (clean && failure == null) {
          signal = availability;
          availability = new CompletableFuture<>();
        }
      }
      return signal;
    }

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

  /** 実行枠で入力の準備、演算の開始、完了確認、資源の回収を行うモデルのバッチ。 */
  public interface Work {
    /** 単一バッチを診断する場合に、計測区間と出力先を返す。通常の推論ではnullを返す。 */
    default InferenceProfile profile() {
      return null;
    }

    /** 診断対象の推論バッチだけが、系列に閉じた識別子と実行容量を渡す。 */
    default BatchMetadata batchMetadata() {
      return null;
    }

    long estimatedWorkUnits();

    void stage(Slot slot);

    void submit(Slot slot);

    /** モデルが発行したデバイス処理の完了を判定する。登録済み出力のホストへの転送は、この判定後に共通実装が確認する。 */
    boolean isComplete(Slot slot);

    void complete(Slot slot);

    void release(Slot slot);

    boolean abort(Slot slot, Throwable failure);

    void fail(Throwable failure);

    void publish();
  }

  /** 一つの実行枠で再利用するホストバッファ、イベント、ストリームへの参照を保持する。 */
  public static final class Slot implements AutoCloseable {
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

    private Slot(
        int index,
        PtNDManager manager,
        PtStream h2dStream,
        PtStream computeStream,
        PtStream d2hStream) {
      this.index = index;
      this.manager = manager;
      this.h2dStream = h2dStream;
      this.computeStream = computeStream;
      this.d2hStream = d2hStream;
      h2dReady = h2dStream.newEvent();
      computeReady = computeStream.newEvent();
      outputReady = d2hStream.newEvent();
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

    private void markCompleting() {
      require(State.SUBMITTED);
      state = State.COMPLETING;
    }

    private void markCompleting(Throwable failure) {
      markCompleting();
      this.failure = failure;
    }

    private void reset() {
      require(State.COMPLETING);
      entry = null;
      failure = null;
      outputSource = null;
      outputStarted = false;
      outputRecorded = false;
      state = State.FREE;
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
        throw new IllegalStateException(
            "Decision slot " + index + " is " + state + ", expected " + expected);
      }
    }

    public PtStream h2dStream() {
      return h2dStream;
    }

    public PtStream computeStream() {
      return computeStream;
    }

    public PtEvent h2dReady() {
      return h2dReady;
    }

    public PtEvent computeReady() {
      return computeReady;
    }

    /** 出力テンソルを回収まで借用し、演算完了後の転送先だけを用意します。 */
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

    /** 演算待ちのD2Hを先にコピーエンジンへ発行せず、後続H2Dの進行を保ちます。 */
    private boolean completeOutput() {
      if (outputSource == null) {
        return true;
      }
      if (!outputStarted) {
        try (var attached = InferenceProfile.attach(entry.work.profile());
            var ignored = d2hStream.openScope();
            var profiled = InferenceProfile.section("transfer.d2h")) {
          outputStarted = true;
          computeReady.waitOnStream();
          outputSource.enqueueCopyTo(outputBuffer);
          outputReady.record();
          outputRecorded = true;
        }
      }
      return outputReady.isComplete();
    }

    /** 途中まで発行したD2Hも失敗経路で回収し、完了を証明できなければ資源を保持します。 */
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

    public FloatBuffer outputFloats() {
      return outputBuffer
          .getByteBuffer()
          .duplicate()
          .order(ByteOrder.nativeOrder())
          .asFloatBuffer();
    }

    @Override
    public void close() {
      outputReady.close();
      computeReady.close();
      h2dReady.close();
      if (outputBuffer != null) {
        outputBuffer.close();
        outputBuffer = null;
      }
    }
  }

  /** 識別子は実行中の参照だけを保持し、系列のスキーマを解釈しない。 */
  public record BatchMetadata(
      Object model, Object bucket, Object resultKind, int rows, int capacity) {}

  /** パイプラインが空だったホスト時間。GPU カーネルの空転時間とは異なる。 */
  public record Metrics(
      String device,
      long elapsedNanos,
      long pipelineIdleNanos,
      int maxHostReadyCells,
      List<BatchMetrics> batches) {}

  /**
   * 同じモデル・入力区分・結果種別について、投入したバッチの行数・容量・処理時間を集計する。
   *
   * <p>処理投入までの平均時間は{@code batches}、完了確認以降の平均時間は{@code observedCompletions}を分母にして求める。 {@code
   * completionWaitNanos}は、処理投入の終了から、演算と必要な出力転送の完了をホストが確認するまでの経過時間である。GPUの実行時間そのものではない。
   */
  public record BatchMetrics(
      String model,
      String bucket,
      String resultKind,
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
