package com.epsilon.runtime;

import java.util.ArrayDeque;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/** 入力の符号化前にホスト側の容量を予約し、符号化完了後に実行先を選ぶ。 */
public final class InferenceDispatcher implements AutoCloseable {
  private final ExecutorService dispatcher =
      Executors.newSingleThreadExecutor(
          Thread.ofPlatform().daemon().name("epsilon-inference-dispatch").factory());
  // 利用側のコールバックから同期推論や終了処理を呼べるよう、結果の通知は要求を投入するスレッドとは別のスレッドで実行する。
  private static final Executor DELIVERY =
      task -> Thread.ofVirtual().name("epsilon-inference-result").start(task);
  private final AtomicBoolean scheduled = new AtomicBoolean();
  private final Map<Object, Lane> lanes = new LinkedHashMap<>();
  private final Set<Registration<?, ?>> registrations = new LinkedHashSet<>();
  private final Set<Reservation<?, ?>> host = new LinkedHashSet<>();
  private final ArrayDeque<Reservation<?, ?>> ready = new ArrayDeque<>();
  private final ArrayDeque<Reservation<?, ?>> waiting = new ArrayDeque<>();
  private CompletableFuture<Void> availability = new CompletableFuture<>();
  private int capacity;
  private int peakHostReservations;
  private int pending;
  private boolean closed;

  /** 同じ物理デバイスの容量は、登録するモデル数にかかわらず一回だけ計上する。 */
  public synchronized <B, R> Registration<B, R> register(List<Backend<B, R>> backends) {
    if (closed) throw new IllegalStateException("Inference dispatcher is closed");
    if (backends.isEmpty()) throw new IllegalArgumentException("Inference requires a backend");
    for (Backend<B, R> backend : backends) {
      Device device = backend.device();
      Lane lane = lanes.get(device.key());
      if (lane == null) {
        lanes.put(device.key(), new Lane(device));
        capacity += device.hostCapacity();
      } else if (lane.device != device) {
        throw new IllegalArgumentException("Inference device must be shared: " + device.key());
      }
    }
    Registration<B, R> registration = new Registration<>(this, backends);
    registrations.add(registration);
    return registration;
  }

  private synchronized <B, R> Attempt<B, R> reserve(Registration<B, R> owner) {
    requireAccepting(owner);
    if (host.size() >= capacity) return new Attempt<>(null, availability);
    Reservation<B, R> reservation = new Reservation<>(owner);
    host.add(reservation);
    peakHostReservations = Math.max(peakHostReservations, host.size());
    owner.pending++;
    pending++;
    return new Attempt<>(reservation, null);
  }

  private void requireAccepting(Registration<?, ?> owner) {
    if (!owner.accepting) throw new IllegalStateException("Inference registration is closed");
    Throwable failure = deviceFailure(owner);
    if (failure != null) throw new IllegalStateException("Inference device failed", failure);
  }

  private Throwable deviceFailure(Registration<?, ?> owner) {
    for (Backend<?, ?> backend : owner.backends) {
      Throwable failure = lanes.get(backend.device().key()).failure;
      if (failure != null) return failure;
    }
    return null;
  }

  private synchronized <B, R> CompletableFuture<R> submit(Registration<B, R> owner, B batch) {
    requireAccepting(owner);
    Reservation<B, R> reservation = new Reservation<>(owner);
    reservation.batch = batch;
    reservation.state = State.WAITING;
    owner.pending++;
    pending++;
    waiting.addLast(reservation);
    admitWaiting();
    signal();
    return reservation.result;
  }

  /** 新規要求の受付を停止した後も、受け付け済みの直接入力は容量が空くたびに実行準備へ進める。 */
  private void admitWaiting() {
    while (host.size() < capacity && !waiting.isEmpty()) {
      Reservation<?, ?> reservation = waiting.removeFirst();
      reservation.state = State.READY;
      host.add(reservation);
      peakHostReservations = Math.max(peakHostReservations, host.size());
      ready.addLast(reservation);
    }
  }

  private synchronized boolean needsWork(Registration<?, ?> candidate) {
    for (Lane lane : lanes.values()) lane.supply = lane.device.suppliedBatches();
    // 実行可能なデバイスが少ないバッチから仮に割り当てる。GPU 0だけで実行できるバッチを、GPU 1の予定負荷に含めない。
    for (int width = 1; width <= lanes.size(); width++) {
      for (Reservation<?, ?> reservation : host) {
        if ((reservation.state == State.OPEN || reservation.state == State.READY)
            && reservation.owner.backends.size() == width) {
          Lane selected = null;
          for (Backend<?, ?> backend : reservation.owner.backends) {
            Lane lane = lanes.get(backend.device().key());
            if (selected == null || lane.supply < selected.supply) selected = lane;
          }
          selected.supply++;
        }
      }
    }
    for (Backend<?, ?> backend : candidate.backends) {
      if (lanes.get(backend.device().key()).supply < backend.device().slots()) return true;
    }
    return false;
  }

  private synchronized <B, R> CompletableFuture<R> submit(Reservation<B, R> reservation, B batch) {
    reservation.requireOpen();
    reservation.batch = batch;
    reservation.state = State.READY;
    ready.addLast(reservation);
    signal();
    return reservation.result;
  }

  /** 実行先は容量が不足していればnullを返す。一つの実行先が満杯でも、別のデバイスへ投入できるバッチの処理は続ける。 */
  private synchronized void pump() {
    var iterator = waiting.iterator();
    while (iterator.hasNext()) {
      Reservation<?, ?> reservation = iterator.next();
      Throwable failure = deviceFailure(reservation.owner);
      if (failure != null) {
        iterator.remove();
        finish(reservation, null, failure);
      }
    }
    int remaining = ready.size();
    while (remaining-- > 0) {
      Reservation<?, ?> reservation = ready.removeFirst();
      if (!dispatch(reservation)) ready.addLast(reservation);
    }
  }

  private synchronized void signal() {
    if (closed) return;
    if (scheduled.compareAndSet(false, true))
      dispatcher.execute(
          () -> {
            scheduled.set(false);
            pump();
          });
  }

  private <B, R> boolean dispatch(Reservation<B, R> reservation) {
    List<Backend<B, R>> backends = reservation.owner.backends;
    Throwable deviceFailure = deviceFailure(reservation.owner);
    if (deviceFailure != null) {
      finish(reservation, null, deviceFailure);
      return true;
    }
    int first = Math.floorMod(reservation.owner.nextDevice++, backends.size());
    int preferred = first;
    long minimum = backends.get(first).device().pendingWork();
    for (int offset = 1; offset < backends.size(); offset++) {
      int index = (first + offset) % backends.size();
      long work = backends.get(index).device().pendingWork();
      if (work < minimum) {
        minimum = work;
        preferred = index;
      }
    }
    reservation.state = State.DISPATCHED;
    for (int offset = 0; offset < backends.size(); offset++) {
      Backend<B, R> backend = backends.get((preferred + offset) % backends.size());
      CompletableFuture<Void> changed;
      CompletableFuture<R> completion;
      try {
        changed = backend.device().capacityAvailable();
        completion = backend.trySubmit(reservation.batch, () -> releaseHost(reservation));
      } catch (RuntimeException | Error failure) {
        finish(reservation, null, failure);
        return true;
      }
      if (completion != null) {
        completion.whenCompleteAsync(
            (result, failure) -> finish(reservation, result, failure), dispatcher);
        return true;
      }
      Lane lane = lanes.get(backend.device().key());
      if (lane.notification != changed) {
        lane.notification = changed;
        changed.whenComplete(
            (ignored, failure) -> {
              synchronized (InferenceDispatcher.this) {
                if (failure != null) lane.failure = failure;
              }
              signal();
            });
      }
    }
    reservation.state = State.READY;
    return false;
  }

  private void releaseHost(Reservation<?, ?> reservation) {
    CompletableFuture<Void> available = null;
    synchronized (this) {
      if (!host.remove(reservation)) return;
      admitWaiting();
      if (!ready.isEmpty()) signal();
      if (host.size() < capacity) {
        available = availability;
        availability = new CompletableFuture<>();
      }
    }
    if (available != null) {
      CompletableFuture<Void> signal = available;
      DELIVERY.execute(() -> signal.complete(null));
    }
  }

  private <R> void finish(Reservation<?, R> reservation, R result, Throwable failure) {
    releaseHost(reservation);
    synchronized (this) {
      reservation.state = State.FINISHED;
      reservation.owner.pending--;
      pending--;
      notifyAll();
    }
    DELIVERY.execute(
        () -> {
          if (failure == null) reservation.result.complete(result);
          else reservation.result.completeExceptionally(failure);
        });
  }

  private void abort(Reservation<?, ?> reservation) {
    synchronized (this) {
      reservation.requireOpen();
      reservation.state = State.FINISHED;
      reservation.owner.pending--;
      pending--;
      notifyAll();
    }
    releaseHost(reservation);
  }

  private synchronized void awaitIdle(Registration<?, ?> registration) {
    boolean interrupted = false;
    while (registration.pending != 0) {
      try {
        wait();
      } catch (InterruptedException ignored) {
        interrupted = true;
      }
    }
    if (interrupted) Thread.currentThread().interrupt();
  }

  /** 新規要求の投入を停止した後、全モデルの符号化中・実行待ち・実行中のバッチを回収する。 */
  public synchronized void awaitIdle() {
    boolean interrupted = false;
    while (pending != 0) {
      try {
        wait();
      } catch (InterruptedException ignored) {
        interrupted = true;
      }
    }
    if (interrupted) Thread.currentThread().interrupt();
  }

  /** 同じ物理デバイスを一度だけ計上した容量と、計測区間内の最大予約数です。 */
  public synchronized Metrics metrics() {
    return new Metrics(capacity, peakHostReservations);
  }

  /** 要求を投入する処理停止・全予約回収後に、次の計測区間へ切り替えます。 */
  public synchronized void resetMetrics() {
    peakHostReservations = 0;
  }

  public record Metrics(int hostCapacity, int peakHostReservations) {}

  private synchronized void unregister(Registration<?, ?> registration) {
    registration.accepting = false;
    awaitIdle(registration);
    registrations.remove(registration);
    var iterator = lanes.entrySet().iterator();
    while (iterator.hasNext()) {
      Map.Entry<Object, Lane> entry = iterator.next();
      Device remaining = null;
      for (Registration<?, ?> current : registrations) {
        for (Backend<?, ?> backend : current.backends) {
          if (backend.device().key().equals(entry.getKey())) {
            remaining = backend.device();
            break;
          }
        }
        if (remaining != null) break;
      }
      if (remaining == null) {
        capacity -= entry.getValue().device.hostCapacity();
        iterator.remove();
      }
    }
  }

  @Override
  public void close() {
    synchronized (this) {
      if (!registrations.isEmpty())
        throw new IllegalStateException("Close inference registrations first");
      closed = true;
    }
    dispatcher.close();
  }

  /** モデル系列側で入力・出力を処理し、共通側ではデバイスの待機バッチ数と容量を扱う。 */
  public interface Backend<B, R> {
    Device device();

    CompletableFuture<R> trySubmit(B batch, Runnable slotHandoff);
  }

  /** 物理デバイスの待機バッチ数と容量を管理する。CPUも同じ実行環境内で一つの実行枠を共有する。 */
  public interface Device {
    Object key();

    int slots();

    int hostCapacity();

    int suppliedBatches();

    long pendingWork();

    CompletableFuture<Void> capacityAvailable();
  }

  public record Attempt<B, R>(Reservation<B, R> reservation, CompletableFuture<Void> available) {
    public boolean acquired() {
      return reservation != null;
    }
  }

  public static final class Registration<B, R> implements AutoCloseable {
    private final InferenceDispatcher dispatcher;
    private final List<Backend<B, R>> backends;
    private boolean accepting = true;
    private int pending;
    private int nextDevice;

    private Registration(InferenceDispatcher dispatcher, List<Backend<B, R>> backends) {
      this.dispatcher = dispatcher;
      this.backends = backends;
    }

    public Attempt<B, R> tryReserve() {
      return dispatcher.reserve(this);
    }

    /** 構築済み入力の参照を受け付け、容量待ちの要求も終了時に回収する対象へ含める。 */
    public CompletableFuture<R> submit(B batch) {
      return dispatcher.submit(this, batch);
    }

    public boolean needsWork() {
      return dispatcher.needsWork(this);
    }

    public void awaitIdle() {
      dispatcher.awaitIdle(this);
    }

    @Override
    public void close() {
      dispatcher.unregister(this);
    }
  }

  public static final class Reservation<B, R> implements AutoCloseable {
    private final Registration<B, R> owner;
    private final CompletableFuture<R> result = new CompletableFuture<>();
    private State state = State.OPEN;
    private B batch;

    private Reservation(Registration<B, R> owner) {
      this.owner = owner;
    }

    public CompletableFuture<R> submit(B batch) {
      return owner.dispatcher.submit(this, batch);
    }

    @Override
    public void close() {
      owner.dispatcher.abort(this);
    }

    private void requireOpen() {
      if (state != State.OPEN)
        throw new IllegalStateException("Inference reservation already transferred");
    }
  }

  private static final class Lane {
    final Device device;
    CompletableFuture<Void> notification;
    Throwable failure;
    int supply;

    Lane(Device device) {
      this.device = device;
    }
  }

  private enum State {
    OPEN,
    WAITING,
    READY,
    DISPATCHED,
    FINISHED
  }
}
