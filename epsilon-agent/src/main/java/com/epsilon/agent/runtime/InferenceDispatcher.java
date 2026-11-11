package com.epsilon.agent.runtime;

import java.util.ArrayDeque;
import java.util.Iterator;
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

/**
 * 推論のバッチを符号化する前にホストの容量を予約し、符号化を終えたバッチを実行先のデバイスへ出す。
 *
 * <p>ホストの容量は、登録した物理デバイスごとの容量（{@link Device#hostCapacity}）の和。同じデバイスは、いくつのモデルが登録しても
 * 1 回だけ数える。予約した容量は、バッチがスロットへ移ったとき（{@link Backend#trySubmit} に渡した処理が呼ばれたとき）か、バッチが
 * 終わったときに返す。1 つの実行先が満杯でも、ほかのデバイスへ出せるバッチは出し続ける。結果はバッチを出すスレッドとは別のスレッドで
 * 知らせるので、完了の処理から次の推論や登録の解除を呼べる。
 */
public final class InferenceDispatcher implements AutoCloseable {

  private static final Executor DELIVERY = task -> Thread.ofVirtual().name("epsilon-inference-result").start(task);

  private final ExecutorService pump =
      Executors.newSingleThreadExecutor(Thread.ofPlatform().daemon().name("epsilon-inference-dispatch").factory());
  private final AtomicBoolean scheduled = new AtomicBoolean();
  private final Map<Object, Lane> lanes = new LinkedHashMap<>();
  private final Set<Registration<?, ?>> registrations = new LinkedHashSet<>();
  private final Set<Reservation<?, ?>> host = new LinkedHashSet<>();
  private final ArrayDeque<Reservation<?, ?>> ready = new ArrayDeque<>();
  private CompletableFuture<Void> availability = new CompletableFuture<>();
  private int capacity;
  private int peakReservations;
  private int pending;
  private boolean closed;

  /**
   * 1 つのモデルの実行先を、デバイスごとの {@link Backend} で登録する。キーが同じデバイスは同じオブジェクトでなければならず、違えば
   * IllegalArgumentException を投げる。
   */
  public synchronized <B, R> Registration<B, R> register(List<? extends Backend<B, R>> backends) {
    if (closed) {
      throw new IllegalStateException("The inference dispatcher is closed");
    }
    if (backends.isEmpty()) {
      throw new IllegalArgumentException("A registration needs at least one backend");
    }
    for (Backend<B, R> backend : backends) {
      Device device = backend.device();
      Lane lane = lanes.get(device.key());
      if (lane == null) {
        lanes.put(device.key(), new Lane(device));
        capacity += device.hostCapacity();
      } else if (lane.device != device) {
        throw new IllegalArgumentException("Devices with one key must be one object: " + device.key());
      }
    }
    Registration<B, R> registration = new Registration<>(this, List.copyOf(backends));
    registrations.add(registration);
    return registration;
  }

  /** 全てのモデルについて、符号化中・実行待ち・実行中のバッチが終わるまで待つ。新しいバッチを出すのを止めてから呼ぶ。 */
  public synchronized void awaitIdle() {
    boolean interrupted = false;
    while (pending != 0) {
      try {
        wait();
      } catch (InterruptedException ignored) {
        interrupted = true;
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  public synchronized Metrics metrics() {
    return new Metrics(capacity, peakReservations);
  }

  /** 次の計測の区間を始める。バッチを出すのを止め、全ての予約が返った後に呼ぶ。 */
  public synchronized void resetMetrics() {
    peakReservations = 0;
  }

  /** 登録を全て閉じた後に呼ぶ。閉じていない登録が残っていれば IllegalStateException を投げる。 */
  @Override
  public void close() {
    synchronized (this) {
      if (!registrations.isEmpty()) {
        throw new IllegalStateException("Close the inference registrations first");
      }
      closed = true;
    }
    pump.close();
  }

  private synchronized <B, R> Attempt<B, R> reserve(Registration<B, R> owner) {
    requireAccepting(owner);
    if (host.size() >= capacity) {
      return new Attempt<>(null, availability);
    }
    Reservation<B, R> reservation = new Reservation<>(owner);
    host.add(reservation);
    peakReservations = Math.max(peakReservations, host.size());
    owner.pending++;
    pending++;
    return new Attempt<>(reservation, null);
  }

  private void requireAccepting(Registration<?, ?> owner) {
    if (!owner.accepting) {
      throw new IllegalStateException("The inference registration is closed");
    }
    Throwable failure = deviceFailure(owner);
    if (failure != null) {
      throw new IllegalStateException("An inference device failed", failure);
    }
  }

  private Throwable deviceFailure(Registration<?, ?> owner) {
    for (Backend<?, ?> backend : owner.backends) {
      Throwable failure = lanes.get(backend.device().key()).failure;
      if (failure != null) {
        return failure;
      }
    }
    return null;
  }

  /**
   * 予約済みのバッチを実行先へ割り振ったと見込んでも、候補のモデルの実行先のどれかでスロットが埋まらないなら true。見込みでは、
   * 実行先の少ない予約から順に、供給が最も少ないデバイスへ置く。こうすると、GPU 0 でしか実行できないバッチを GPU 1 の供給に数えない。
   */
  private synchronized boolean needsWork(Registration<?, ?> candidate) {
    for (Lane lane : lanes.values()) {
      lane.supply = lane.device.suppliedBatches();
    }
    for (int width = 1; width <= lanes.size(); width++) {
      for (Reservation<?, ?> reservation : host) {
        if ((reservation.state == State.OPEN || reservation.state == State.READY)
            && reservation.owner.backends.size() == width) {
          Lane selected = null;
          for (Backend<?, ?> backend : reservation.owner.backends) {
            Lane lane = lanes.get(backend.device().key());
            if (selected == null || lane.supply < selected.supply) {
              selected = lane;
            }
          }
          selected.supply++;
        }
      }
    }
    for (Backend<?, ?> backend : candidate.backends) {
      if (lanes.get(backend.device().key()).supply < backend.device().slots()) {
        return true;
      }
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

  private synchronized void signal() {
    if (closed) {
      return;
    }
    if (scheduled.compareAndSet(false, true)) {
      pump.execute(
          () -> {
            scheduled.set(false);
            dispatchReady();
          });
    }
  }

  private synchronized void dispatchReady() {
    int remaining = ready.size();
    while (remaining-- > 0) {
      Reservation<?, ?> reservation = ready.removeFirst();
      if (!dispatch(reservation)) {
        ready.addLast(reservation);
      }
    }
  }

  /**
   * バッチを出す。待ちの作業量が最も少ない実行先から試し、満杯なら並び順で次の実行先を試す。どの実行先も満杯なら false を返す。
   * そのバッチは、容量が空いたときの通知でもう一度出す。
   */
  private <B, R> boolean dispatch(Reservation<B, R> reservation) {
    List<Backend<B, R>> backends = reservation.owner.backends;
    Throwable failure = deviceFailure(reservation.owner);
    if (failure != null) {
      finish(reservation, null, failure);
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
        // 試す前に通知を取っておく。試した後に容量が空いても、その通知を取りこぼさない
        changed = backend.device().capacityAvailable();
        completion = backend.trySubmit(reservation.batch, () -> releaseHost(reservation));
      } catch (RuntimeException | Error submitFailure) {
        finish(reservation, null, submitFailure);
        return true;
      }
      if (completion != null) {
        completion.whenCompleteAsync((result, resultFailure) -> finish(reservation, result, resultFailure), pump);
        return true;
      }
      Lane lane = lanes.get(backend.device().key());
      if (lane.notification != changed) {
        lane.notification = changed;
        changed.whenComplete(
            (ignored, notificationFailure) -> {
              synchronized (InferenceDispatcher.this) {
                if (notificationFailure != null) {
                  lane.failure = notificationFailure;
                }
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
      if (!host.remove(reservation)) {
        return;
      }
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
          if (failure == null) {
            reservation.result.complete(result);
          } else {
            reservation.result.completeExceptionally(failure);
          }
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
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  private synchronized void unregister(Registration<?, ?> registration) {
    registration.accepting = false;
    awaitIdle(registration);
    registrations.remove(registration);
    Iterator<Map.Entry<Object, Lane>> iterator = lanes.entrySet().iterator();
    while (iterator.hasNext()) {
      Map.Entry<Object, Lane> entry = iterator.next();
      if (registrations.stream()
          .flatMap(other -> other.backends.stream())
          .noneMatch(backend -> backend.device().key().equals(entry.getKey()))) {
        capacity -= entry.getValue().device.hostCapacity();
        iterator.remove();
      }
    }
  }

  /**
   * @param hostCapacity 登録したデバイスのホストの容量の和
   * @param peakReservations 計測の区間で同時に持った予約の数の最大
   */
  public record Metrics(int hostCapacity, int peakReservations) {}

  /** 1 つのモデルを 1 つのデバイスで動かすバックエンド。バッチの入力と出力はモデルの系列が扱う。 */
  public interface Backend<B, R> {
    Device device();

    /**
     * バッチをスロットへ出す。
     *
     * @param slotHandoff バッチがスロットへ移ったときに一度だけ呼ぶ処理。ホストの容量を返す
     * @return バッチの推論が終わると完了する結果。実行先が満杯なら null
     */
    CompletableFuture<R> trySubmit(B batch, Runnable slotHandoff);
  }

  /** 物理デバイスのスロット・ホストの容量・待ちの作業量。 */
  public interface Device {
    /** 物理デバイスを表す値。値が同じデバイスは 1 つとして数える。 */
    Object key();

    /** 同時に実行できるバッチの数。 */
    int slots();

    /** 符号化中と実行待ちのバッチを、合わせていくつまで持てるか。 */
    int hostCapacity();

    /** 実行中と実行待ちのバッチの数。 */
    int suppliedBatches();

    /** 待ちの作業量の見積もり。単位はデバイスの所有者が決める。 */
    long pendingWork();

    /** 次に容量が空いたときに完了する通知を返す。デバイスが壊れたら例外で完了する。 */
    CompletableFuture<Void> capacityAvailable();
  }

  /** 予約を試した結果。予約か、次に容量が空いたときの通知の、どちらか一方を持つ。 */
  public record Attempt<B, R>(Reservation<B, R> reservation, CompletableFuture<Void> available) {
    public boolean acquired() {
      return reservation != null;
    }
  }

  /** 1 つのモデルの登録。 */
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

    /** 待たずにホストの容量を予約する。 */
    public Attempt<B, R> tryReserve() {
      return dispatcher.reserve(this);
    }

    /** 予約済みのバッチを数えても、このモデルの実行先でまだバッチが足りないなら true。 */
    public boolean needsWork() {
      return dispatcher.needsWork(this);
    }

    /** このモデルの予約が全て終わるまで待つ。 */
    public void awaitIdle() {
      dispatcher.awaitIdle(this);
    }

    /** 新しい予約を受け付けなくし、このモデルの予約が全て終わってから登録を外す。 */
    @Override
    public void close() {
      dispatcher.unregister(this);
    }
  }

  /** 1 つのバッチの予約。予約を持つ側は、バッチを出す（{@link #submit}）か予約を返す（{@link #close}）かのどちらかを行う。 */
  public static final class Reservation<B, R> implements AutoCloseable {
    private final Registration<B, R> owner;
    private final CompletableFuture<R> result = new CompletableFuture<>();
    private State state = State.OPEN;
    private B batch;

    private Reservation(Registration<B, R> owner) {
      this.owner = owner;
    }

    /** 符号化を終えたバッチを出す。結果は、実行先がリソースを返した後に完了する。 */
    public CompletableFuture<R> submit(B batch) {
      return owner.dispatcher.submit(this, batch);
    }

    /** 出していない予約を返す。 */
    @Override
    public void close() {
      owner.dispatcher.abort(this);
    }

    private void requireOpen() {
      if (state != State.OPEN) {
        throw new IllegalStateException("The inference reservation was already used");
      }
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
    READY,
    DISPATCHED,
    FINISHED
  }
}
