package com.epsilon.runtime;

import java.lang.management.ManagementFactory;
import java.lang.management.ThreadMXBean;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Phaser;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * 入力行の順序を保ち、モデル系列ごとの格納領域へ並列に符号化する。
 *
 * <p>各入力行の符号化は一度だけ実行する。入力は完了まで借用するため、呼び出し元はその間に局面を進めたり、同じ局面・プレイヤーを重複して渡したりしてはならない。
 * 格納方法と完成後の出力型は、モデル系列側のコールバックで定義する。
 */
public final class BatchEncodingExecutor<K, B, H> implements AutoCloseable {

  private static final int MINIMUM_PARALLEL_ROWS = 128;
  private static final int TARGET_ROWS_PER_TASK = 64;
  private static final AtomicInteger THREAD_SEQUENCE = new AtomicInteger();

  private final int workers;
  private final ExecutorService executor;
  private final BiFunction<Integer, K, B> createStorage;
  private final Function<B, H> finishStorage;
  private final Phaser pendingBatches = new Phaser(1);
  private final ThreadMXBean threadMetrics = ManagementFactory.getThreadMXBean();

  public BatchEncodingExecutor(
      int workers, BiFunction<Integer, K, B> createStorage, Function<B, H> finishStorage) {
    this.workers = workers;
    this.createStorage = createStorage;
    this.finishStorage = finishStorage;
    executor =
        Executors.newFixedThreadPool(
            workers,
            task -> {
              Thread thread =
                  new Thread(
                      task, "epsilon-decision-arena-encode-" + THREAD_SEQUENCE.getAndIncrement());
              thread.setDaemon(true);
              return thread;
            });
  }

  /** 格納領域の確保も実行サービスへ移し、互いに重ならない行範囲を並列に構築します。 */
  public <R> CompletableFuture<H> encodeAsync(
      List<R> rows, K key, RangeEncoder<? super R, B> rangeEncoder) {
    int taskCount =
        rows.size() < MINIMUM_PARALLEL_ROWS
            ? 1
            : Math.min(workers, Math.max(1, rows.size() / TARGET_ROWS_PER_TASK));
    InputBatchProfile.Encoding profile =
        InputBatchProfile.beginEncoding(key, rows.size(), taskCount, -1, "TOTAL");
    long enqueuedNanos = profile == null ? 0 : System.nanoTime();
    pendingBatches.register();
    try {
      return CompletableFuture.supplyAsync(
              () -> {
                try {
                  if (profile != null) profile.queueWaitNanos = System.nanoTime() - enqueuedNanos;
                  CompletableFuture<H> encoded =
                      encodeAllocated(rows, key, rangeEncoder, taskCount, profile);
                  // 呼び出し元が返却済みの非同期結果をキャンセルしても、書き込み中のバッチは解放しない。
                  encoded.whenComplete(
                      (ignored, failure) -> {
                        completeProfile(profile, failure == null);
                        pendingBatches.arriveAndDeregister();
                      });
                  return encoded;
                } catch (RuntimeException | Error failure) {
                  completeProfile(profile, false);
                  pendingBatches.arriveAndDeregister();
                  throw failure;
                }
              },
              executor)
          .thenCompose(Function.identity());
    } catch (RuntimeException | Error failure) {
      completeProfile(profile, false);
      pendingBatches.arriveAndDeregister();
      throw failure;
    }
  }

  private <R> CompletableFuture<H> encodeAllocated(
      List<R> rows,
      K key,
      RangeEncoder<? super R, B> rangeEncoder,
      int taskCount,
      InputBatchProfile.Encoding profile) {
    B storage = allocate(rows.size(), key, profile);
    if (taskCount == 1) {
      encodeRange(rows, storage, rangeEncoder, 0, rows.size(), profile, 0, 0);
      return CompletableFuture.completedFuture(finish(storage, profile));
    }

    ArrayList<CompletableFuture<Void>> tasks = new ArrayList<>(taskCount);
    for (int task = 0; task < taskCount; task++) {
      int from = task * rows.size() / taskCount;
      int to = (task + 1) * rows.size() / taskCount;
      int taskIndex = task;
      long enqueuedNanos = profile == null ? 0 : System.nanoTime();
      tasks.add(
          CompletableFuture.runAsync(
              () ->
                  encodeRange(
                      rows, storage, rangeEncoder, from, to, profile, taskIndex, enqueuedNanos),
              executor));
    }
    return CompletableFuture.allOf(tasks.toArray(CompletableFuture[]::new))
        .thenApply(ignored -> finish(storage, profile));
  }

  private B allocate(int rows, K key, InputBatchProfile.Encoding profile) {
    InputBatchProfile.Encoding event = beginPhase(profile, rows, -1, "ALLOCATE");
    long started = event == null ? 0 : System.nanoTime();
    long cpuStarted = event == null ? -1 : currentThreadCpuNanos();
    boolean success = false;
    try {
      B storage = createStorage.apply(rows, key);
      success = true;
      return storage;
    } finally {
      completePhase(event, started, cpuStarted, success);
    }
  }

  private H finish(B storage, InputBatchProfile.Encoding profile) {
    InputBatchProfile.Encoding event =
        beginPhase(profile, profile == null ? 0 : profile.rows, -1, "FINISH");
    long started = event == null ? 0 : System.nanoTime();
    long cpuStarted = event == null ? -1 : currentThreadCpuNanos();
    boolean success = false;
    try {
      H result = finishStorage.apply(storage);
      success = true;
      return result;
    } finally {
      completePhase(event, started, cpuStarted, success);
    }
  }

  private <R> void encodeRange(
      List<R> rows,
      B storage,
      RangeEncoder<? super R, B> rangeEncoder,
      int from,
      int to,
      InputBatchProfile.Encoding profile,
      int taskIndex,
      long enqueuedNanos) {
    InputBatchProfile.Encoding event = beginPhase(profile, to - from, taskIndex, "ROWS");
    long started = event == null ? 0 : System.nanoTime();
    long cpuStarted = event == null ? -1 : currentThreadCpuNanos();
    if (event != null && enqueuedNanos != 0) event.queueWaitNanos = started - enqueuedNanos;
    boolean success = false;
    try {
      rangeEncoder.encode(rows, storage, from, to);
      success = true;
    } finally {
      completePhase(event, started, cpuStarted, success);
    }
  }

  private static InputBatchProfile.Encoding beginPhase(
      InputBatchProfile.Encoding profile, int rows, int taskIndex, String phase) {
    if (profile == null) return null;
    InputBatchProfile.Encoding event =
        InputBatchProfile.beginEncoding(profile.bucket, rows, profile.taskCount, taskIndex, phase);
    if (event != null) event.batchRows = profile.rows;
    return event;
  }

  private void completePhase(
      InputBatchProfile.Encoding event, long started, long cpuStarted, boolean success) {
    if (event == null) return;
    event.end();
    event.runNanos = System.nanoTime() - started;
    long cpuFinished = cpuStarted < 0 ? -1 : currentThreadCpuNanos();
    if (cpuFinished >= cpuStarted && cpuStarted >= 0)
      event.threadCpuNanos = cpuFinished - cpuStarted;
    event.success = success;
    event.commit();
  }

  private static void completeProfile(InputBatchProfile.Encoding profile, boolean success) {
    if (profile == null) return;
    profile.end();
    profile.success = success;
    profile.commit();
  }

  private long currentThreadCpuNanos() {
    return threadMetrics.isCurrentThreadCpuTimeSupported() && threadMetrics.isThreadCpuTimeEnabled()
        ? threadMetrics.getCurrentThreadCpuTime()
        : -1;
  }

  /**
   * 受け付け済みバッチの書き込みを完了させてから実行サービスを終了します。
   *
   * <p>この実行サービスを作成した単一の所有者が、新規符号化受付を止めてから呼びます。終了処理中の新規受付や複数スレッドからの同時終了は行いません。
   */
  @Override
  public void close() {
    pendingBatches.arriveAndAwaitAdvance();
    executor.close();
  }

  @FunctionalInterface
  public interface RangeEncoder<R, B> {
    /** 指定範囲だけを書き込み、戻るまでに借用した作業領域を返却します。 */
    void encode(List<? extends R> sources, B storage, int fromInclusive, int toExclusive);
  }
}
