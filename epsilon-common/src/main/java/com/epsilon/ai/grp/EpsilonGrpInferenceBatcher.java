package com.epsilon.ai.grp;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;

/**
 * 自己対戦で得た局開始時点のGRP推論要求をまとめ、バッチとして実行する。
 *
 * <p>非同期モードでは要求をキューに蓄積し、最大件数または待ち時間の上限に達した時点でまとめて推論する。呼び出し元は、返された非同期処理の結果を必要な時点で取得する。キューには上限を設け、推論が追いつかない場合は要求を投入する側が空きを待つ。
 */
public final class EpsilonGrpInferenceBatcher implements EpsilonGrpRankPredictor, AutoCloseable {

  @FunctionalInterface
  interface MarginalBackend {
    List<float[]> predict(List<float[]> sequences);
  }

  private final MarginalBackend backend;
  private final boolean asyncBatchingEnabled;
  private final int maxBatch;
  private final long coalesceWaitNanos;
  private final ArrayBlockingQueue<Request> queue;
  private final Semaphore queueSlots;
  private final Thread worker;
  private final Object lifecycleLock = new Object();
  private final LongAdder requests = new LongAdder();
  private final LongAdder physicalCalls = new LongAdder();
  private final LongAdder singletonCalls = new LongAdder();
  private final LongAdder queueWaitNanos = new LongAdder();
  private final LongAdder enqueueWaitNanos = new LongAdder();
  private final LongAdder completionNanos = new LongAdder();
  private final LongAdder inferenceNanos = new LongAdder();
  private final LongAdder actualSteps = new LongAdder();
  private final LongAdder paddedSteps = new LongAdder();
  private final AtomicInteger maxObservedBatch = new AtomicInteger();
  private final AtomicInteger maxQueueDepth = new AtomicInteger();
  private volatile boolean accepting = true;
  private volatile Throwable terminalFailure;

  /**
   * 有効な GRP 推論設定からバッチ化処理を構築する。
   *
   * @param inference バッチ推論を実行する推論器
   * @param async 要求を非同期に集約するなら {@code true}
   * @param maxBatch 一回の推論にまとめる最大要求数
   * @param coalesceWaitMicros 要求をまとめるための最大待ち時間（マイクロ秒）
   * @return 設定済みバッチ化処理
   */
  public static EpsilonGrpInferenceBatcher create(
      EpsilonGrpInference inference, boolean async, int maxBatch, int coalesceWaitMicros) {
    return new EpsilonGrpInferenceBatcher(
        inference::predictMarginalProbabilities, async, maxBatch, coalesceWaitMicros);
  }

  EpsilonGrpInferenceBatcher(
      MarginalBackend backend, boolean asyncBatchingEnabled, int maxBatch, int coalesceWaitMicros) {
    this.backend = backend;
    this.asyncBatchingEnabled = asyncBatchingEnabled;
    this.maxBatch = maxBatch;
    this.coalesceWaitNanos = TimeUnit.MICROSECONDS.toNanos(coalesceWaitMicros);
    this.queue = new ArrayBlockingQueue<>(maxBatch * 4);
    this.queueSlots = new Semaphore(maxBatch * 4);
    if (asyncBatchingEnabled) {
      worker = new Thread(this::runWorker, "epsilon-grp-inference-batcher");
      worker.setDaemon(true);
      worker.start();
    } else {
      worker = null;
    }
  }

  @Override
  public CompletableFuture<float[]> predictRankProbabilitiesAsync(float[] sequence, int seat) {
    if (sequence == null || EpsilonGrpFeature.steps(sequence) <= 0) {
      return CompletableFuture.completedFuture(null);
    }
    if (seat < 0 || seat >= EpsilonGrpRanks.SEAT_COUNT) {
      throw new IllegalArgumentException("seat must be 0-3: " + seat);
    }
    return submit(sequence, seat);
  }

  @Override
  public CompletableFuture<float[]> predictMarginalProbabilitiesAsync(float[] sequence) {
    if (sequence == null || EpsilonGrpFeature.steps(sequence) <= 0) {
      return CompletableFuture.completedFuture(null);
    }
    return submit(sequence, -1);
  }

  private CompletableFuture<float[]> submit(float[] sequence, int seat) {
    requests.increment();
    Throwable failure = terminalFailure;
    if (failure != null) {
      return CompletableFuture.failedFuture(failure);
    }
    if (!accepting) {
      return CompletableFuture.failedFuture(
          new IllegalStateException("GRP inference batcher is closed"));
    }
    if (!asyncBatchingEnabled) {
      return predictDirect(sequence, seat, System.nanoTime());
    }

    Request request = new Request(sequence, seat, System.nanoTime());
    long enqueueStartedNanos = System.nanoTime();
    try {
      queueSlots.acquire();
      synchronized (lifecycleLock) {
        if (terminalFailure != null) {
          queueSlots.release();
          return CompletableFuture.failedFuture(terminalFailure);
        }
        if (!accepting) {
          queueSlots.release();
          return CompletableFuture.failedFuture(
              new IllegalStateException("GRP inference batcher is closed"));
        }
        queue.add(request);
        maxQueueDepth.accumulateAndGet(queue.size(), Math::max);
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      request.future.completeExceptionally(e);
    } finally {
      enqueueWaitNanos.add(System.nanoTime() - enqueueStartedNanos);
    }
    return request.future;
  }

  private CompletableFuture<float[]> predictDirect(
      float[] sequence, int seat, long submittedNanos) {
    try {
      List<float[]> result = backend.predict(List.of(sequence));
      recordPhysicalBatch(List.of(sequence), submittedNanos);
      completionNanos.add(System.nanoTime() - submittedNanos);
      return CompletableFuture.completedFuture(prediction(result.getFirst(), seat));
    } catch (RuntimeException | Error e) {
      synchronized (lifecycleLock) {
        terminalFailure = e;
        accepting = false;
      }
      return CompletableFuture.failedFuture(e);
    }
  }

  private void runWorker() {
    while (accepting || !queue.isEmpty()) {
      Request first = pollFirst();
      if (first == null) {
        continue;
      }
      ArrayList<Request> batch = new ArrayList<>(maxBatch);
      batch.add(first);
      drainTo(batch, maxBatch - 1);
      fillUntilDeadline(batch, first.submittedNanos + coalesceWaitNanos);
      if (!predictBatch(batch)) {
        return;
      }
    }
  }

  private Request pollFirst() {
    try {
      Request request = accepting ? queue.poll(100L, TimeUnit.MILLISECONDS) : queue.poll();
      if (request != null) {
        queueSlots.release();
      }
      return request;
    } catch (InterruptedException e) {
      Request request = queue.poll();
      if (request != null) {
        queueSlots.release();
      }
      return request;
    }
  }

  private void fillUntilDeadline(ArrayList<Request> batch, long deadlineNanos) {
    while (batch.size() < maxBatch) {
      drainTo(batch, maxBatch - batch.size());
      if (batch.size() >= maxBatch || coalesceWaitNanos == 0L || !accepting) {
        return;
      }
      long remainingNanos = deadlineNanos - System.nanoTime();
      if (remainingNanos <= 0L) {
        return;
      }
      try {
        Request request = queue.poll(remainingNanos, TimeUnit.NANOSECONDS);
        if (request == null) {
          return;
        }
        queueSlots.release();
        batch.add(request);
      } catch (InterruptedException e) {
        return;
      }
    }
  }

  private void drainTo(ArrayList<Request> batch, int limit) {
    int before = batch.size();
    queue.drainTo(batch, limit);
    queueSlots.release(batch.size() - before);
  }

  private boolean predictBatch(List<Request> batch) {
    long startedNanos = System.nanoTime();
    ArrayList<float[]> sequences = new ArrayList<>(batch.size());
    for (Request request : batch) {
      queueWaitNanos.add(startedNanos - request.submittedNanos);
      sequences.add(request.sequence);
    }
    try {
      List<float[]> result = backend.predict(sequences);
      if (result.size() != batch.size()) {
        throw new IllegalStateException(
            "GRP prediction count mismatch: requested=" + batch.size() + " got=" + result.size());
      }
      recordPhysicalBatch(sequences, startedNanos);
      for (int i = 0; i < batch.size(); i++) {
        Request request = batch.get(i);
        completionNanos.add(System.nanoTime() - request.submittedNanos);
        request.future.complete(prediction(result.get(i), request.seat));
      }
      return true;
    } catch (RuntimeException | Error e) {
      fail(batch, e);
      return false;
    }
  }

  private void recordPhysicalBatch(List<float[]> sequences, long startedNanos) {
    inferenceNanos.add(System.nanoTime() - startedNanos);
    physicalCalls.increment();
    if (sequences.size() == 1) {
      singletonCalls.increment();
    }
    maxObservedBatch.accumulateAndGet(sequences.size(), Math::max);
    int maxSteps = 0;
    long batchActualSteps = 0L;
    for (float[] sequence : sequences) {
      int steps = EpsilonGrpFeature.steps(sequence);
      batchActualSteps += steps;
      maxSteps = Math.max(maxSteps, steps);
    }
    actualSteps.add(batchActualSteps);
    paddedSteps.add((long) maxSteps * sequences.size());
  }

  private void fail(List<Request> currentBatch, Throwable cause) {
    synchronized (lifecycleLock) {
      terminalFailure = cause;
      accepting = false;
      for (Request request : currentBatch) {
        request.future.completeExceptionally(cause);
      }
      Request queued;
      while ((queued = queue.poll()) != null) {
        queueSlots.release();
        queued.future.completeExceptionally(cause);
      }
    }
  }

  private static float[] seatMarginal(float[] marginals, int seat) {
    float[] out = new float[EpsilonGrpRanks.RANK_COUNT];
    System.arraycopy(
        marginals, seat * EpsilonGrpRanks.RANK_COUNT, out, 0, EpsilonGrpRanks.RANK_COUNT);
    return out;
  }

  private static float[] prediction(float[] marginals, int seat) {
    return seat < 0 ? marginals : seatMarginal(marginals, seat);
  }

  /**
   * 現在までのバッチ化、待ち時間、パディング指標をスナップショット化する。
   *
   * @return カウンターの一貫した読み取りスナップショット
   */
  public Metrics metrics() {
    long requestCount = requests.sum();
    long physicalCallCount = physicalCalls.sum();
    long actualStepCount = actualSteps.sum();
    return new Metrics(
        asyncBatchingEnabled,
        requestCount,
        physicalCallCount,
        physicalCallCount == 0L ? 0.0 : requestCount / (double) physicalCallCount,
        maxObservedBatch.get(),
        singletonCalls.sum(),
        maxBatch,
        TimeUnit.NANOSECONDS.toMicros(coalesceWaitNanos),
        requestCount == 0L
            ? 0.0
            : queueWaitNanos.sum() / (double) requestCount / TimeUnit.MICROSECONDS.toNanos(1L),
        enqueueWaitNanos.sum() / 1_000_000.0,
        requestCount == 0L
            ? 0.0
            : completionNanos.sum() / (double) requestCount / TimeUnit.MICROSECONDS.toNanos(1L),
        inferenceNanos.sum() / 1_000_000.0,
        actualStepCount,
        paddedSteps.sum(),
        actualStepCount == 0L ? 1.0 : paddedSteps.sum() / (double) actualStepCount,
        maxQueueDepth.get());
  }

  @Override
  public void close() {
    synchronized (lifecycleLock) {
      accepting = false;
      if (worker != null) {
        worker.interrupt();
      }
    }
    if (worker == null) {
      return;
    }
    boolean interrupted = false;
    while (worker.isAlive()) {
      try {
        worker.join();
      } catch (InterruptedException e) {
        interrupted = true;
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
  }

  private static final class Request {
    private final float[] sequence;
    private final int seat;
    private final long submittedNanos;
    private final CompletableFuture<float[]> future = new CompletableFuture<>();

    private Request(float[] sequence, int seat, long submittedNanos) {
      this.sequence = sequence;
      this.seat = seat;
      this.submittedNanos = submittedNanos;
    }
  }

  /**
   * GRP 推論バッチ化処理のキュー・パディング・実行呼び出し指標。
   *
   * @param asyncBatchingEnabled 非同期バッチ集約を有効にしたか
   * @param requests 受理した論理要求数
   * @param physicalCalls 実行した順伝播数
   * @param averageBatch 順伝播の平均要求数
   * @param observedMaxBatch 観測した最大要求数
   * @param singletonCalls 1 要求だけだった実行呼び出し数
   * @param configuredMaxBatch 設定上の最大バッチ
   * @param coalesceWaitMicros バッチへの集約待ち時間（マイクロ秒）
   * @param averageQueueWaitMicros 要求の平均キュー待ち時間
   * @param enqueueWaitMillis キュー追加側で待った累計時間
   * @param averageCompletionMicros キュー追加から完了までの平均時間
   * @param inferenceMillis 順伝播の累計時間
   * @param actualSteps パディング前の系列ステップ数
   * @param paddedSteps パディング後の系列ステップ数
   * @param paddingRatio パディングステップの割合
   * @param maxQueueDepth 観測した最大キュー深さ
   */
  public record Metrics(
      boolean asyncBatchingEnabled,
      long requests,
      long physicalCalls,
      double averageBatch,
      int observedMaxBatch,
      long singletonCalls,
      int configuredMaxBatch,
      long coalesceWaitMicros,
      double averageQueueWaitMicros,
      double enqueueWaitMillis,
      double averageCompletionMicros,
      double inferenceMillis,
      long actualSteps,
      long paddedSteps,
      double paddingRatio,
      int maxQueueDepth) {

    /**
     * ログ出力向けの1行要約を返す。
     *
     * @return ロケール非依存のバッチ化指標文字列
     */
    public String summary() {
      return String.format(
          Locale.ROOT,
          "async=%s requests=%d calls=%d avgBatch=%.3f observedMaxBatch=%d "
              + "configuredMaxBatch=%d singletonCalls=%d coalesceWaitUs=%d "
              + "avgQueueWaitUs=%.3f enqueueWaitMs=%.3f avgCompletionUs=%.3f "
              + "inferenceMs=%.3f actualSteps=%d paddedSteps=%d padRatio=%.3f maxQueueDepth=%d",
          asyncBatchingEnabled,
          requests,
          physicalCalls,
          averageBatch,
          observedMaxBatch,
          configuredMaxBatch,
          singletonCalls,
          coalesceWaitMicros,
          averageQueueWaitMicros,
          enqueueWaitMillis,
          averageCompletionMicros,
          inferenceMillis,
          actualSteps,
          paddedSteps,
          paddingRatio,
          maxQueueDepth);
    }
  }
}
