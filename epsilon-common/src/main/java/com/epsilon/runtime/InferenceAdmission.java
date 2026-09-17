package com.epsilon.runtime;

import com.epsilon.config.settings.InferenceBatchingSettings;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.function.Predicate;
import java.util.function.ToIntBiFunction;

/** 待機時間の上限、バッチの充填状況、実行先の空きに基づいて推論を開始する。容量を予約できた場合だけ入力行をキューから取り出す。 */
public final class InferenceAdmission<E, K, R> {
  private final InferenceBatchQueue<E, K, R> queue;
  private final BiFunction<E, K, DecisionInferenceIngress.Attempt> acquireIngress;
  private final Predicate<E> needsWork;
  private final IdentityHashMap<E, Boolean> unavailable = new IdentityHashMap<>();
  private final ArrayList<CompletableFuture<Void>> availabilitySignals = new ArrayList<>();
  private final Predicate<E> availableEvaluator = evaluator -> !unavailable.containsKey(evaluator);
  private final Predicate<E> underSuppliedEvaluator;
  private final long[] batchesByReason = new long[BatchDispatchReason.values().length];
  private final long[] rowsByReason = new long[BatchDispatchReason.values().length];
  private long totalOldestRowWaitNanos;
  private long maxOldestRowWaitNanos;
  private final IdentityHashMap<CompletableFuture<Void>, Boolean> observedCapacity =
      new IdentityHashMap<>();
  private CompletableFuture<Void> nextIngressAvailable = new CompletableFuture<>();
  private int outstandingBatchCount;

  public InferenceAdmission(
      InferenceBatchingSettings settings,
      ToIntBiFunction<E, K> batchSize,
      BiFunction<E, K, DecisionInferenceIngress.Attempt> acquireIngress,
      Predicate<E> needsWork) {
    queue = new InferenceBatchQueue<>(settings.maxBatchWaitNanos(), batchSize);
    this.acquireIngress = acquireIngress;
    this.needsWork = needsWork;
    underSuppliedEvaluator =
        evaluator -> availableEvaluator.test(evaluator) && this.needsWork.test(evaluator);
  }

  public void add(E evaluator, K key, R row, long nowNanos) {
    queue.add(evaluator, key, row, nowNanos);
  }

  /** 満杯または期限到達のバッチを優先し、入力生成が進行中なら未充填バッチの先行発行を待ちます。 入力生成が終われば残りの行を発行できるため、推論の完了待ちとは区別して渡します。 */
  public AdmittedBatch<E, K, R> startNextEncoding(long nowNanos, boolean inputGenerationInFlight) {
    InputBatchProfile.Selection profile = InputBatchProfile.beginSelection();
    AdmittedBatch<E, K, R> admitted = null;
    try {
      unavailable.clear();
      availabilitySignals.clear();
      admitted = tryStart(BatchDispatchReason.DEADLINE, availableEvaluator, nowNanos, profile);
      if (admitted == null)
        admitted = tryStart(BatchDispatchReason.FULL, availableEvaluator, nowNanos, profile);
      if (admitted == null && !inputGenerationInFlight)
        admitted = tryStart(BatchDispatchReason.SUPPLY, underSuppliedEvaluator, nowNanos, profile);
      prepareCapacityWake();
      return admitted;
    } finally {
      if (profile != null) {
        profile.end();
        if (admitted != null) {
          profile.admitted = true;
          Object evaluator = admitted.evaluator();
          profile.model =
              evaluator.getClass().getSimpleName()
                  + "@"
                  + Integer.toHexString(System.identityHashCode(evaluator));
          profile.bucket = String.valueOf(admitted.key());
          profile.dispatchReason = admitted.reason().name();
          profile.rows = admitted.rows().size();
          profile.oldestRowWaitNanos = admitted.queueWaitNanos();
        }
        profile.commit();
      }
    }
  }

  /** 符号化失敗時だけ、まだ移譲していない予約を返します。 */
  public void encodingFailed(AdmittedBatch<E, K, R> admitted) {
    admitted.ingress.abort();
    outstandingBatchCount--;
  }

  /** 投入済みの推論を回収します。予約は実行先が既に所有しています。 */
  public void inferenceFinished() {
    outstandingBatchCount--;
  }

  public CompletableFuture<Void> ingressAvailable() {
    return nextIngressAvailable;
  }

  public boolean isWaitingForIngress() {
    return !availabilitySignals.isEmpty();
  }

  /** 容量通知と完了イベントを待ち、未予約行の期限ではnullを返して発行を再試行します。 */
  public <T> T awaitEvent(BlockingQueue<T> events) throws InterruptedException {
    long deadline = queue.nextDeadlineNanos(availableEvaluator);
    return deadline == Long.MAX_VALUE
        ? events.take()
        : events.poll(Math.max(0, deadline - System.nanoTime()), TimeUnit.NANOSECONDS);
  }

  public boolean hasQueuedRows() {
    return queue.hasQueuedRows();
  }

  public boolean hasInFlightBatches() {
    return outstandingBatchCount > 0;
  }

  public Metrics metrics() {
    return new Metrics(
        batchesByReason[0],
        rowsByReason[0],
        batchesByReason[1],
        rowsByReason[1],
        batchesByReason[2],
        rowsByReason[2],
        totalOldestRowWaitNanos,
        maxOldestRowWaitNanos);
  }

  private AdmittedBatch<E, K, R> tryStart(
      BatchDispatchReason reason,
      Predicate<E> allowed,
      long nowNanos,
      InputBatchProfile.Selection profile) {
    while (true) {
      var candidate = queue.peekBest(reason, allowed, nowNanos);
      if (candidate == null) return null;
      var attempt = acquireIngress.apply(candidate.evaluator(), candidate.key());
      if (!attempt.acquired()) {
        unavailable.put(candidate.evaluator(), Boolean.TRUE);
        availabilitySignals.add(attempt.available());
        continue;
      }
      InferenceBatchQueue.Batch<E, K, R> batch;
      try {
        batch = queue.remove(candidate, profile, nowNanos);
      } catch (RuntimeException | Error failure) {
        attempt.ingress().abort();
        throw failure;
      }
      outstandingBatchCount++;
      long queueWait = nowNanos - batch.enqueuedNanos();
      batchesByReason[reason.ordinal()]++;
      rowsByReason[reason.ordinal()] += batch.rows().size();
      totalOldestRowWaitNanos += queueWait;
      maxOldestRowWaitNanos = Math.max(maxOldestRowWaitNanos, queueWait);
      return new AdmittedBatch<>(batch, reason, attempt.ingress(), queueWait);
    }
  }

  /** 監視対象に容量の解放通知を追加する。いずれかの評価器で容量が空いたら、待機中の処理を再開できるよう通知する。 */
  private void prepareCapacityWake() {
    if (nextIngressAvailable.isDone()) {
      nextIngressAvailable = new CompletableFuture<>();
      observedCapacity.clear();
    }
    CompletableFuture<Void> wake = nextIngressAvailable;
    for (CompletableFuture<Void> available : availabilitySignals) {
      if (observedCapacity.put(available, Boolean.TRUE) == null) {
        available.whenComplete(
            (ignored, failure) -> {
              if (failure == null) wake.complete(null);
              else wake.completeExceptionally(failure);
            });
      }
    }
  }

  /** スケジューラーが所有する行集合です。エンコーダーと実行先には同じ参照を渡します。 */
  public record AdmittedBatch<E, K, R>(
      InferenceBatchQueue.Batch<E, K, R> batch,
      BatchDispatchReason reason,
      DecisionInferenceIngress ingress,
      long queueWaitNanos) {
    public E evaluator() {
      return batch.evaluator();
    }

    public K key() {
      return batch.key();
    }

    public List<R> rows() {
      return batch.rows();
    }
  }

  /** 待ち時間は各バッチの最古行について集計し、全行の平均待ちとは区別します。 */
  public record Metrics(
      long fullBatches,
      long fullRows,
      long supplyBatches,
      long supplyRows,
      long deadlineBatches,
      long deadlineRows,
      long totalOldestRowWaitNanos,
      long maxOldestRowWaitNanos) {
    public static Metrics empty() {
      return new Metrics(0, 0, 0, 0, 0, 0, 0, 0);
    }

    public Metrics plus(Metrics other) {
      return new Metrics(
          fullBatches + other.fullBatches,
          fullRows + other.fullRows,
          supplyBatches + other.supplyBatches,
          supplyRows + other.supplyRows,
          deadlineBatches + other.deadlineBatches,
          deadlineRows + other.deadlineRows,
          totalOldestRowWaitNanos + other.totalOldestRowWaitNanos,
          Math.max(maxOldestRowWaitNanos, other.maxOldestRowWaitNanos));
    }
  }
}
