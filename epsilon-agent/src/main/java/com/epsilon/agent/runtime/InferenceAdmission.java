package com.epsilon.agent.runtime;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.function.Predicate;
import java.util.function.ToIntBiFunction;

/**
 * 推論を待つ行を評価器と区分ごとに並べ、いつバッチにして出すかを決める。行をキューから取り出すのは、容量を予約できたときだけ。
 *
 * <p>バッチを出す順は、最も古い行が待てる上限を過ぎたバッチ（{@link Reason#DEADLINE}）、満杯のバッチ（{@link Reason#FULL}）、
 * 満たないバッチ（{@link Reason#SUPPLY}）。満たないバッチを出すのは、入力を作る処理が止まっていて、評価器の実行先でバッチが
 * 足りないときだけ。入力を作る処理が動いている間は、満たないバッチは出さずに行が溜まるのを待つ。呼んでよいのは 1 本のスレッド
 * （収集のドライバ）だけ。
 *
 * @param <E> 評価器（同一性で区別する）
 * @param <K> バッチの区分（equals で区別する）
 * @param <R> 行
 */
public final class InferenceAdmission<E, K, R> {

  /** バッチを出した理由。 */
  public enum Reason {
    FULL,
    SUPPLY,
    DEADLINE
  }

  private final RowQueues<E, K, R> queues;
  private final BiFunction<E, K, DecisionInferenceIngress.Attempt> acquire;
  private final IdentityHashMap<E, Boolean> blocked = new IdentityHashMap<>();
  private final List<CompletableFuture<Void>> blockedWakes = new ArrayList<>();
  private final IdentityHashMap<CompletableFuture<Void>, Boolean> watched = new IdentityHashMap<>();
  private final Predicate<E> unblocked = evaluator -> !blocked.containsKey(evaluator);
  private final Predicate<E> starving;
  private final long[] batchesByReason = new long[Reason.values().length];
  private final long[] rowsByReason = new long[Reason.values().length];
  private long totalOldestWaitNanos;
  private long maxOldestWaitNanos;
  private CompletableFuture<Void> wake = new CompletableFuture<>();
  private int inFlight;

  /**
   * @param maxWaitNanos 行がバッチを待てる上限（ナノ秒）
   * @param batchSize 評価器と区分ごとに、バッチに入る行数の上限を返す
   * @param acquire 評価器の容量を、待たずに予約する処理
   * @param needsWork 評価器の実行先でバッチが足りないなら true を返す
   */
  public InferenceAdmission(
      long maxWaitNanos,
      ToIntBiFunction<E, K> batchSize,
      BiFunction<E, K, DecisionInferenceIngress.Attempt> acquire,
      Predicate<E> needsWork) {
    if (maxWaitNanos < 0) {
      throw new IllegalArgumentException("The batch wait must not be negative: " + maxWaitNanos);
    }
    queues = new RowQueues<>(maxWaitNanos, batchSize);
    this.acquire = acquire;
    starving = evaluator -> unblocked.test(evaluator) && needsWork.test(evaluator);
  }

  public void add(E evaluator, K key, R row, long nowNanos) {
    queues.add(evaluator, key, row, nowNanos);
  }

  /**
   * 次のバッチの容量を予約して、行をキューから取り出す。
   *
   * @param generating 入力を作る処理がまだ動いているなら true。動いている間は満たないバッチを出さない
   * @return 取り出したバッチ。出せるバッチが無いか、出せるバッチの評価器が全て容量を待っていれば null
   */
  public Admitted<E, K, R> admit(long nowNanos, boolean generating) {
    blocked.clear();
    blockedWakes.clear();
    Admitted<E, K, R> admitted = tryAdmit(Reason.DEADLINE, unblocked, nowNanos);
    if (admitted == null) {
      admitted = tryAdmit(Reason.FULL, unblocked, nowNanos);
    }
    if (admitted == null && !generating) {
      admitted = tryAdmit(Reason.SUPPLY, starving, nowNanos);
    }
    watchBlockedWakes();
    return admitted;
  }

  /**
   * 符号化に失敗したバッチを、出したバッチの数から外す。予約は、まだ移されても返されてもいなければ返す。評価器が失敗の前に返すか
   * 受け取った予約には触らない。
   */
  public void encodingFailed(Admitted<E, K, R> admitted) {
    inFlight--;
    admitted.ingress().abortIfOpen();
  }

  /** 出したバッチの推論が終わったことを知らせる。予約は実行先が返すので、ここでは返さない。 */
  public void inferenceFinished() {
    inFlight--;
  }

  /** 容量を待っている評価器のどれかで、容量が空いたときに完了する通知を返す。 */
  public CompletableFuture<Void> ingressAvailable() {
    return wake;
  }

  /** 直前にバッチを取り出そうとしたとき、容量を待つ評価器があったなら true。 */
  public boolean waitingForIngress() {
    return !blockedWakes.isEmpty();
  }

  /**
   * 渡したキューのイベントを待つ。容量を待っていない評価器の行のうち、最も古い行が先に期限を迎えたら null を返す。呼び出し側は、そこで
   * もう一度バッチを出す。
   */
  public <T> T awaitEvent(BlockingQueue<T> events) throws InterruptedException {
    long deadline = queues.nextDeadlineNanos(unblocked);
    return deadline == Long.MAX_VALUE
        ? events.take()
        : events.poll(Math.max(0, deadline - System.nanoTime()), TimeUnit.NANOSECONDS);
  }

  public boolean hasQueuedRows() {
    return queues.size > 0;
  }

  public boolean hasInFlightBatches() {
    return inFlight > 0;
  }

  public Metrics metrics() {
    Map<Reason, Metrics.Count> admitted = new EnumMap<>(Reason.class);
    for (Reason reason : Reason.values()) {
      admitted.put(reason, new Metrics.Count(batchesByReason[reason.ordinal()], rowsByReason[reason.ordinal()]));
    }
    return new Metrics(admitted, totalOldestWaitNanos, maxOldestWaitNanos);
  }

  private Admitted<E, K, R> tryAdmit(Reason reason, Predicate<E> allowed, long nowNanos) {
    while (true) {
      RowQueues.Candidate<E, K, R> candidate = queues.best(reason, allowed, nowNanos);
      if (candidate == null) {
        return null;
      }
      DecisionInferenceIngress.Attempt attempt = acquire.apply(candidate.evaluator(), candidate.key());
      if (!attempt.acquired()) {
        blocked.put(candidate.evaluator(), Boolean.TRUE);
        blockedWakes.add(attempt.available());
        continue;
      }
      long oldest = candidate.rows().oldest();
      List<R> rows;
      try {
        rows = queues.remove(candidate);
      } catch (RuntimeException | Error failure) {
        attempt.ingress().abort();
        throw failure;
      }
      inFlight++;
      long waited = nowNanos - oldest;
      batchesByReason[reason.ordinal()]++;
      rowsByReason[reason.ordinal()] += rows.size();
      totalOldestWaitNanos += waited;
      maxOldestWaitNanos = Math.max(maxOldestWaitNanos, waited);
      return new Admitted<>(candidate.evaluator(), candidate.key(), rows, reason, attempt.ingress(), waited);
    }
  }

  /** 容量を待つ評価器の通知を、容量の空きを知らせる通知につなぐ。すでにつないだ通知は、もう一度つながない。 */
  private void watchBlockedWakes() {
    if (wake.isDone()) {
      wake = new CompletableFuture<>();
      watched.clear();
    }
    CompletableFuture<Void> target = wake;
    for (CompletableFuture<Void> available : blockedWakes) {
      if (watched.put(available, Boolean.TRUE) == null) {
        available.whenComplete(
            (ignored, failure) -> {
              if (failure == null) {
                target.complete(null);
              } else {
                target.completeExceptionally(failure);
              }
            });
      }
    }
  }

  /**
   * 容量を予約して、キューから取り出したバッチ。
   *
   * @param rows バッチの行。符号化と実行先には同じリストを渡す
   * @param queueWaitNanos バッチで最も古い行が待った時間（ナノ秒）
   */
  public record Admitted<E, K, R>(
      E evaluator, K key, List<R> rows, Reason reason, DecisionInferenceIngress ingress, long queueWaitNanos) {}

  /**
   * 出したバッチの集計。待ち時間は、各バッチで最も古い行が待った時間を集めたもので、全ての行の待ち時間の平均ではない。
   *
   * @param admitted 理由ごとのバッチ数と行数
   */
  public record Metrics(Map<Reason, Count> admitted, long totalOldestWaitNanos, long maxOldestWaitNanos) {

    public record Count(long batches, long rows) {}

    public long batches() {
      return admitted.values().stream().mapToLong(Count::batches).sum();
    }
  }

  /** 評価器と区分ごとに行を並べるキュー。評価器は同一性で、区分は equals で区別する。 */
  private static final class RowQueues<E, K, R> {
    private final long maxWaitNanos;
    private final ToIntBiFunction<E, K> batchSize;
    /** 評価器の一覧。評価器ごとの区分のキューは、下のリストの同じ添字に置く。評価器は数個なので、同一性で前から探す。 */
    private final List<E> evaluators = new ArrayList<>();

    private final List<LinkedHashMap<K, Ring<R>>> keysOf = new ArrayList<>();
    private int size;
    private int nextEvaluator;

    RowQueues(long maxWaitNanos, ToIntBiFunction<E, K> batchSize) {
      this.maxWaitNanos = maxWaitNanos;
      this.batchSize = batchSize;
    }

    void add(E evaluator, K key, R row, long nowNanos) {
      int index = evaluators.size() - 1;
      while (index >= 0 && evaluators.get(index) != evaluator) {
        index--;
      }
      LinkedHashMap<K, Ring<R>> keys;
      if (index < 0) {
        keys = new LinkedHashMap<>();
        evaluators.add(evaluator);
        keysOf.add(keys);
      } else {
        keys = keysOf.get(index);
      }
      Ring<R> rows = keys.get(key);
      if (rows == null) {
        rows = new Ring<>();
        keys.put(key, rows);
      }
      rows.add(row, nowNanos);
      size++;
    }

    /**
     * 出す理由に合うバッチの候補を、行を取り出さずに選ぶ。期限切れと満杯のバッチでは、最も古い行を持つ候補を選ぶ。満たないバッチでは、
     * 最も埋まった候補を選び、埋まり方が同じなら最も古い行を持つ候補を選ぶ。評価器は、前に出した評価器の次から順に見る。
     */
    Candidate<E, K, R> best(Reason reason, Predicate<E> allowed, long nowNanos) {
      E selectedEvaluator = null;
      K selectedKey = null;
      Ring<R> selectedRows = null;
      int selectedIndex = 0;
      double selectedFill = 0;
      for (int offset = 0; offset < evaluators.size(); offset++) {
        int index = (nextEvaluator + offset) % evaluators.size();
        E evaluator = evaluators.get(index);
        if (!allowed.test(evaluator)) {
          continue;
        }
        for (Map.Entry<K, Ring<R>> entry : keysOf.get(index).entrySet()) {
          Ring<R> rows = entry.getValue();
          if (rows.size == 0) {
            continue;
          }
          int capacity = batchSize.applyAsInt(evaluator, entry.getKey());
          if (reason == Reason.DEADLINE && nowNanos - rows.oldest() < maxWaitNanos) {
            continue;
          }
          if (reason == Reason.FULL && rows.size < capacity) {
            continue;
          }
          double fill = Math.min(rows.size, capacity) / (double) capacity;
          boolean fuller = reason == Reason.SUPPLY && fill > selectedFill;
          boolean tied = reason != Reason.SUPPLY || fill == selectedFill;
          if (selectedRows == null || fuller || (tied && rows.oldest() < selectedRows.oldest())) {
            selectedEvaluator = evaluator;
            selectedKey = entry.getKey();
            selectedRows = rows;
            selectedIndex = index;
            selectedFill = fill;
          }
        }
      }
      return selectedRows == null ? null : new Candidate<>(selectedEvaluator, selectedKey, selectedRows, selectedIndex);
    }

    /** 候補のバッチに入る分の行を、入れた順に取り出す。残った行の入れた時刻は変えない。 */
    List<R> remove(Candidate<E, K, R> candidate) {
      Ring<R> source = candidate.rows();
      int count = Math.min(source.size, batchSize.applyAsInt(candidate.evaluator(), candidate.key()));
      List<R> rows = new ArrayList<>(count);
      for (int row = 0; row < count; row++) {
        rows.add(source.remove());
      }
      size -= count;
      nextEvaluator = (candidate.evaluatorIndex() + 1) % evaluators.size();
      return rows;
    }

    /** 対象の評価器の行のうち、最も古い行の期限を返す。行が無ければ Long.MAX_VALUE。 */
    long nextDeadlineNanos(Predicate<E> allowed) {
      long oldest = Long.MAX_VALUE;
      for (int index = 0; index < evaluators.size(); index++) {
        if (!allowed.test(evaluators.get(index))) {
          continue;
        }
        for (Ring<R> rows : keysOf.get(index).values()) {
          if (rows.size != 0) {
            oldest = Math.min(oldest, rows.oldest());
          }
        }
      }
      return oldest == Long.MAX_VALUE ? Long.MAX_VALUE : oldest + maxWaitNanos;
    }

    private record Candidate<E, K, R>(E evaluator, K key, Ring<R> rows, int evaluatorIndex) {}
  }

  /** 行と、行を入れた時刻を同じ添字に持つリングバッファ。行ごとに包むオブジェクトを作らない。 */
  private static final class Ring<R> {
    private Object[] rows = new Object[16];
    private long[] times = new long[16];
    private int head;
    private int size;

    void add(R row, long nowNanos) {
      if (size == rows.length) {
        grow();
      }
      int tail = (head + size) % rows.length;
      rows[tail] = row;
      times[tail] = nowNanos;
      size++;
    }

    long oldest() {
      return times[head];
    }

    @SuppressWarnings("unchecked")
    R remove() {
      R row = (R) rows[head];
      rows[head] = null;
      head = (head + 1) % rows.length;
      size--;
      return row;
    }

    private void grow() {
      Object[] nextRows = new Object[rows.length * 2];
      long[] nextTimes = new long[times.length * 2];
      int first = rows.length - head;
      System.arraycopy(rows, head, nextRows, 0, first);
      System.arraycopy(rows, 0, nextRows, first, head);
      System.arraycopy(times, head, nextTimes, 0, first);
      System.arraycopy(times, 0, nextTimes, first, head);
      rows = nextRows;
      times = nextTimes;
      head = 0;
    }
  }
}
