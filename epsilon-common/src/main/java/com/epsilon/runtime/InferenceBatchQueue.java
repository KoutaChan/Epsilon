package com.epsilon.runtime;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.function.Predicate;
import java.util.function.ToIntBiFunction;

/** 評価器の同一性と系列の互換キーごとに、借用した行と投入時刻を保持します。 */
public final class InferenceBatchQueue<E, K, R> {
  private final long maxWaitNanos;
  private final ToIntBiFunction<E, K> batchSize;
  private final IdentityHashMap<E, LinkedHashMap<K, RowQueue<R>>> queues = new IdentityHashMap<>();
  private final ArrayList<E> evaluators = new ArrayList<>();
  private int queuedRows;
  private int nextEvaluator;

  public InferenceBatchQueue(long maxWaitNanos, ToIntBiFunction<E, K> batchSize) {
    this.maxWaitNanos = maxWaitNanos;
    this.batchSize = batchSize;
  }

  public void add(E evaluator, K key, R row, long nowNanos) {
    var keys = queues.get(evaluator);
    if (keys == null) {
      keys = new LinkedHashMap<>();
      queues.put(evaluator, keys);
      evaluators.add(evaluator);
    }
    keys.computeIfAbsent(key, ignored -> new RowQueue<>()).add(row, nowNanos);
    queuedRows++;
  }

  /** 行を取り出さず、次に優先する評価器の位置も変更せずに、送信するバッチ候補を選ぶ。期限による送信では、全評価器の中で最も古い行を含む候補を優先する。 */
  public Candidate<E, K, R> peekBest(
      BatchDispatchReason reason, Predicate<E> evaluatorAllowed, long nowNanos) {
    E selectedEvaluator = null;
    K selectedKey = null;
    RowQueue<R> selectedRows = null;
    int selectedIndex = 0;
    double selectedFill = 0;
    for (int offset = 0; offset < evaluators.size(); offset++) {
      int index = (nextEvaluator + offset) % evaluators.size();
      E evaluator = evaluators.get(index);
      if (!evaluatorAllowed.test(evaluator)) continue;
      for (var entry : queues.get(evaluator).entrySet()) {
        RowQueue<R> rows = entry.getValue();
        if (rows.size == 0) continue;
        int capacity = batchSize.applyAsInt(evaluator, entry.getKey());
        if (reason == BatchDispatchReason.DEADLINE && nowNanos - rows.oldest() < maxWaitNanos)
          continue;
        if (reason == BatchDispatchReason.FULL && rows.size < capacity) continue;
        double fill = Math.min(rows.size, capacity) / (double) capacity;
        boolean higherFill = reason == BatchDispatchReason.SUPPLY && fill > selectedFill;
        boolean samePriority = reason != BatchDispatchReason.SUPPLY || fill == selectedFill;
        if (selectedRows == null
            || higherFill
            || (samePriority && rows.oldest() < selectedRows.oldest())) {
          selectedEvaluator = evaluator;
          selectedKey = entry.getKey();
          selectedRows = rows;
          selectedIndex = index;
          selectedFill = fill;
        }
      }
    }
    return selectedRows == null
        ? null
        : new Candidate<>(selectedEvaluator, selectedKey, selectedRows, selectedIndex);
  }

  /** 確保済みの容量に収まる行を投入順に取り出す。残る行の投入時刻は維持する。 */
  public Batch<E, K, R> remove(Candidate<E, K, R> candidate) {
    return remove(candidate, null, 0);
  }

  Batch<E, K, R> remove(
      Candidate<E, K, R> candidate, InputBatchProfile.Selection profile, long nowNanos) {
    RowQueue<R> source = candidate.rows;
    int capacity = batchSize.applyAsInt(candidate.evaluator, candidate.key);
    int count = Math.min(source.size, capacity);
    long oldest = source.oldest();
    var rows = new ArrayList<R>(count);
    if (profile == null) {
      for (int row = 0; row < count; row++) rows.add(source.remove());
    } else {
      profile.capacity = capacity;
      for (int row = 0; row < count; row++) {
        profile.totalRowWaitNanos += nowNanos - source.oldest();
        rows.add(source.remove());
      }
    }
    queuedRows -= count;
    nextEvaluator = (candidate.evaluatorIndex + 1) % evaluators.size();
    return new Batch<>(candidate.evaluator, candidate.key, rows, oldest);
  }

  public boolean hasQueuedRows() {
    return queuedRows > 0;
  }

  /** 最古の待機行の期限を返します。空ならLong.MAX_VALUEです。 */
  public long nextDeadlineNanos(Predicate<E> evaluatorAllowed) {
    long oldest = Long.MAX_VALUE;
    for (var evaluator : queues.entrySet()) {
      if (!evaluatorAllowed.test(evaluator.getKey())) continue;
      for (var rows : evaluator.getValue().values()) {
        if (rows.size != 0) oldest = Math.min(oldest, rows.oldest());
      }
    }
    return oldest == Long.MAX_VALUE ? Long.MAX_VALUE : oldest + maxWaitNanos;
  }

  public record Batch<E, K, R>(E evaluator, K key, List<R> rows, long enqueuedNanos) {}

  public static final class Candidate<E, K, R> {
    private final E evaluator;
    private final K key;
    private final RowQueue<R> rows;
    private final int evaluatorIndex;

    private Candidate(E evaluator, K key, RowQueue<R> rows, int evaluatorIndex) {
      this.evaluator = evaluator;
      this.key = key;
      this.rows = rows;
      this.evaluatorIndex = evaluatorIndex;
    }

    public E evaluator() {
      return evaluator;
    }

    public K key() {
      return key;
    }
  }

  /** 行ごとのラッパーオブジェクトを作らず、参照とプリミティブ型時刻を同じ循環位置に保持します。 */
  private static final class RowQueue<R> {
    private Object[] rows = new Object[16];
    private long[] times = new long[16];
    private int head;
    private int size;

    private void add(R row, long nowNanos) {
      if (size == rows.length) grow();
      int tail = (head + size) % rows.length;
      rows[tail] = row;
      times[tail] = nowNanos;
      size++;
    }

    private long oldest() {
      return times[head];
    }

    @SuppressWarnings("unchecked")
    private R remove() {
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
