package com.epsilon.pico.ai.decision.benchmark;

import com.epsilon.pico.ai.decision.input.DecisionBucket;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Consumer;

/**
 * 実際に推論した行から無作為抽出した入力を保存し、再現可能な性能測定へ渡す。
 *
 * <p>抽出件数を固定するリザーバーサンプリングを使い、採取時だけ独立した記憶領域へ複製する。確定後はデータを変更しない。必要な行数が保存行数を超える場合は、保存した行を先頭から繰り返して任意サイズのバッチを作る。
 */
public final class EpsilonDecisionInferenceReplayCorpus {

  static final int PRIMARY_IDENTITY_ROWS = 4096;
  static final int PRIMARY_TRANSITION_ROWS = 2048;
  static final int OTHER_BUCKET_ROWS = 512;
  private static final Comparator<DecisionBucket> BUCKET_ORDER =
      Comparator.comparingInt(DecisionBucket::legalActionCapacity)
          .thenComparingInt(DecisionBucket::actionTransitionCapacity);

  private EpsilonDecisionInferenceReplayCorpus() {}

  /** 採取元モデルの役割。 */
  enum Role {
    /** 自己対局方策の方策と価値を返す推論。 */
    ACTOR_POLICY_AND_VALUE,
    /** 対戦相手の方策だけを返す推論。 */
    OPPONENT_POLICY_ONLY
  }

  /**
   * 一つの役割について実RowBatchを同期採取します。
   *
   * <p>{@link #accept}から戻った時点で必要な行の独立複製は完了しています。観測者の呼出元は元バッチの寿命を延長する必要がありません。
   */
  static final class Recorder implements Consumer<DecisionHostBatch.RowBatch> {

    private final Role role;
    private final Map<DecisionBucket, Integer> targetOverrides;
    private final Map<DecisionBucket, MutableBucket> buckets = new HashMap<>();
    private Corpus frozen;

    Recorder(Role role) {
      this(role, Map.of());
    }

    /**
     * 容量区分別保持上限を上書きしてrecorderを作ります。
     *
     * @param role 採取元モデルの役割
     * @param targetOverrides 既定以外の保持行数。未指定容量区分には標準上限を使う
     */
    Recorder(Role role, Map<DecisionBucket, Integer> targetOverrides) {
      this.role = java.util.Objects.requireNonNull(role, "role");
      HashMap<DecisionBucket, Integer> checkedTargets = new HashMap<>();
      targetOverrides.forEach(
          (bucket, rows) -> {
            if (rows == null || rows < 1) {
              throw new IllegalArgumentException("replay target rows must be positive: " + rows);
            }
            checkedTargets.put(java.util.Objects.requireNonNull(bucket, "bucket"), rows);
          });
      this.targetOverrides = Map.copyOf(checkedTargets);
    }

    /**
     * 観測者から渡された実行を容量区分上限まで同期採取します。
     *
     * <p>保持上限へ到達した後もobserved 行ヒストグラムは更新します。
     *
     * @param rows 推論サーバーが実行した1つの実バッチ
     */
    @Override
    public synchronized void accept(DecisionHostBatch.RowBatch rows) {
      if (frozen != null) {
        throw new IllegalStateException("replay corpus is already frozen");
      }
      MutableBucket bucket =
          buckets.computeIfAbsent(
              rows.bucket(),
              key ->
                  new MutableBucket(
                      targetOverrides.getOrDefault(key, defaultTargetRows(key)),
                      samplingSeed(role, key)));
      for (int part = 0; part < rows.sliceCount(); part++) {
        DecisionHostBatch.RowSlice slice = rows.slice(part);
        for (int row = 0; row < slice.size(); row++) {
          long sequence = Math.addExact(bucket.observedRows, 1L);
          bucket.observedRows = sequence;
          int selected = bucket.select(sequence);
          if (selected < 0) {
            continue;
          }
          DecisionHostBatch copy = slice.source().copyRows(slice.fromInclusive() + row, 1);
          if (selected == bucket.parts.size()) {
            bucket.parts.add(copy);
          } else {
            bucket.parts.set(selected, copy);
          }
        }
      }
    }

    /** 指定容量区分で現在までに観測した行数を返します。 */
    synchronized long observedRows(DecisionBucket bucket) {
      MutableBucket state = buckets.get(bucket);
      return state == null ? 0L : state.observedRows;
    }

    /** 指定容量区分で独立所有している行数を返します。 */
    synchronized int retainedRows(DecisionBucket bucket) {
      MutableBucket state = buckets.get(bucket);
      return state == null ? 0 : state.parts.size();
    }

    /** 指定容量区分が保持目標へ到達していれば{@code true}を返します。 */
    synchronized boolean targetReached(DecisionBucket bucket) {
      MutableBucket state = buckets.get(bucket);
      return state != null && state.parts.size() == state.targetRows;
    }

    /**
     * 採取済み行を容量区分ごとの連続記憶領域へ固定します。
     *
     * <p>二回目以降は同じ変更不可固定入力データを返します。固定後の{@link #accept}は拒否します。
     *
     * @return 役割別の固定再生固定入力データ
     */
    synchronized Corpus freeze() {
      if (frozen == null) {
        frozen = Corpus.freeze(role, buckets);
        buckets.clear();
      }
      return frozen;
    }
  }

  /** 役割別に固定した実入力固定入力データ。 */
  public static final class Corpus {

    private final Role role;
    private final Map<DecisionBucket, DecisionHostBatch> batches;
    private final Report report;

    private Corpus(Role role, Map<DecisionBucket, DecisionHostBatch> batches, Report report) {
      this.role = role;
      this.batches = batches;
      this.report = report;
    }

    private static Corpus freeze(Role role, Map<DecisionBucket, MutableBucket> mutableBuckets) {
      ArrayList<DecisionBucket> orderedBuckets = new ArrayList<>(mutableBuckets.keySet());
      orderedBuckets.sort(BUCKET_ORDER);
      LinkedHashMap<DecisionBucket, DecisionHostBatch> frozenBatches = new LinkedHashMap<>();
      LinkedHashMap<DecisionBucket, BucketStatistics> statistics = new LinkedHashMap<>();
      for (DecisionBucket bucket : orderedBuckets) {
        MutableBucket mutable = mutableBuckets.get(bucket);
        if (mutable.parts.isEmpty()) {
          continue;
        }
        DecisionHostBatch batch = DecisionHostBatch.concatenate(mutable.parts);
        frozenBatches.put(bucket, batch);
        statistics.put(
            bucket,
            new BucketStatistics(mutable.observedRows, mutable.parts.size(), mutable.targetRows));
      }
      Report report =
          new Report(
              role,
              statistics.values().stream().mapToLong(BucketStatistics::observedRows).sum(),
              statistics.values().stream().mapToInt(BucketStatistics::retainedRows).sum(),
              "DETERMINISTIC_RESERVOIR",
              Collections.unmodifiableMap(statistics));
      return new Corpus(role, Collections.unmodifiableMap(frozenBatches), report);
    }

    /** 永続化済みバッチと統計を検証し、複製せず変更不可固定入力データへ復元します。 */
    static Corpus restore(
        Role role,
        Map<DecisionBucket, DecisionHostBatch> restoredBatches,
        String sampling,
        Map<DecisionBucket, BucketStatistics> restoredStatistics) {
      ArrayList<DecisionBucket> orderedBuckets = new ArrayList<>(restoredBatches.keySet());
      orderedBuckets.sort(BUCKET_ORDER);
      LinkedHashMap<DecisionBucket, DecisionHostBatch> batches = new LinkedHashMap<>();
      LinkedHashMap<DecisionBucket, BucketStatistics> statistics = new LinkedHashMap<>();
      for (DecisionBucket bucket : orderedBuckets) {
        DecisionHostBatch batch = restoredBatches.get(bucket);
        BucketStatistics bucketStatistics = restoredStatistics.get(bucket);
        if (bucketStatistics == null || bucketStatistics.retainedRows() != batch.size()) {
          throw new IllegalArgumentException(
              "restored corpus bucket metadata does not match payload");
        }
        batches.put(bucket, batch);
        statistics.put(bucket, bucketStatistics);
      }
      if (!statistics.keySet().equals(restoredStatistics.keySet())) {
        throw new IllegalArgumentException("restored corpus has metadata without a payload");
      }
      Report report =
          new Report(
              role,
              statistics.values().stream().mapToLong(BucketStatistics::observedRows).sum(),
              statistics.values().stream().mapToInt(BucketStatistics::retainedRows).sum(),
              sampling,
              Collections.unmodifiableMap(statistics));
      return new Corpus(role, Collections.unmodifiableMap(batches), report);
    }

    /** 永続化境界へ渡す内部所有バッチを返します。 */
    DecisionHostBatch storedBatch(DecisionBucket bucket) {
      DecisionHostBatch batch = batches.get(bucket);
      if (batch == null) {
        throw new IllegalArgumentException("replay corpus has no real rows for bucket " + bucket);
      }
      return batch;
    }

    /** 固定入力データを採取した役割を返します。 */
    Role role() {
      return role;
    }

    /** 実行を保持する容量区分集合を標準形式の順で返します。 */
    Set<DecisionBucket> buckets() {
      return batches.keySet();
    }

    /** 指定容量区分の実行を一行以上保持していれば{@code true}を返します。 */
    public boolean hasBucket(DecisionBucket bucket) {
      return batches.containsKey(bucket);
    }

    /** 全容量区分で観測した実行数を返します。 */
    long observedRows() {
      return report.observedRows();
    }

    /** 指定容量区分で観測した実行数を返します。未観測容量区分は0です。 */
    long observedRows(DecisionBucket bucket) {
      BucketStatistics statistics = report.buckets().get(bucket);
      return statistics == null ? 0L : statistics.observedRows();
    }

    /**
     * 保持した実行だけから、指定行数の連続再生バッチを作ります。
     *
     * <p>保持行数を超える部分は先頭へ戻って循環します。毎回同じ行順となり、返却バッチは固定入力データから独立しています。
     *
     * @param bucket 再生する容量区分
     * @param rows 作成する行数
     * @return 標準形式の連続バッファを独立所有する連続バッチ
     */
    public DecisionHostBatch batch(DecisionBucket bucket, int rows) {
      if (rows < 1) {
        throw new IllegalArgumentException("replay rows must be positive: " + rows);
      }
      DecisionHostBatch source = batches.get(bucket);
      if (source == null) {
        throw new IllegalArgumentException("replay corpus has no real rows for bucket " + bucket);
      }
      if (rows <= source.size()) {
        return source.copyRows(0, rows);
      }
      int[] indexes = new int[rows];
      for (int row = 0; row < rows; row++) {
        indexes[row] = row % source.size();
      }
      return source.selectRows(indexes, rows);
    }

    /** 保持済み入力の先頭行を借用する。利用完了まで固定入力データを保持する。 */
    public DecisionHostBatch.RowBatch rows(DecisionBucket bucket, int rowCount) {
      return DecisionHostBatch.RowBatch.of(batches.get(bucket).sliceRows(0, rowCount));
    }

    /** ヒストグラムと保持量を返します。 */
    Report report() {
      return report;
    }
  }

  /** 一容量区分の観測・保持量。 */
  record BucketStatistics(long observedRows, int retainedRows, int targetRows) {}

  /** 一役割の固定入力データレポート。 */
  record Report(
      Role role,
      long observedRows,
      int retainedRows,
      String sampling,
      Map<DecisionBucket, BucketStatistics> buckets) {}

  private static final class MutableBucket {

    private final int targetRows;
    private final List<DecisionHostBatch> parts = new ArrayList<>();
    private long observedRows;
    private long randomState;

    private MutableBucket(int targetRows, long seed) {
      this.targetRows = targetRows;
      randomState = mix64(seed ^ targetRows ^ 0xD1B54A32D192ED03L);
    }

    private int select(long sequence) {
      if (parts.size() < targetRows) {
        return parts.size();
      }
      randomState = mix64(randomState + 0x9E3779B97F4A7C15L);
      long candidate = Long.remainderUnsigned(randomState, sequence);
      return candidate < targetRows ? (int) candidate : -1;
    }
  }

  private static long mix64(long value) {
    value = (value ^ (value >>> 30)) * 0xBF58476D1CE4E5B9L;
    value = (value ^ (value >>> 27)) * 0x94D049BB133111EBL;
    return value ^ (value >>> 31);
  }

  private static long samplingSeed(Role role, DecisionBucket bucket) {
    long key = ((long) role.ordinal() << 48);
    key ^= ((long) bucket.legalActionCapacity() << 24);
    key ^= bucket.actionTransitionCapacity();
    return mix64(key ^ 0xA0761D6478BD642FL);
  }

  static int defaultTargetRows(DecisionBucket bucket) {
    if (bucket.legalActionCapacity() == 16 && bucket.actionTransitionCapacity() == 1) {
      return PRIMARY_IDENTITY_ROWS;
    }
    if (bucket.legalActionCapacity() == 4 && bucket.actionTransitionCapacity() == 16) {
      return PRIMARY_TRANSITION_ROWS;
    }
    return OTHER_BUCKET_ROWS;
  }
}
