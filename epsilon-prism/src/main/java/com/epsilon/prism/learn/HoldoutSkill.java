package com.epsilon.prism.learn;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.model.PrismModel;
import com.epsilon.prism.rows.PrismRow;
import com.epsilon.prism.rows.RowSource;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * 保留の行での手の差の当てはまり（毎反復の保留の指標）。保留（HOLDOUT）のラベル行で値を測った手 C が 2 つ
 * 以上の行について、C の全ての対 (a, b)（順序なし）の予測の差 g と測った差 m = Y_a − Y_b を比べる。s は網の出力（効用の単位）。
 *
 * <p>予測の差は 2 通り出す。差分整合（{@link com.epsilon.prism.model.PrismLoss}）の最適は s = Y + λ_c·log p0_C + 行の定数なので、完全に学べた網でも
 * s の差は m + λ_c·Δlog p0 になる（α = 4 なら向聴 1 つで打牌 0.04、確約 0.08 ずれ、典型の手の差より大きい）。
 *
 * <ul>
 *   <li>raw（{@link Group#pairs}）: g = s(a) − s(b)。
 *   <li>debiased（{@link Group#debiased}）: g = u(a) − u(b)、u = s − λ_c·log max(p0, 1e-30)。手の価値の差の当てはまり（論文 E0 (c) の較正の傾き）は
 *       こちらで読む。
 * </ul>
 *
 * どちらも次の量を出す。
 *
 * <ul>
 *   <li>skill S = 対の平均 (2·g·m − g²)。E[m²] − E[(m − g)²] の不偏な推定で、大きいほど良い（g = 0 なら 0）。標準誤差は行で束ねる: S = ΣT_i / Σn_i
 *       （T_i は行 i の対の和、n_i は対の数）の線形化で se² = R/(R−1)·Σ(T_i − S·n_i)² / (Σn_i)²。
 *   <li>傾き（m を g に回帰）と相関は原点を通す: Σgm/Σg²、Σgm/√(Σg²·Σm²)。
 *   <li>符号の一致は m ≠ 0 の対で g·m &gt; 0 の割合（g = 0 は 1/2）。
 *   <li>決着がついた行（|Y_best − Y_second| &gt; {@value #DECIDED_GAP}）で argmax_C s = argmax_C Y の割合（実際に打つ比べ方なので s のまま。s の
 *       同点は p0 の大きい手、Y の同点は番号の小さい手）。
 * </ul>
 */
public final class HoldoutSkill {
  /** 決着がついた行の閾値 |Y_best − Y_second|。 */
  public static final double DECIDED_GAP = 0.05;

  private HoldoutSkill() {}

  /**
   * 対の指標。
   *
   * @param rows |C| ≥ 2 の行の数
   * @param pairs 対の数
   * @param skill S
   * @param skillError S の標準誤差（行で束ねる。行が 2 未満なら NaN）
   * @param meanSquaredGap 対の平均 m²（S の上限）
   * @param explained S / 平均 m²
   * @param slope m を g に回帰した傾き（原点を通す）
   * @param correlation g と m の相関（原点を通す）
   * @param signAgreement m ≠ 0 の対で符号が一致した割合
   * @param signPairs m ≠ 0 の対の数
   */
  public record Pairs(
      long rows,
      long pairs,
      double skill,
      double skillError,
      double meanSquaredGap,
      double explained,
      double slope,
      double correlation,
      double signAgreement,
      long signPairs) {}

  /**
   * 決着がついた行の一位の一致。
   *
   * @param rows |Y_best − Y_second| &gt; {@value #DECIDED_GAP} の行の数
   * @param agree argmax_C s = argmax_C Y の行の数
   * @param rate agree / rows
   */
  public record Decided(long rows, long agree, double rate) {}

  /**
   * 1 つの区分の指標。
   *
   * @param pairs g = s の差（raw）
   * @param debiased g = u の差（u = s − λ_c·log p0。事前の傾きを除いた手の価値の差）
   */
  public record Group(Pairs pairs, Pairs debiased, Decided decided) {}

  /**
   * 報告。
   *
   * @param rows 測った行の数
   * @param all 全体
   * @param byClass 判断の類ごと（行のない類は載せない）
   * @param bySource 行の種類ごと（MAIN・ROLLIN。ROLLIN は写しの中の逸脱した席の行で、論文 E0 (c) の較正の傾きを見る。行のない種類は載せない）
   */
  public record Report(
      int rows,
      Group all,
      Map<String, Group> byClass,
      Map<String, Group> bySource,
      double seconds) {}

  /**
   * rows を全て測る（|C| &lt; 2 の行は数えない）。
   *
   * @param rows 保留のラベル行で測った手が 2 つ以上の行（呼び出し側が索引で選ぶ）
   * @param settings 類ごとの傾き（{@link FitSettings#tilt}。debiased の u に使う。一様の基準なら 0 で debiased = raw）
   */
  public static Report measure(
      RowSource source,
      BatchAssembler assembler,
      PrismModel model,
      int[] rows,
      int batchSize,
      FitSettings settings) {
    long started = System.nanoTime();
    if (batchSize < 1) throw new IllegalArgumentException("batchSize must be positive: " + batchSize);
    var all = new Accumulator();
    var classes = new EnumMap<DecisionClass, Accumulator>(DecisionClass.class);
    var sources = new EnumMap<PrismRow.Source, Accumulator>(PrismRow.Source.class);
    int n = rows.length;
    if (n > 0) {
      CompletableFuture<BatchAssembler.Decoded> next =
          assembler.decode(source, BatchAssembler.batch(rows, 0, batchSize));
      for (int from = 0; from < n; from += batchSize) {
        BatchAssembler.Decoded decoded = StudentFit.await(next);
        if (from + batchSize < n)
          next = assembler.decode(source, BatchAssembler.batch(rows, from + batchSize, batchSize));
        float[][] scores = model.scores(Arrays.asList(decoded.hosts()));
        for (int j = 0; j < decoded.rows().length; j++) {
          PrismRow row = decoded.rows()[j];
          if (row.measuredCount() < 2) continue;
          if (scores[j].length != row.actions())
            throw new IllegalStateException(
                "Row of game "
                    + row.gameId()
                    + " has "
                    + row.actions()
                    + " actions but the model scored "
                    + scores[j].length);
          float lambda = settings.tilt(row.decisionClass());
          all.add(scores[j], row.values(), row.prior(), lambda);
          classes
              .computeIfAbsent(row.decisionClass(), ignored -> new Accumulator())
              .add(scores[j], row.values(), row.prior(), lambda);
          sources
              .computeIfAbsent(row.source(), ignored -> new Accumulator())
              .add(scores[j], row.values(), row.prior(), lambda);
        }
      }
    }
    var byClass = new LinkedHashMap<String, Group>();
    for (var entry : classes.entrySet()) byClass.put(entry.getKey().name(), entry.getValue().group());
    var bySource = new LinkedHashMap<String, Group>();
    for (var entry : sources.entrySet())
      bySource.put(entry.getKey().name(), entry.getValue().group());
    return new Report(n, all.group(), byClass, bySource, (System.nanoTime() - started) / 1e9);
  }

  /** 1 つの区分の和。対の指標は raw と debiased の 2 組（{@link PairSums}）。 */
  static final class Accumulator {
    private final PairSums raw = new PairSums(), debiased = new PairSums();
    private long decidedRows, decidedAgree;

    /**
     * 1 行を足す（|C| &lt; 2 なら何もしない）。
     *
     * @param scores 合法手ごとの s
     * @param values 合法手ごとの Y（NaN は測っていない手）
     * @param prior 合法手ごとの p0（debiased の u と、s の同点の割り）
     * @param lambda 行の類の λ_c
     */
    void add(float[] scores, float[] values, float[] prior, float lambda) {
      int[] measured = new int[values.length];
      int count = 0;
      for (int a = 0; a < values.length; a++) if (!Float.isNaN(values[a])) measured[count++] = a;
      if (count < 2) return;
      double[] tilted = new double[values.length];
      for (int i = 0; i < count; i++) {
        int a = measured[i];
        tilted[a] = (double) scores[a] - lambda * Math.log(Math.max(prior[a], 1e-30));
      }
      raw.begin();
      debiased.begin();
      for (int i = 0; i < count; i++) {
        for (int k = i + 1; k < count; k++) {
          int a = measured[i], b = measured[k];
          double m = (double) values[a] - values[b];
          raw.pair((double) scores[a] - scores[b], m);
          debiased.pair(tilted[a] - tilted[b], m);
        }
      }
      raw.end();
      debiased.end();
      // 一位（Y の同点は添字の小さい手）と二位の値。予測は実際に打つ比べ方（s、同点は p0）
      int best = measured[0], predicted = measured[0];
      for (int i = 1; i < count; i++) {
        int a = measured[i];
        if (values[a] > values[best]) best = a;
        if (Greedy.before(a, predicted, scores, prior)) predicted = a;
      }
      double second = Double.NEGATIVE_INFINITY;
      for (int i = 0; i < count; i++)
        if (measured[i] != best) second = Math.max(second, values[measured[i]]);
      if (values[best] - second > DECIDED_GAP) {
        decidedRows++;
        if (predicted == best) decidedAgree++;
      }
    }

    Group group() {
      return new Group(
          raw.pairs(),
          debiased.pairs(),
          new Decided(
              decidedRows,
              decidedAgree,
              decidedRows > 0 ? decidedAgree / (double) decidedRows : Double.NaN));
    }
  }

  /** 対の指標の和。行ごとに {@link #begin}・{@link #pair}…・{@link #end} で足す。 */
  static final class PairSums {
    private long rows, pairs, signPairs;
    private double sumT, sumN, sumT2, sumTN, sumN2, sumGM, sumGG, sumMM, signAgree;
    private double t;
    private int n;

    void begin() {
      t = 0;
      n = 0;
    }

    /** 予測の差 g と測った差 m の対を 1 つ足す。 */
    void pair(double g, double m) {
      t += 2 * g * m - g * g;
      n++;
      sumGM += g * m;
      sumGG += g * g;
      sumMM += m * m;
      if (m != 0) {
        signPairs++;
        signAgree += g * m > 0 ? 1 : g == 0 ? 0.5 : 0;
      }
    }

    void end() {
      rows++;
      pairs += n;
      sumT += t;
      sumN += n;
      sumT2 += t * t;
      sumTN += t * n;
      sumN2 += (double) n * n;
    }

    Pairs pairs() {
      double skill = sumN > 0 ? sumT / sumN : Double.NaN;
      double error = Double.NaN;
      if (rows >= 2) {
        double squares = Math.max(0, sumT2 - 2 * skill * sumTN + skill * skill * sumN2);
        error = Math.sqrt(rows / (rows - 1.0) * squares) / sumN;
      }
      double meanSquared = sumN > 0 ? sumMM / sumN : Double.NaN;
      return new Pairs(
          rows,
          pairs,
          skill,
          error,
          meanSquared,
          skill / meanSquared,
          sumGG > 0 ? sumGM / sumGG : Double.NaN,
          sumGG > 0 && sumMM > 0 ? sumGM / Math.sqrt(sumGG * sumMM) : Double.NaN,
          signPairs > 0 ? signAgree / signPairs : Double.NaN,
          signPairs);
    }
  }
}
