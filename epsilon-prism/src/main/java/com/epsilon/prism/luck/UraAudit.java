package com.epsilon.prism.luck;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.DecisionClass;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLongArray;

/**
 * 裏ドラの期待値を監査するための集計。局末の数、立直の和了で終わった局末の数、期待値を求めた局末の数、基本点の組の数、かかった時間を数える。
 * 期待値を求めた局末の行は、判断のクラスと槓の有無で区分に分ける。区分ごとに、実際の値と期待値の差 d の和と d² の和、裏ドラで決まる分散の和、
 * 実際の値の和とその 2 乗の和を持つ。
 *
 * <p>同じ局の行は、主経路・写し・分岐のどの局末の行も、同じ牌山の裏ドラ表示牌で d が決まるので独立ではない。d の平均の標準誤差は、局で
 * 束ねた分散から求める。局は、半荘 ID と局の序数のハッシュで {@link #CLUSTERS} 個の束に振り分け、束ごとに行の数と d の和を持つ。違う局の d は
 * 相関しないので、束の和も相関しない。区分の束 g の行の数を m_g、d の和を D_g、行のある束の数を G、行の数を N = Σm_g、d の平均を
 * d̄ = ΣD_g / N とすると、d̄ の分散は G/(G−1)·Σ_g (D_g − m_g·d̄)² / N²。どの束も行が 1 つなら、行を独立とみなした分散
 * Σ(d − d̄)² / (N·(N−1)) と同じになる。束が少ない区分では、行がいくつあっても標準誤差の見積もりが粗く、偏りが無くても |z| が大きく出る
 * （束が 2 つなら |z| > 3 が約 2 割）。z は、行のある束が {@link #MIN_CLUSTERS} 個以上の区分だけで求める。
 *
 * <p>和は 2^−32 を単位にした固定小数点の整数で足すので、どのスレッドがどの順で足しても同じ値になる。2 乗と積の和は、丸めた整数の 2 乗と積を
 * 整数のまま足す。分散は丸めた値について厳密に求まるので、値が全て同じ区分ではちょうど 0 になる。どのスレッドから呼んでもよい。
 */
public final class UraAudit {

  /** 基本点の組の数を数えるビンの数。最後のビンはそれ以上の数をまとめる。 */
  public static final int GROUP_BINS = 8;

  /** 局を振り分ける束の数。 */
  static final int CLUSTERS = 4096;

  /** 区分の z を求めるのに要る、行のある束の数。 */
  public static final int MIN_CLUSTERS = 30;

  private static final double SCALE = 0x1p32;
  private static final int KAN_STATES = 2;
  private static final int CELLS = DecisionClass.COUNT * KAN_STATES;

  /** 局末について数える値。 */
  private enum Count {
    HAND_ENDS,
    RIICHI_WINS,
    AVERAGED,
    NANOS
  }

  /** 監査に数える行。行の判断のクラスで区分を決め、行の席の値を数える。 */
  public interface Row {

    DecisionClass decisionClass();

    int seat();
  }

  private final AtomicLongArray counts = new AtomicLongArray(Count.values().length);
  private final AtomicLongArray groups = new AtomicLongArray(GROUP_BINS);

  /** [区分][束] → 行の数と d の和。 */
  private final ClusterSums clusters = new ClusterSums();

  /** [区分] → 裏ドラで決まる分散の和。 */
  private final AtomicLongArray variances = new AtomicLongArray(CELLS);

  /** [区分] → d の和と 2 乗の和。 */
  private final Moments differences = new Moments();

  /** [区分] → 実際の値の和と 2 乗の和。 */
  private final Moments actuals = new Moments();

  /** 局末を 1 つ数える。局末の値を求めた後に 1 度だけ呼ぶ。 */
  public void handEnd(UraExpectation values) {
    counts.incrementAndGet(Count.HAND_ENDS.ordinal());
    if (values.riichiWin()) counts.incrementAndGet(Count.RIICHI_WINS.ordinal());
    if (values.averaged()) {
      counts.incrementAndGet(Count.AVERAGED.ordinal());
      groups.incrementAndGet(Math.min(values.groups(), GROUP_BINS - 1));
      counts.addAndGet(Count.NANOS.ordinal(), values.nanos());
    }
  }

  /**
   * 期待値を求めた局末の行を数える。期待値を求めなかった局末では何もしない。
   *
   * @param values 局末の値
   * @param gameId 半荘 ID
   * @param hand 半荘の中の局の序数
   * @param rows 局末の値を受け取る行
   */
  public void rows(UraExpectation values, long gameId, int hand, List<? extends Row> rows) {
    if (!values.averaged()) return;
    boolean kan = values.kan();
    int cluster = cluster(gameId, hand);
    for (Row row : rows) {
      int seat = row.seat(), cell = cell(row.decisionClass(), kan);
      double actual = values.actual(seat);
      long difference = fixed(actual - values.expected(seat));
      differences.add(cell, difference);
      variances.addAndGet(cell, fixed(values.variance(seat)));
      actuals.add(cell, fixed(actual));
      clusters.add(cell, cluster, difference);
    }
  }

  /** ここまでの集計の写しを返す。 */
  public Report report() {
    long[] bins = new long[GROUP_BINS];
    for (int bin = 0; bin < GROUP_BINS; bin++) bins[bin] = groups.get(bin);
    List<Cell> cells = new ArrayList<>();
    for (DecisionClass decisionClass : DecisionClass.values()) {
      for (int kan = 0; kan < KAN_STATES; kan++) {
        int cell = cell(decisionClass, kan == 1);
        Clustered clustered = clusters.cell(cell);
        long cellRows = clustered.rows();
        if (cellRows == 0) continue;
        cells.add(
            Cell.of(
                decisionClass,
                kan == 1,
                cellRows,
                clustered.clusters(),
                differences.mean(cell, cellRows),
                clustered.meanVariance(),
                differences.meanSquare(cell, cellRows),
                variances.get(cell) / SCALE / cellRows,
                actuals.variance(cell, cellRows)));
      }
    }
    return new Report(
        counts.get(Count.HAND_ENDS.ordinal()),
        counts.get(Count.RIICHI_WINS.ordinal()),
        counts.get(Count.AVERAGED.ordinal()),
        bins,
        counts.get(Count.NANOS.ordinal()),
        List.copyOf(cells));
  }

  /** 値を 2^−32 を単位にした整数に丸める。 */
  private static long fixed(double value) {
    return Math.round(value * SCALE);
  }

  private static int cell(DecisionClass decisionClass, boolean kan) {
    return decisionClass.ordinal() * KAN_STATES + (kan ? 1 : 0);
  }

  /** 局を入れる束。半荘 ID と局の序数だけで決まるので、同じ局の主経路・写し・分岐の局末は同じ束に入る。 */
  static int cluster(long gameId, int hand) {
    return Math.floorMod(CrnSeeds.extend(gameId, hand), CLUSTERS);
  }

  /** 区分ごとの、丸めた値 q の和と q² の和。 */
  private static final class Moments {

    private final AtomicLongArray sums = new AtomicLongArray(CELLS);
    private final Products squares = new Products();

    void add(int cell, long q) {
      sums.addAndGet(cell, q);
      squares.add(cell, q, q);
    }

    /** Σq²（2^−64 を単位にした整数）。 */
    BigInteger squareSum(int cell) {
      return squares.sum(cell);
    }

    /** 丸めた値の平均。 */
    double mean(int cell, long rows) {
      return sums.get(cell) / SCALE / rows;
    }

    /** 丸めた値の 2 乗の平均。 */
    double meanSquare(int cell, long rows) {
      return squareSum(cell).doubleValue() / SCALE / SCALE / rows;
    }

    /** 丸めた値の、行の間の分散。(n·Σq² − (Σq)²) / n² を整数のまま求めてから double にするので、値が全て同じなら 0。 */
    double variance(int cell, long rows) {
      BigInteger sum = BigInteger.valueOf(sums.get(cell));
      BigInteger centred = squareSum(cell).multiply(BigInteger.valueOf(rows)).subtract(sum.multiply(sum));
      return centred.doubleValue() / SCALE / SCALE / rows / rows;
    }
  }

  /** 区分ごとの、整数の積の和。積は 64 ビットに収まらないことがあるので、2^32 で割った商（切り捨て）と余りに分けて足す。 */
  private static final class Products {

    private static final long LOW_BITS = 0xFFFF_FFFFL;

    private final AtomicLongArray high = new AtomicLongArray(CELLS);
    private final AtomicLongArray low = new AtomicLongArray(CELLS);

    void add(int cell, long a, long b) {
      long product = a * b;
      high.addAndGet(cell, (Math.multiplyHigh(a, b) << 32) | (product >>> 32));
      low.addAndGet(cell, product & LOW_BITS);
    }

    BigInteger sum(int cell) {
      return BigInteger.valueOf(high.get(cell)).shiftLeft(32).add(BigInteger.valueOf(low.get(cell)));
    }
  }

  /** 区分と束ごとの、行の数 m_g と丸めた d の和 D_g。 */
  private static final class ClusterSums {

    private final AtomicLongArray rows = new AtomicLongArray(CELLS * CLUSTERS);
    private final AtomicLongArray sums = new AtomicLongArray(CELLS * CLUSTERS);

    /** 束に行を 1 つ足す。 */
    void add(int cell, int cluster, long difference) {
      int index = cell * CLUSTERS + cluster;
      rows.incrementAndGet(index);
      sums.addAndGet(index, difference);
    }

    /**
     * 区分の束をまとめる。束ねた分散は、Σ_g (N·D_g − m_g·ΣD_g)² を整数のまま求めてから double にして N⁴ で割るので、d が全て同じなら 0。
     */
    Clustered cell(int cell) {
      long[] counts = new long[CLUSTERS], differences = new long[CLUSTERS];
      long total = 0, sum = 0, clusters = 0;
      for (int cluster = 0; cluster < CLUSTERS; cluster++) {
        counts[cluster] = rows.get(cell * CLUSTERS + cluster);
        if (counts[cluster] == 0) continue;
        differences[cluster] = sums.get(cell * CLUSTERS + cluster);
        total += counts[cluster];
        sum += differences[cluster];
        clusters++;
      }
      if (total == 0) return new Clustered(0, 0, 0);
      BigInteger n = BigInteger.valueOf(total), all = BigInteger.valueOf(sum), centred = BigInteger.ZERO;
      for (int cluster = 0; cluster < CLUSTERS; cluster++) {
        if (counts[cluster] == 0) continue;
        BigInteger deviation =
            BigInteger.valueOf(differences[cluster])
                .multiply(n)
                .subtract(BigInteger.valueOf(counts[cluster]).multiply(all));
        centred = centred.add(deviation.multiply(deviation));
      }
      return new Clustered(total, clusters, centred.doubleValue() / SCALE / SCALE / total / total / total / total);
    }
  }

  /**
   * 1 つの区分の束のまとめ。
   *
   * @param rows 行の数 N
   * @param clusters 行のある束の数 G
   * @param meanVariance d̄ の、局で束ねた分散に G/(G−1) を掛ける前の値 Σ_g (D_g − m_g·d̄)² / N²
   */
  private record Clustered(long rows, long clusters, double meanVariance) {}

  /**
   * 監査の集計の写し。
   *
   * @param handEnds 局末の数
   * @param riichiWins 立直の和了で終わった局末の数
   * @param averaged 期待値を求めた局末の数
   * @param groups [基本点の組の数] → 局末の数。最後のビンはそれ以上をまとめる
   * @param nanos 期待値を求めた局末で、局末の値を求めるのにかかった時間の和（ナノ秒）
   * @param cells 行が 1 つ以上あった区分
   */
  public record Report(long handEnds, long riichiWins, long averaged, long[] groups, long nanos, List<Cell> cells) {

    /** 期待値を求めた局末 1 つあたりの時間（µs）。 */
    public double microsPerAveraged() {
      return averaged == 0 ? 0 : nanos / 1e3 / averaged;
    }

    /** 収集の段の報告に書く要約を返す。 */
    public Summary summary() {
      List<Long> bins = new ArrayList<>(groups.length);
      for (long bin : groups) bins.add(bin);
      List<CellSummary> summaries = new ArrayList<>(cells.size());
      for (Cell cell : cells)
        summaries.add(
            new CellSummary(
                cell.decisionClass(),
                cell.kan(),
                cell.rows(),
                LedgerReport.finite(cell.z()),
                LedgerReport.finite(cell.reduction())));
      return new Summary(handEnds, riichiWins, averaged, bins, microsPerAveraged(), summaries);
    }
  }

  /**
   * 収集の段の報告に書く監査の要約。有限でない値は書かない。
   *
   * @param handEnds 局末の数
   * @param riichiWins 立直の和了で終わった局末の数
   * @param averaged 期待値を求めた局末の数
   * @param groups [基本点の組の数] → 局末の数。最後のビンはそれ以上をまとめる
   * @param microsPerAveraged 期待値を求めた局末 1 つあたりの時間（µs）
   * @param cells 行が 1 つ以上あった区分
   */
  public record Summary(
      long handEnds,
      long riichiWins,
      long averaged,
      List<Long> groups,
      double microsPerAveraged,
      List<CellSummary> cells) {

    public Summary {
      groups = List.copyOf(groups);
      cells = List.copyOf(cells);
    }
  }

  /**
   * 1 つの区分の要約。d は実際の値 − 期待値。
   *
   * @param rows 行の数
   * @param z d の平均を、局で束ねた標準誤差で割った値。束が {@link #MIN_CLUSTERS} 個に満たない区分には無い
   * @param reduction 期待値にして消える分が、実際の値の分散に占める割合
   */
  public record CellSummary(
      DecisionClass decisionClass,
      boolean kan,
      long rows,
      @Key(doc = "実際の値と期待値の差の平均を、局で束ねた標準誤差で割った値", presence = Presence.ABSENT) Double z,
      @Key(doc = "期待値にして消える分が、実際の値の分散に占める割合", presence = Presence.ABSENT) Double reduction) {}

  /**
   * 判断のクラスと槓の有無で分けた 1 つの区分の監査。d は実際の値 − 期待値。
   *
   * @param rows 行の数
   * @param clusters 区分に行のある束の数
   * @param meanDifference d の平均
   * @param standardError d の平均の、局で束ねた標準誤差
   * @param z d の平均を標準誤差で割った値。束が {@link #MIN_CLUSTERS} 個に満たなければ NaN
   * @param realizedVariance d² の平均（裏ドラの運が実際に動かした分）
   * @param expectedVariance 裏ドラで決まる分散の平均（期待値にして消える分）
   * @param labelVariance 実際の値の、行の間での分散
   * @param reduction 期待値にして消える分が、実際の値の分散に占める割合
   */
  public record Cell(
      DecisionClass decisionClass,
      boolean kan,
      long rows,
      long clusters,
      double meanDifference,
      double standardError,
      double z,
      double realizedVariance,
      double expectedVariance,
      double labelVariance,
      double reduction) {

    /**
     * 区分の平均と分散から、標準誤差・z・分散が減る割合を求める。標準誤差は束ねた分散に G/(G−1) を掛けた値の平方根で、束が 1 つなら NaN。
     * z は、束が {@link #MIN_CLUSTERS} 個に満たなければ NaN。
     *
     * @param meanVariance d の平均の、局で束ねた分散 Σ_g (D_g − m_g·d̄)² / N²（G/(G−1) を掛ける前）
     */
    static Cell of(
        DecisionClass decisionClass,
        boolean kan,
        long rows,
        long clusters,
        double meanDifference,
        double meanVariance,
        double realizedVariance,
        double expectedVariance,
        double labelVariance) {
      double error = clusters > 1 ? Math.sqrt(meanVariance * clusters / (clusters - 1)) : Double.NaN;
      return new Cell(
          decisionClass,
          kan,
          rows,
          clusters,
          meanDifference,
          error,
          clusters >= MIN_CLUSTERS && error > 0 ? meanDifference / error : Double.NaN,
          realizedVariance,
          expectedVariance,
          labelVariance,
          labelVariance > 0 ? expectedVariance / labelVariance : Double.NaN);
    }
  }
}
