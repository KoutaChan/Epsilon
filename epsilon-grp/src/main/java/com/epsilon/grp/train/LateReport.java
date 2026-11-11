package com.epsilon.grp.train;

import com.epsilon.grp.FinalRound;
import java.util.ArrayList;
import java.util.List;
import java.util.function.IntPredicate;

/**
 * 終盤の数え上げ（LATE）の値と、同じ GRP を使う GRP のモードの値を、数え上げの対象の局末で比べた報告。対象の局末は
 * {@code FinalRound.covers} が true の局末で、次の局が南場の最後の局かそれより後の局末。差は全て、同じ局末どうしの対の差
 * （LATE − GRP のモード）。標準誤差は半荘をクラスタとして求める（GrpMetrics）。
 *
 * <p>決まり具合の区間は、今トップの席が 1 着で終わる確率で分ける（0.5 未満・0.5 以上 0.9 未満・0.9 以上）。確率には GRP のモードの
 * 値を使う。区間を片方のモードの値だけで決めるので、同じ局末はどちらのモードでも同じ区間に入り、区間の中でも対で比べられる。
 *
 * @param handEnds 対象の局末の数
 * @param late 対象の局末での、LATE の値の較正
 * @param grp 対象の局末での、GRP のモードの値の較正
 * @param mse 対の MSE の差
 * @param rankNll 最終順位の周辺確率の NLL の差（席・局末あたり）
 * @param bins 決まり具合の区間（確率の小さい順）
 * @param enumeration 最後の局の数え上げの集計（対象の局末あたりの平均）
 */
public record LateReport(
    long handEnds,
    GrpReport late,
    GrpReport grp,
    Paired mse,
    Paired rankNll,
    List<Bin> bins,
    Enumeration enumeration) {

  /** 決まり具合の区間の境界。区間は下の境界を含み、上の境界を含まない（最後の区間だけは 1 も含む）。 */
  private static final double[] EDGES = {0, 0.5, 0.9, 1};

  /** 対象の局末が入る区分の数（最後の区分から数える）。 */
  private static final int LATE_GROUPS = 2;

  public LateReport {
    bins = List.copyOf(bins);
  }

  /**
   * 決まり具合の区間 1 つ。
   *
   * @param low 確率の下の境界（含む）
   * @param high 確率の上の境界（含まない。最後の区間だけは含む）
   * @param handEnds 区間に入る局末の数
   * @param late 終盤の 2 区分での、LATE の値の較正（区分の順）
   * @param grp 終盤の 2 区分での、GRP のモードの値の較正（区分の順）
   * @param mse 対の MSE の差
   */
  public record Bin(
      double low, double high, long handEnds, List<GrpReport.Fit> late, List<GrpReport.Fit> grp, Paired mse) {

    public Bin {
      late = List.copyOf(late);
      grp = List.copyOf(grp);
    }
  }

  /**
   * 最後の局の数え上げの集計で、対象の局末あたりの平均。LATE の値が数え上げた値でなければ全て 0。
   *
   * @param filled 打ち切って葉のネットワークで埋めた確率
   * @param terminal 局の後に半荘が終わる組の確率の和
   * @param settled 精算して局を進めた組の数
   * @param leaves 半荘が続く葉の数
   */
  public record Enumeration(double filled, double terminal, double settled, double leaves) {}

  /**
   * 数え上げの対象の局末で測る。
   *
   * @param ends 局末の列
   * @param late その列の LATE の値
   * @param grp その列の、同じ GRP を使う GRP のモードの値
   * @throws IllegalArgumentException 値がその局末の列について求めたものでないとき
   */
  public static LateReport measure(HandEnds ends, HandEndValues late, HandEndValues grp) {
    GrpMetrics.requireValues(ends, late);
    GrpMetrics.requireValues(ends, grp);
    HandEnds covered = ends.select(h -> FinalRound.covers(ends.state(h)));
    HandEndValues lateValues = late.select(covered);
    HandEndValues grpValues = grp.select(covered);
    byte[] topBins = topBins(covered, grpValues);
    List<Bin> bins = new ArrayList<>(EDGES.length - 1);
    for (int bin = 0; bin < EDGES.length - 1; bin++) {
      byte chosen = (byte) bin;
      IntPredicate inBin = h -> topBins[h] == chosen;
      long count = 0;
      for (byte value : topBins) {
        count += value == chosen ? 1 : 0;
      }
      bins.add(new Bin(
          EDGES[bin],
          EDGES[bin + 1],
          count,
          lateFits(covered, GrpMetrics.measure(covered, lateValues, inBin)),
          lateFits(covered, GrpMetrics.measure(covered, grpValues, inBin)),
          GrpMetrics.pairedMse(covered, grpValues, lateValues, inBin)));
    }
    return new LateReport(
        covered.handEnds(),
        GrpMetrics.measure(covered, lateValues),
        GrpMetrics.measure(covered, grpValues),
        GrpMetrics.pairedMse(covered, grpValues, lateValues),
        GrpMetrics.pairedRankNll(covered, grpValues, lateValues),
        bins,
        enumeration(lateValues));
  }

  /** 局末ごとに、決まり具合の区間の番号を求める。区間は、GRP のモードの値で今トップの席が 1 着で終わる確率から決める。 */
  private static byte[] topBins(HandEnds covered, HandEndValues grp) {
    byte[] bins = new byte[covered.handEnds()];
    for (int h = 0; h < bins.length; h++) {
      double top = grp.marginal(h, covered.seatAt(h, 0), 0);
      int bin = 0;
      while (bin < EDGES.length - 2 && top >= EDGES[bin + 1]) {
        bin++;
      }
      bins[h] = (byte) bin;
    }
    return bins;
  }

  /** 終盤の 2 区分の較正（区分の順）。 */
  private static List<GrpReport.Fit> lateFits(HandEnds ends, GrpReport report) {
    return report.groups().subList(ends.groups() - LATE_GROUPS, ends.groups());
  }

  /** 局末あたりの数え上げの集計。局末が無ければ全て NaN。 */
  private static Enumeration enumeration(HandEndValues late) {
    int count = late.ends().handEnds();
    double filled = 0;
    double terminal = 0;
    double settled = 0;
    double leaves = 0;
    for (int h = 0; h < count; h++) {
      filled += late.filledMass(h);
      terminal += late.terminalMass(h);
      settled += late.settled(h);
      leaves += late.leaves(h);
    }
    return new Enumeration(filled / count, terminal / count, settled / count, leaves / count);
  }
}
