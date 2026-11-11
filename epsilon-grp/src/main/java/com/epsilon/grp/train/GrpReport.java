package com.epsilon.grp.train;

import java.util.List;

/**
 * Φ 1 つの較正を、半荘の列で測った結果（{@link GrpMetrics#measure}）。残差 r = u − Φ は、半荘が続く局末で席ごとに 1 つ取る。
 * 標準誤差は全て、半荘をクラスタとして求める。「終盤の 2 区分」は最後の 2 つの区分のことで、四人打ちなら S3・S4+、三人打ちなら
 * S2・S3+。
 *
 * @param mse 残差の二乗の平均
 * @param dSpg 偏りの二乗の不偏推定量 Σ_c (n_c/N)(m_c² − v_c)。セル c は区分 × 今の順位 × 点差（2,000・6,000・12,000 点で 4 区間に
 *     分ける）。m_c はセルの残差の平均、v_c はその分散。半荘が 2 つ未満のセルは除く
 * @param dSb 同じ量を、区分 × Φ の 10 区間のセルで求めた値。Φ の区間は、最小の効用から最大の効用までを等幅に分ける
 * @param groups 区分ごとの較正（HandEnds の区分の順）
 * @param phases 段階ごとの較正（GrpPhase の順）
 * @param rankNll 最終順位の周辺確率の負の対数尤度。席・局末あたりの値で、確率の下限は 1e-9
 * @param brier 最終順位の周辺確率の Brier スコア。席・局末あたりの値
 * @param fourthEce 最下位になる確率の ECE。区間は等幅の 15 個
 * @param jointNll 実際の順列の同時 NLL。局末あたりの値で、同時分布を出さない Φ の項目では NaN
 * @param closeCells 終盤の 2 区分の、僅差の 8 セル。点差は、今最下位の席なら 1 つ上の席との差、最下位の 1 つ上の席なら
 *     最下位との差で、2,000 点未満と 2,000〜5,999 点の 2 区間に分ける
 * @param jumps 終盤の 2 区分で、今の順位ごとに見た段差。今の順位のままで終わる率の段差と、Φ の段差がある
 */
public record GrpReport(
    long games,
    long seatHands,
    double mse,
    double dSpg,
    double dSb,
    List<Fit> groups,
    List<Fit> phases,
    double rankNll,
    double brier,
    double fourthEce,
    double jointNll,
    Tails tails,
    List<CloseCell> closeCells,
    List<JumpSeries> jumps) {

  public GrpReport {
    groups = List.copyOf(groups);
    phases = List.copyOf(phases);
    closeCells = List.copyOf(closeCells);
    jumps = List.copyOf(jumps);
  }

  /**
   * 区分 1 つ（または段階 1 つ）の較正。
   *
   * @param b4 今最下位の席の残差の平均
   * @param slope u を Φ に回帰した傾き β
   * @param slopeSe 半荘をクラスタとしたサンドイッチ型の標準誤差。半荘の数が 200 未満なら NaN
   * @param slopeZ (β − 1) / SE
   */
  public record Fit(long seatHands, double b4, double slope, double slopeSe, double slopeZ) {

    /** |β − 1| ≤ 0.01 と |b4| ≤ 0.005 の基準を満たすか。この基準は全ての区分に求める。測れない量（NaN）は問わない。 */
    public boolean within() {
      return !(Math.abs(slope - 1) > 0.01) && !(Math.abs(b4) > 0.005);
    }
  }

  /** 端の区間の、予測の平均と実測の平均。標準誤差は実測の平均の標準誤差。 */
  public record Tail(long rows, double predicted, double realized, double standardError) {

    /** 確率の端の基準を満たすか。稀な側の率で、予測が [実測 / 2 − 2 SE, 実測 × 2 ＋ 2 SE] に入れば満たす。行が無ければ問わない。 */
    public boolean within() {
      return rows == 0
          || predicted >= realized / 2 - 2 * standardError && predicted <= realized * 2 + 2 * standardError;
    }
  }

  /**
   * 確率の端と Φ の端。確率の端は 2 つとも、稀な側の率で基準と比べる。
   *
   * @param lowProbability 全ての席と順位の周辺確率のうち、予測 p が 1% 以下のもの。実測はその順位で終わった率
   * @param highProbability p が 95% 以上のもの。外れる側の率で持ち、予測は 1 − p、実測はその順位で終わらなかった率
   * @param leaderLast 終盤の 2 区分で首位が 2 着を 12,000 点以上離している局末で、首位が最下位で終わる率
   * @param lastFirst 終盤の 2 区分で最下位が 1 つ上に 12,000 点以上離されている局末で、最下位が 1 着で終わる率
   * @param lowPhi Φ を効用の幅で 0〜1 に直した値が 1% 以下の席・局末。実測は u の平均
   * @param highPhi 同じく 95% 以上の席・局末
   */
  public record Tails(
      Tail lowProbability,
      Tail highProbability,
      Tail leaderLast,
      Tail lastFirst,
      double minPhi,
      double maxPhi,
      Tail lowPhi,
      Tail highPhi) {}

  /**
   * 僅差のセル。区分、今の順位、点差の区間（下の境を含み、上の境を含まない）で決まる。
   *
   * @param keepGap 今の順位のままで終わる率の、実測 − 予測
   */
  public record CloseCell(
      int group, int rank, int lowGap, int highGap, long seatHands, double keepGap, double standardError) {

    public double z() {
      return keepGap / standardError;
    }

    /** 僅差のセルの基準を満たすか。実測 − 予測の差が 1 ポイント以内か |z| < 3 なら満たす。 */
    public boolean within() {
      return Math.abs(keepGap) <= 0.01 || Math.abs(z()) < 3;
    }
  }

  public enum JumpMeasure {
    /** 今の順位のままで終わるか。実測は 0 か 1、予測はその順位の周辺確率。 */
    KEEP,
    /** 実測は u、予測は Φ。 */
    PHI
  }

  /**
   * 区分 1 つと今の順位 1 つについて、段差をまとめたもの。点差を 500 点ごとの区間に分けたとき、隣り合う区間の間で（実測 − 予測）が
   * どれだけ変わるかを段差と呼ぶ。
   *
   * @param pairs 段差の数
   * @param chiSquarePerDf 段差の z² の平均
   * @param maxAbsZ 段差の |z| の最大
   * @param stepChiSquarePerDf 段差の検定の値。今の順位のままで終わる率について、区間ごとの（実測 − 予測）を 3 次曲線に当てはめ、
   *     曲線からのずれを χ²/df で表す。Φ を測る系列と、区間が 5 つ未満のときは NaN
   */
  public record JumpSeries(
      int group,
      int rank,
      JumpMeasure measure,
      int pairs,
      double chiSquarePerDf,
      double maxAbsZ,
      double stepChiSquarePerDf) {

    /** 段差の χ²/df ≤ 1.5、|z| > 4 の段差が無い、段差の検定の χ²/df ≤ 2 を全て満たすか。測れない量（NaN）は問わない。 */
    public boolean within() {
      return !(chiSquarePerDf > 1.5) && !(maxAbsZ > 4) && !(stepChiSquarePerDf > 2);
    }
  }
}
