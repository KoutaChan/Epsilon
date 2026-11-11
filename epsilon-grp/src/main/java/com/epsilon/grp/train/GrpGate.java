package com.epsilon.grp.train;

import java.util.EnumSet;
import java.util.List;

/**
 * 候補のモデルを関門で判定する。関門で測るのは、再学習に使っていない半荘。{@link GateCohort#NEXT} なら関門のコホートの全ての
 * 半荘を、{@link GateCohort#GUARDED} ならその関門用の半荘を使う。通る条件は次の 3 つ。
 *
 * <ul>
 *   <li>値が全て有限である。
 *   <li>単調性の違反率が max(1.5 × 使用中のモデルの率, 0.005) 以下である。
 *   <li>対応のある MSE の差が {@link GateKind} の基準を満たす。
 * </ul>
 *
 * 監視でモデルを戻してから 8 反復の間は、戻したモデルとの差が +1 SE 以内であることも求める。違反率は MonotoneProbe で測る。使用中の
 * モデルの違反率を測れないとき（点数の Φ）は、その率に 0 を渡す。しきい値は下限の 0.005 になる。
 */
public final class GrpGate {

  static final double MONOTONE_FACTOR = 1.5;
  static final double MONOTONE_FLOOR = 0.005;

  /** 戻したモデルより悪くてよい幅（SE の倍数）。戻した効果をすぐに消さないため。 */
  static final double ANCHOR_STANDARD_ERRORS = 1;

  private GrpGate() {}

  /** 毎回の再学習の関門。 */
  public static GateReport refit(
      HandEnds gate, HandEndValues used, HandEndValues candidate, double usedMonotone, double candidateMonotone) {
    return judge(
        GateKind.REFIT,
        candidate.finite() && Double.isFinite(candidateMonotone),
        GrpMetrics.pairedMse(gate, used, candidate),
        null,
        usedMonotone,
        candidateMonotone);
  }

  /**
   * 切り替えの後の監視でモデルを戻してから 8 反復の間に使う、再学習の関門。
   *
   * @param anchor 戻したモデルの値
   */
  public static GateReport refit(
      HandEnds gate,
      HandEndValues used,
      HandEndValues candidate,
      HandEndValues anchor,
      double usedMonotone,
      double candidateMonotone) {
    return judge(
        GateKind.REFIT,
        candidate.finite() && Double.isFinite(candidateMonotone),
        GrpMetrics.pairedMse(gate, used, candidate),
        GrpMetrics.pairedMse(gate, anchor, candidate),
        usedMonotone,
        candidateMonotone);
  }

  /** 最初の切り替えと、モデルの構造を入れ替えるときの関門。 */
  public static GateReport replacement(
      HandEnds gate, HandEndValues used, HandEndValues candidate, double usedMonotone, double candidateMonotone) {
    return judge(
        GateKind.REPLACEMENT,
        candidate.finite() && Double.isFinite(candidateMonotone),
        GrpMetrics.pairedMse(gate, used, candidate),
        null,
        usedMonotone,
        candidateMonotone);
  }

  /**
   * 測った値から判定する。
   *
   * @param versusAnchor 戻したモデルとの差。モデルを戻してから 8 反復の間でなければ null
   */
  static GateReport judge(
      GateKind kind,
      boolean finite,
      Paired versusUsed,
      Paired versusAnchor,
      double usedMonotone,
      double candidateMonotone) {
    double line = Math.max(MONOTONE_FACTOR * usedMonotone, MONOTONE_FLOOR);
    EnumSet<GateFailure> failures = EnumSet.noneOf(GateFailure.class);
    if (!finite) {
      failures.add(GateFailure.NOT_FINITE);
    }
    if (!kind.accepts(versusUsed)) {
      failures.add(kind.failure());
    }
    if (versusAnchor != null
        && !(versusAnchor.mean() <= ANCHOR_STANDARD_ERRORS * versusAnchor.standardError())) {
      failures.add(GateFailure.WORSE_THAN_ANCHOR);
    }
    if (!(candidateMonotone <= line)) {
      failures.add(GateFailure.NOT_MONOTONE);
    }
    return new GateReport(kind, List.copyOf(failures), versusUsed, versusAnchor, candidateMonotone, line);
  }
}
