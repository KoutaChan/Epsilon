package com.epsilon.prism.grp;

import java.util.Optional;

/**
 * GRP の学習の窓: 取り置きのコホート h（学習に使わない最新の封印済みのコホート。関門はこれで測る）の直前の w 個のコホート。
 * w = clamp(⌈h/4⌉, min({@value #MIN_COHORTS}, grpWindow), grpWindow)、コホート 1〜{@value #SKIPPED_COHORTS} は使わない。
 *
 * @param first 窓の最初のコホート
 * @param last 窓の最後のコホート（h − 1）
 * @param holdout 取り置きのコホート h
 */
public record RefitWindow(int first, int last, int holdout) {

  /** 窓のコホートの数の下限（grpWindow がこれより小さければ grpWindow）。 */
  public static final int MIN_COHORTS = 4;

  /** 学習に使わない始めのコホートの数（白紙に近い網の打ち方）。 */
  public static final int SKIPPED_COHORTS = 2;

  /**
   * 取り置き h の窓。
   *
   * @param holdout 取り置きのコホート h
   * @param grpWindow 窓のコホートの数の上限（1 以上）
   * @return 窓（使えるコホートが 1 つもなければ空）
   */
  public static Optional<RefitWindow> ending(int holdout, int grpWindow) {
    if (grpWindow < 1) throw new IllegalArgumentException("grpWindow must be positive: " + grpWindow);
    int width = Math.clamp((holdout + 3) / 4, Math.min(MIN_COHORTS, grpWindow), grpWindow);
    int first = Math.max(SKIPPED_COHORTS + 1, holdout - width);
    int last = holdout - 1;
    return last < first ? Optional.empty() : Optional.of(new RefitWindow(first, last, holdout));
  }

  /** 窓のコホートの数。 */
  public int cohorts() {
    return last - first + 1;
  }
}
