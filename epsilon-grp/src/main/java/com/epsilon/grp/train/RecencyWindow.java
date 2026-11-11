package com.epsilon.grp.train;

/**
 * 学習し直すときに使うコホートの範囲と、その重み。コホート c の重みは w_c = 2^{−(n − c)/τ} で、n は最新のコホート、τ は半減期
 * （+∞ なら一様）。最新のコホートの重みは 1 になる。w_c が 1/16 未満のコホートと、始めの 2 つのコホート（ほぼ初期化したままの
 * ネットワークで集めたコホート 1 と 2）は使わない。使うのは、最新から数えてコホートの数の上限まで。
 *
 * @param newest 最新のコホート
 * @param halfLife 重みの半減期 τ（コホートの数で数える）
 * @param maxCohorts 使うコホートの数の上限
 */
public record RecencyWindow(int newest, double halfLife, int maxCohorts) {

  /** 学習に使わない、始めのコホートの数。使うのは、番号がこの数 ＋ 1 以上のコホート。 */
  public static final int SKIPPED_COHORTS = 2;

  /** これより軽いコホートは使わない。 */
  public static final double MIN_WEIGHT = 1.0 / 16;

  public RecencyWindow {
    if (!(halfLife > 0)) {
      throw new IllegalArgumentException("The half-life must be positive: " + halfLife);
    }
    if (maxCohorts < 1) {
      throw new IllegalArgumentException("A window needs at least one cohort: " + maxCohorts);
    }
  }

  /** τ = +∞ で重みが一様な範囲。 */
  public static RecencyWindow uniform(int newest, int maxCohorts) {
    return new RecencyWindow(newest, Double.POSITIVE_INFINITY, maxCohorts);
  }

  /** 範囲の最初のコホートを返す。使えるコホートが無ければ、最新のコホート ＋ 1 を返す。 */
  public int first() {
    int first = newest + 1;
    while (first - 1 > SKIPPED_COHORTS
        && newest - first + 2 <= maxCohorts
        && decay(newest - first + 1) >= MIN_WEIGHT) {
      first--;
    }
    return first;
  }

  /** 範囲のコホートの数。 */
  public int cohorts() {
    return newest - first() + 1;
  }

  public boolean contains(int cohort) {
    return cohort >= first() && cohort <= newest;
  }

  /** コホートの重みを返す。範囲の外のコホートは 0。 */
  public double weight(int cohort) {
    return contains(cohort) ? decay(newest - cohort) : 0;
  }

  private double decay(int age) {
    return Math.pow(2, -age / halfLife);
  }
}
