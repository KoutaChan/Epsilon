package com.epsilon.grp.train;

/**
 * 学習の行に掛ける、コホートごとの重みの表。GrpRows と OutcomeRows が使う。各行は自分のコホートを、範囲の最初のコホートからの差
 * （符号なしの byte）で持ち、重みはその差で引く。そのため範囲に入るコホートは 256 個まで。
 */
final class CohortWeights {

  static final int MAX_COHORTS = 256;

  private final RecencyWindow window;

  /** 範囲の最初のコホートからの差ごとの重み。 */
  private final float[] weights;

  /** 範囲に入るコホートが 1 から 256 個でなければ IllegalArgumentException を投げる。 */
  CohortWeights(RecencyWindow window) {
    if (window.cohorts() < 1 || window.cohorts() > MAX_COHORTS) {
      throw new IllegalArgumentException("A window must hold 1 to " + MAX_COHORTS + " cohorts: " + window);
    }
    this.window = window;
    weights = new float[window.cohorts()];
    for (int offset = 0; offset < weights.length; offset++) {
      weights[offset] = (float) window.weight(window.first() + offset);
    }
  }

  RecencyWindow window() {
    return window;
  }

  /** 範囲の最初のコホートから、渡したコホートまでの差。 */
  byte offset(int cohort) {
    return (byte) (cohort - window.first());
  }

  /** 差から元のコホートに戻す。 */
  int cohort(byte offset) {
    return window.first() + Byte.toUnsignedInt(offset);
  }

  /** 差で表したコホートの重み。 */
  float weight(byte offset) {
    return weights[Byte.toUnsignedInt(offset)];
  }

  /** 差の列の先頭から指定した行数ぶんについて、重みを足した値。 */
  double total(byte[] offsets, int size) {
    double total = 0;
    for (int row = 0; row < size; row++) {
      total += weight(offsets[row]);
    }
    return total;
  }
}
