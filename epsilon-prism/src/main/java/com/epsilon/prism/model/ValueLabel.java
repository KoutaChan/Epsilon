package com.epsilon.prism.model;

import java.util.List;

/**
 * 差分整合の損失に使う、1 行分の教師。KL の基準は C の上の一様分布に固定するので、p0 は持たない。
 *
 * @param values 手ごとの Y（入力の手の順）。測っていない手は NaN にする。測った手の集まりが C
 * @param lambda この行の判断のクラスの温度 λ_c
 * @param weight この行の損失の重み（Horvitz–Thompson の重み。ふつうは 1）
 */
public record ValueLabel(float[] values, float lambda, float weight) {

  /**
   * 値を確かめる。
   *
   * @throws IllegalArgumentException λ が正の有限値でないとき、重みが負か有限でないとき、Y に無限大があるとき
   */
  public ValueLabel {
    if (!(lambda > 0) || !Float.isFinite(lambda)) {
      throw new IllegalArgumentException("lambda must be positive: " + lambda);
    }
    if (!(weight >= 0) || !Float.isFinite(weight)) {
      throw new IllegalArgumentException("weight must be non-negative: " + weight);
    }
    for (float value : values) {
      if (Float.isInfinite(value)) {
        throw new IllegalArgumentException("values must be finite or NaN");
      }
    }
  }

  /**
   * 測った手の数を返す。
   *
   * @return |C|
   */
  public int measuredCount() {
    int count = 0;
    for (float value : values) {
      if (!Float.isNaN(value)) {
        count++;
      }
    }
    return count;
  }

  /**
   * 各行の手が C に入るかを、バッチの手の順につないで返す。{@link Pass#training} に渡す。
   *
   * @param labels バッチの行の順に並べた教師
   * @return バッチの手ごとに、C に入るなら true
   */
  public static boolean[] measured(List<ValueLabel> labels) {
    int total = 0;
    for (ValueLabel label : labels) {
      total += label.values.length;
    }
    boolean[] flags = new boolean[total];
    int at = 0;
    for (ValueLabel label : labels) {
      for (float value : label.values) {
        flags[at++] = !Float.isNaN(value);
      }
    }
    return flags;
  }
}
