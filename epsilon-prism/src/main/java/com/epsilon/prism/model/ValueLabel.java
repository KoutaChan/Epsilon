package com.epsilon.prism.model;

import java.util.List;

/**
 * 差分整合の 1 行の教師。KL の基準は C の上の一様分布に固定するので、p0 は持たない。
 *
 * @param values 行の手ごとの Y（入力の手の順。測っていない手は NaN、測った手が C）
 * @param lambda 行の類の温度 λ_c
 * @param weight 行の損失の重み（Horvitz–Thompson の重み。普通は 1）
 */
public record ValueLabel(float[] values, float lambda, float weight) {

  /**
   * 値を確かめる。
   *
   * @throws IllegalArgumentException λ が正でないか、重みが負か、Y が無限のとき
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
   * 行の並びの C の印を、塊の手の順につないで返す（{@link Pass#training} に渡す）。
   *
   * @param labels 塊の行の順の教師
   * @return 塊の手ごとの C の印
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
