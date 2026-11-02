package com.epsilon.prism.grp;

/**
 * 局末の列の Φ（{@link HandEndPhi#values}）。添字は入力の局末の順。
 *
 * @param utilities 4 席の Φ（期待効用。較正の表まで掛けた、ラベルに入る値）
 * @param rankProbabilities 順位分布（席 i・順位 r が [4i + r]。温度の後、較正の表の前。点数の順位は 0/1）
 */
public record PhiValues(double[][] utilities, double[][] rankProbabilities) {

  public PhiValues {
    if (utilities.length != rankProbabilities.length)
      throw new IllegalArgumentException("Φ values and rank probabilities differ in length");
  }

  /** 局末の数。 */
  public int size() {
    return utilities.length;
  }
}
