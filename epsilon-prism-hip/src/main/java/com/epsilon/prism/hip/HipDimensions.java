package com.epsilon.prism.hip;

/**
 * 重みの置き場の大きさを決める数（形と入力の約束から）。native も同じ式で置き場を照らす。
 *
 * @param hidden H
 * @param ffn F
 * @param gate G
 * @param codes 鍵の符号の数
 * @param relationRows 関係の偏りの表の行
 * @param tiles 牌の種類の数
 * @param gateKinds 門の種類の数
 * @param discardExact B の道の幅
 * @param actionExact 門の道の幅
 */
public record HipDimensions(
    int hidden,
    int ffn,
    int gate,
    int codes,
    int relationRows,
    int tiles,
    int gateKinds,
    int discardExact,
    int actionExact) {

  /**
   * 数を確かめる。
   *
   * @throws IllegalArgumentException 正でない数のとき
   */
  public HipDimensions {
    if (hidden <= 0
        || ffn <= 0
        || gate <= 0
        || codes <= 0
        || relationRows <= 0
        || tiles <= 0
        || gateKinds <= 0
        || discardExact <= 0
        || actionExact <= 0) {
      throw new IllegalArgumentException("dimensions must be positive: " + this);
    }
  }

  /** 注意の幅 A = H/2。 */
  public int attention() {
    return hidden / 2;
  }

  /** 頭の数 H/32。 */
  public int heads() {
    return attention() / HipShape.HEAD_WIDTH;
  }
}
