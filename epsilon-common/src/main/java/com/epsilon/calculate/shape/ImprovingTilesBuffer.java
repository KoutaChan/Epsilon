package com.epsilon.calculate.shape;

/** 受け入れ牌種マスク・牌種数・見えていない物理牌枚数を受け取る再利用バッファ。 */
public final class ImprovingTilesBuffer {

  int remainingTileCount;
  int tileTypeCount;
  long tileTypeMask;

  public int remainingTileCount() {
    return remainingTileCount;
  }

  public int tileTypeCount() {
    return tileTypeCount;
  }

  public long tileTypeMask() {
    return tileTypeMask;
  }
}
