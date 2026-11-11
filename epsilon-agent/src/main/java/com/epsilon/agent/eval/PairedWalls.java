package com.epsilon.agent.eval;

/**
 * 同じ牌山の集合で打った 2 つの卓について、牌山ごとに値の差をとる。添字が同じなら同じ牌山で、短い方の長さまでを使う。牌山の運は両方の卓に
 * 同じ向きに効くので、その分だけ、2 つの卓の標準誤差から別々に出すより標準誤差が小さくなる。
 */
public final class PairedWalls {

  private PairedWalls() {}

  /**
   * @param a 1 つ目の卓の牌山ごとの値（牌山の番号の順）
   * @param b 2 つ目の卓の牌山ごとの値（同じ牌山の集合、番号の順）
   * @return 牌山ごとの a − b を、番号の順に足し込んだもの
   * @throws IllegalArgumentException 共通の牌山が無いとき
   */
  public static WallMean difference(double[] a, double[] b) {
    int walls = Math.min(a.length, b.length);
    if (walls == 0) {
      throw new IllegalArgumentException("No common walls");
    }
    WallMean difference = new WallMean();
    for (int i = 0; i < walls; i++) {
      difference.add(a[i] - b[i]);
    }
    return difference;
  }
}
