package com.epsilon.agent.eval;

/**
 * 牌山ごとの値から平均と標準誤差を求める。値は Welford の式で 1 つずつ足すので、全ての値が同じなら標準誤差はちょうど 0 になる。
 * 足す順で丸めが変わるため、牌山の番号の順に足す。
 */
public final class WallMean {

  private int walls;
  private double mean;
  private double squares;

  /** @param value 1 つの牌山の値（有限） */
  public void add(double value) {
    if (!Double.isFinite(value)) {
      throw new IllegalArgumentException("Invalid wall value: " + value);
    }
    walls++;
    double deviation = value - mean;
    mean += deviation / walls;
    squares += deviation * (value - mean);
  }

  /** @return 足した牌山の数 */
  public int walls() {
    return walls;
  }

  /** @return 平均 */
  public double mean() {
    if (walls == 0) {
      throw new IllegalStateException("No walls");
    }
    return mean;
  }

  /** @return 平均の標準誤差。分散には不偏分散（牌山の数 − 1 で割る）を使う */
  public double error() {
    if (walls < 2) {
      throw new IllegalStateException("Standard error needs two walls, have " + walls);
    }
    return Math.sqrt(squares / (walls - 1) / walls);
  }
}
