package com.epsilon.mahjong.rules;

/**
 * 最終順位の効用 u_r（r は 0 始まりの順位）。定数の名前は PRISM の入力のフィンガープリントと系譜の config.json に入るので、変えない。
 * 値を変えたいときは新しい定数を足す。値は席の数ごとに 1 列あり、列の長さは席の数と同じ。
 */
public enum RankUtility {
  /** 四人打ちは [2/3, 1/3, 0, −1]。三人打ちは [2, 0, −2]（三麻の七段の段位 pt の比 +135 : 0 : −135）。 */
  TENHOU(new double[] {2.0 / 3, 1.0 / 3, 0, -1}, new double[] {2, 0, -2}),
  /** u_r = 1 − 2r/(S − 1)。S は席の数。 */
  AVERAGE_RANK(new double[] {1, 1.0 / 3, -1.0 / 3, -1}, new double[] {1, 0, -1});

  /** [席の数] → 順位ごとの値。 */
  private final double[][] bySeats;

  RankUtility(double[]... utilities) {
    int maxSeats = 0;
    for (double[] values : utilities) {
      maxSeats = Math.max(maxSeats, values.length);
    }
    bySeats = new double[maxSeats + 1][];
    for (double[] values : utilities) {
      bySeats[values.length] = values;
    }
  }

  public double value(Variant variant, int rank) {
    return bySeats[variant.seats()][rank];
  }
}
