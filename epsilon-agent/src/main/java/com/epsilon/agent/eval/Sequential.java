package com.epsilon.agent.eval;

/**
 * 群逐次検定の境界。両側の有意水準 α を判定の時点に割り振り、進み具合 t（計画の牌山の数に対する割合）の時点までに合わせて α·t^ρ を使う
 * （Lan–DeMets の冪型の α 消費関数）。判定の時点は等間隔で、どの時点で止めても、全ての時点の区間が同時に確率 1 − α 以上で真の差を含む。
 *
 * <p>臨界値は、まだどの区間も 0 を外れていない経路の密度を、時点ごとに数値積分で進めて求める（Simpson の公式、1,001 点）。積分による臨界値の
 * 誤差は 1e-9 より小さい。
 *
 * @param alpha 両側の有意水準 α（0 より大きく 1 より小さい）
 * @param spending α 消費関数の冪 ρ（正）
 * @param checkEvery 判定の時点の間隔（牌山の数、1 以上）。最後の時点は計画の牌山の数
 */
public record Sequential(double alpha, double spending, int checkEvery) {

  public Sequential {
    if (!(alpha > 0 && alpha < 1)) {
      throw new IllegalArgumentException("Alpha must be in (0, 1): " + alpha);
    }
    if (!(spending > 0 && Double.isFinite(spending))) {
      throw new IllegalArgumentException("Spending power must be positive: " + spending);
    }
    if (checkEvery < 1) {
      throw new IllegalArgumentException("Check interval must be positive: " + checkEvery);
    }
  }

  /**
   * @param walls 計画の牌山の数（間隔の正の倍数）
   * @return 判定の時点の数
   */
  public int looks(int walls) {
    if (walls < checkEvery || walls % checkEvery != 0) {
      throw new IllegalArgumentException(walls + " walls are not a positive multiple of the check interval " + checkEvery);
    }
    return walls / checkEvery;
  }

  /**
   * @param walls 計画の牌山の数（間隔の正の倍数）
   * @return 時点ごとの臨界値。時点 k の区間は、平均 ± 臨界値 × 標準誤差
   */
  public double[] boundaries(int walls) {
    int looks = looks(walls);
    double[] critical = new double[looks];
    Continuation continuation = new Continuation();
    double spentBefore = 0;
    for (int k = 0; k < looks; k++) {
      double time = (double) (k + 1) / looks;
      double spent = k == looks - 1 ? alpha : alpha * Math.pow(time, spending);
      critical[k] = continuation.solve(time, spent - spentBefore);
      continuation.advance(time, critical[k]);
      spentBefore = spent;
    }
    return critical;
  }

  /**
   * 真の差が 0 のとき、区間がその時点で初めて 0 を外れる確率。時点は等間隔とする。テストが {@link #boundaries} の α の配り方を
   * 確かめるのに使う。
   *
   * @param critical 時点ごとの臨界値（正）
   * @return 時点ごとの確率。和は、どこかの時点で 0 を外れる確率
   */
  static double[] crossings(double[] critical) {
    double[] crossing = new double[critical.length];
    Continuation continuation = new Continuation();
    for (int k = 0; k < critical.length; k++) {
      if (!(critical[k] > 0 && Double.isFinite(critical[k]))) {
        throw new IllegalArgumentException("Invalid critical value at look " + k + ": " + critical[k]);
      }
      double time = (double) (k + 1) / critical.length;
      crossing[k] = continuation.crossing(time, critical[k]);
      continuation.advance(time, critical[k]);
    }
    return crossing;
  }

  /**
   * 牌山ごとの値の和を標準化した過程（真の差が 0 なら標準ブラウン運動 B(t)）のうち、これまでのどの時点でも境界 ±c·√t を越えていない部分の
   * 密度。等間隔の格子の上に、Simpson の重みを掛けた値で持つ。密度は原点について対称なので、格子と密度を厳密に対称に作る。
   */
  private static final class Continuation {

    private static final int GRID = 1001;
    private static final double SEARCH_LIMIT = 40;
    private static final double SEARCH_TOLERANCE = 1e-12;

    private final double[] point = new double[GRID];
    private final double[] weighted = new double[GRID];
    private final double[] next = new double[GRID];
    private double time;

    /** @return 時点 t で境界 ±c·√t を初めて越える確率 */
    double crossing(double t, double critical) {
      if (time == 0) {
        return 2 * Normal.upper(critical);
      }
      double bound = critical * Math.sqrt(t);
      double scale = Math.sqrt(t - time);
      double upper = 0;
      for (int i = 0; i < GRID; i++) {
        upper += weighted[i] * Normal.upper((bound - point[i]) / scale);
      }
      return 2 * upper;
    }

    /** @return 時点 t で初めて境界を越える確率が目標の値になる臨界値 */
    double solve(double t, double target) {
      double low = 0;
      double high = SEARCH_LIMIT;
      while (high - low > SEARCH_TOLERANCE) {
        double middle = 0.5 * (low + high);
        if (crossing(t, middle) > target) {
          low = middle;
        } else {
          high = middle;
        }
      }
      return 0.5 * (low + high);
    }

    /** 密度を時点 t まで進め、その時点の境界の内側だけを残す。 */
    void advance(double t, double critical) {
      double bound = critical * Math.sqrt(t);
      double step = 2 * bound / (GRID - 1);
      int half = GRID / 2;
      if (time == 0) {
        double scale = Math.sqrt(t);
        for (int j = 0; j <= half; j++) {
          next[j] = Normal.density((-bound + j * step) / scale) / scale;
        }
      } else {
        double scale = Math.sqrt(t - time);
        for (int j = 0; j <= half; j++) {
          double y = -bound + j * step;
          double sum = 0;
          for (int i = 0; i < GRID; i++) {
            sum += weighted[i] * Normal.density((y - point[i]) / scale);
          }
          next[j] = sum / scale;
        }
      }
      for (int j = 0; j <= half; j++) {
        double y = j == half ? 0 : -bound + j * step;
        double weight = step / 3 * (j == 0 ? 1 : j % 2 == 1 ? 4 : 2);
        point[j] = y;
        point[GRID - 1 - j] = -y;
        weighted[j] = weight * next[j];
        weighted[GRID - 1 - j] = weighted[j];
      }
      time = t;
    }
  }
}
