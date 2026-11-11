package com.epsilon.grp.train;

/**
 * 段差の検定。系列の区間ごとに（実測 − 予測）の平均を求め、区間の中心を x とする 3 次式を、重み 1/v（v は区間の分散）の最小二乗で
 * 当てはめる。返すのは、その曲線からのずれの χ²/df。値は記録に残すだけ。
 */
final class StepTest {

  /** 3 次式の係数の数。 */
  private static final int CUBIC_TERMS = 4;

  private StepTest() {}

  /**
   * 系列 1 つの χ²/df を求める。区間 b の x は、その中心 (b ＋ 0.5)·幅。分散が正の区間が 5 つ未満なら NaN を返す。
   *
   * @param bins 系列・区間ごとに集めた値
   * @param line 系列の番号
   * @param binWidth 区間の幅（x の単位で）
   */
  static double chiSquarePerDf(AdjacentBins bins, int line, double binWidth) {
    double[][] normal = new double[CUBIC_TERMS][CUBIC_TERMS + 1];
    double[] x = new double[bins.bins()];
    double[] y = new double[bins.bins()];
    double[] weight = new double[bins.bins()];
    int used = 0;
    for (int bin = 0; bin < bins.bins(); bin++) {
      double variance = bins.variance(line, bin);
      if (!(variance > 0)) {
        continue;
      }
      x[used] = (bin + 0.5) * binWidth;
      y[used] = bins.mean(line, bin);
      weight[used] = 1 / variance;
      used++;
    }
    if (used <= CUBIC_TERMS) {
      return Double.NaN;
    }
    for (int point = 0; point < used; point++) {
      for (int a = 0; a < CUBIC_TERMS; a++) {
        double termA = Math.pow(x[point], a);
        for (int b = 0; b < CUBIC_TERMS; b++) {
          normal[a][b] += weight[point] * termA * Math.pow(x[point], b);
        }
        normal[a][CUBIC_TERMS] += weight[point] * termA * y[point];
      }
    }
    double[] coefficients = solve(normal);
    double chiSquare = 0;
    for (int point = 0; point < used; point++) {
      double fit = 0;
      for (int a = 0; a < CUBIC_TERMS; a++) {
        fit += coefficients[a] * Math.pow(x[point], a);
      }
      chiSquare += weight[point] * (y[point] - fit) * (y[point] - fit);
    }
    return chiSquare / (used - CUBIC_TERMS);
  }

  /** 拡大係数行列 [n][n ＋ 1] の連立一次方程式を、部分ピボット選択つきの消去法で解く。 */
  private static double[] solve(double[][] augmented) {
    int n = augmented.length;
    for (int column = 0; column < n; column++) {
      int pivot = column;
      for (int row = column + 1; row < n; row++) {
        if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) {
          pivot = row;
        }
      }
      double[] held = augmented[column];
      augmented[column] = augmented[pivot];
      augmented[pivot] = held;
      for (int row = column + 1; row < n; row++) {
        double factor = augmented[row][column] / augmented[column][column];
        for (int k = column; k <= n; k++) {
          augmented[row][k] -= factor * augmented[column][k];
        }
      }
    }
    double[] solution = new double[n];
    for (int row = n - 1; row >= 0; row--) {
      double value = augmented[row][n];
      for (int k = row + 1; k < n; k++) {
        value -= augmented[row][k] * solution[k];
      }
      solution[row] = value / augmented[row][row];
    }
    return solution;
  }
}
