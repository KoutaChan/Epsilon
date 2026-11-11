package com.epsilon.grp.train;

/**
 * セルごとに値を足し、平均とその分散を、半荘をクラスタとして求める。半荘の値を {@link #add} で足し、半荘が終わるたびに
 * {@link #endGame} を呼ぶ。セル c の平均は m_c = Σx / n_c、その分散は v_c = G_c/(G_c − 1) · Σ_g (S_gc − m_c n_gc)² / n_c²。
 * S_gc と n_gc は半荘 g でセル c に足した値の和と数、G_c はセル c に値が入った半荘の数。G_c が 2 未満のセルでは、分散は NaN。
 */
final class Clusters {

  private final double[] count;
  private final double[] sum;
  private final double[] sumSquares;
  private final double[] sumCross;
  private final double[] countSquares;
  private final double[] clusters;
  private final double[] gameSum;
  private final double[] gameCount;
  private final int[] touched;
  private int touchedCount;

  Clusters(int cells) {
    count = new double[cells];
    sum = new double[cells];
    sumSquares = new double[cells];
    sumCross = new double[cells];
    countSquares = new double[cells];
    clusters = new double[cells];
    gameSum = new double[cells];
    gameCount = new double[cells];
    touched = new int[cells];
  }

  void add(int cell, double value) {
    if (gameCount[cell] == 0) {
      touched[touchedCount++] = cell;
    }
    gameSum[cell] += value;
    gameCount[cell]++;
  }

  /** 今の半荘で足した値を、全体の和に移す。 */
  void endGame() {
    for (int index = 0; index < touchedCount; index++) {
      int cell = touched[index];
      double s = gameSum[cell];
      double n = gameCount[cell];
      count[cell] += n;
      sum[cell] += s;
      sumSquares[cell] += s * s;
      sumCross[cell] += s * n;
      countSquares[cell] += n * n;
      clusters[cell]++;
      gameSum[cell] = 0;
      gameCount[cell] = 0;
    }
    touchedCount = 0;
  }

  double count(int cell) {
    return count[cell];
  }

  /** セルに値が入った半荘の数。 */
  double clusters(int cell) {
    return clusters[cell];
  }

  /** セルの平均。空のセルは NaN。 */
  double mean(int cell) {
    return sum[cell] / count[cell];
  }

  /** 平均の分散 v_c。値が入った半荘が 2 つ未満なら NaN。 */
  double variance(int cell) {
    double games = clusters[cell];
    if (games < 2) {
      return Double.NaN;
    }
    double mean = mean(cell);
    double spread = sumSquares[cell] - 2 * mean * sumCross[cell] + mean * mean * countSquares[cell];
    return games / (games - 1) * Math.max(0, spread) / (count[cell] * count[cell]);
  }

  /** Σ_c (n_c/N)(m_c² − v_c) を返す。N は渡した行数。値が入った半荘が 2 つ未満のセルは除く。 */
  double distance(double rows) {
    double distance = 0;
    for (int cell = 0; cell < count.length; cell++) {
      double variance = variance(cell);
      if (variance >= 0) {
        double mean = mean(cell);
        distance += count[cell] / rows * (mean * mean - variance);
      }
    }
    return distance;
  }

  Paired paired(int cell) {
    return new Paired(mean(cell), Math.sqrt(variance(cell)));
  }
}
