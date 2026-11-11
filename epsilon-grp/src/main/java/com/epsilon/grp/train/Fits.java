package com.epsilon.grp.train;

import java.util.ArrayList;
import java.util.List;

/**
 * 区分ごと（または段階ごと）に u を Φ に回帰し、今最下位の席の残差の平均も求める（{@link GrpReport.Fit}）。値を全て {@link #add} で
 * 足してから {@link #solve} を呼ぶ。続いて同じ値をもう一度たどり、{@link #score} で足して半荘ごとに {@link #endGame} を呼ぶと、
 * サンドイッチ型の標準誤差が求まる。
 */
final class Fits {

  /**
   * 傾きの標準誤差と z を出すのに要る半荘の数（その区分に値が入った半荘）の下限。クラスタが少ないと標準誤差がほぼ 0 になり、z が
   * 発散する。
   */
  static final int MIN_SLOPE_GAMES = 200;

  private final double[] count;
  private final double[] sumX;
  private final double[] sumY;
  private final double[] sumXX;
  private final double[] sumXY;
  private final double[] lastSum;
  private final double[] lastCount;
  private final double[] slope;
  private final double[] meanX;
  private final double[] meanY;
  private final double[] spreadX;
  private final double[] gameScore;
  private final boolean[] touched;
  private final double[] squares;
  private final double[] clusters;

  Fits(int categories) {
    count = new double[categories];
    sumX = new double[categories];
    sumY = new double[categories];
    sumXX = new double[categories];
    sumXY = new double[categories];
    lastSum = new double[categories];
    lastCount = new double[categories];
    slope = new double[categories];
    meanX = new double[categories];
    meanY = new double[categories];
    spreadX = new double[categories];
    gameScore = new double[categories];
    touched = new boolean[categories];
    squares = new double[categories];
    clusters = new double[categories];
  }

  /**
   * 席・局末を 1 つ足す。
   *
   * @param category 区分（または段階）の番号
   * @param x Φ
   * @param y 効用 u
   * @param lastPlace 今最下位の席なら true
   */
  void add(int category, double x, double y, boolean lastPlace) {
    count[category]++;
    sumX[category] += x;
    sumY[category] += y;
    sumXX[category] += x * x;
    sumXY[category] += x * y;
    if (lastPlace) {
      lastSum[category] += y - x;
      lastCount[category]++;
    }
  }

  void solve() {
    for (int category = 0; category < count.length; category++) {
      meanX[category] = sumX[category] / count[category];
      meanY[category] = sumY[category] / count[category];
      spreadX[category] = sumXX[category] - count[category] * meanX[category] * meanX[category];
      slope[category] = (sumXY[category] - count[category] * meanX[category] * meanY[category]) / spreadX[category];
    }
  }

  void score(int category, double x, double y) {
    double centered = x - meanX[category];
    double error = y - meanY[category] - slope[category] * centered;
    gameScore[category] += centered * error;
    touched[category] = true;
  }

  void endGame() {
    for (int category = 0; category < count.length; category++) {
      if (touched[category]) {
        squares[category] += gameScore[category] * gameScore[category];
        clusters[category]++;
        gameScore[category] = 0;
        touched[category] = false;
      }
    }
  }

  List<GrpReport.Fit> fits() {
    List<GrpReport.Fit> fits = new ArrayList<>(count.length);
    for (int category = 0; category < count.length; category++) {
      double games = clusters[category];
      double error = games < MIN_SLOPE_GAMES
          ? Double.NaN
          : Math.sqrt(games / (games - 1) * squares[category]) / spreadX[category];
      fits.add(new GrpReport.Fit(
          (long) count[category],
          lastSum[category] / lastCount[category],
          slope[category],
          error,
          (slope[category] - 1) / error));
    }
    return fits;
  }
}
