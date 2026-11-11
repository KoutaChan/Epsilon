package com.epsilon.grp.train;

import java.util.Arrays;

/**
 * 系列ごと・区間ごとに値の平均を取り、隣り合う区間で平均がどれだけ変わるか（D_b = m_{b+1} − m_b）を求める。D_b の分散は、半荘を
 * クラスタとした影響関数で求める: Var = G/(G − 1) · Σ_g ((S_{g,b+1} − m_{b+1} n_{g,b+1})/n_{b+1} − (S_{g,b} − m_b n_{g,b})/n_b)²。
 * G は 2 つの区間のどちらかに値が入った半荘の数。同じ半荘が両方の区間に入るときの共分散も、この式に含まれる。値を全て足してから
 * {@link #differences()} を呼ぶ。
 */
final class AdjacentBins {

  private final int lines;
  private final int bins;
  private final Clusters cells;

  /** 今の半荘で足した値の、セルごとの和と数。半荘が終わったら記録へ移す。 */
  private final double[] gameSum;
  private final double[] gameCount;
  private final int[] touched;
  private int touchedCount;

  /** 半荘ごと・セルごとに和と数を残した記録。半荘 g の記録は {@code gameEnds[g − 1]}（g = 0 なら 0）から {@code gameEnds[g]} の手前まで。 */
  private int[] entryCell = new int[1 << 10];
  private double[] entrySum = new double[entryCell.length];
  private double[] entryCount = new double[entryCell.length];
  private int entries;
  private int[] gameEnds = new int[1 << 10];
  private int games;

  AdjacentBins(int lines, int bins) {
    this.lines = lines;
    this.bins = bins;
    cells = new Clusters(lines * bins);
    gameSum = new double[lines * bins];
    gameCount = new double[lines * bins];
    touched = new int[lines * bins];
  }

  void add(int line, int bin, double value) {
    int cell = line * bins + bin;
    cells.add(cell, value);
    if (gameCount[cell] == 0) {
      touched[touchedCount++] = cell;
    }
    gameSum[cell] += value;
    gameCount[cell]++;
  }

  void endGame() {
    cells.endGame();
    if (touchedCount == 0) {
      return;
    }
    for (int index = 0; index < touchedCount; index++) {
      int cell = touched[index];
      if (entries == entryCell.length) {
        entryCell = Arrays.copyOf(entryCell, 2 * entries);
        entrySum = Arrays.copyOf(entrySum, 2 * entries);
        entryCount = Arrays.copyOf(entryCount, 2 * entries);
      }
      entryCell[entries] = cell;
      entrySum[entries] = gameSum[cell];
      entryCount[entries] = gameCount[cell];
      entries++;
      gameSum[cell] = 0;
      gameCount[cell] = 0;
    }
    touchedCount = 0;
    if (games == gameEnds.length) {
      gameEnds = Arrays.copyOf(gameEnds, 2 * games);
    }
    gameEnds[games++] = entries;
  }

  int bins() {
    return bins;
  }

  double count(int line, int bin) {
    return cells.count(line * bins + bin);
  }

  double clusters(int line, int bin) {
    return cells.clusters(line * bins + bin);
  }

  double mean(int line, int bin) {
    return cells.mean(line * bins + bin);
  }

  double variance(int line, int bin) {
    return cells.variance(line * bins + bin);
  }

  /**
   * 系列ごと・b ごとに D_b とその標準誤差を返す（b は 0 から区間の数 − 2 まで）。2 つの区間のどちらかで、値が入った半荘が 2 つ未満なら
   * 両方とも NaN。
   */
  Paired[][] differences() {
    int pairs = bins - 1;
    Influence influence = new Influence(lines * pairs);
    int start = 0;
    for (int game = 0; game < games; game++) {
      for (int entry = start; entry < gameEnds[game]; entry++) {
        int cell = entryCell[entry];
        int line = cell / bins;
        int bin = cell % bins;
        double deviation = (entrySum[entry] - cells.mean(cell) * entryCount[entry]) / cells.count(cell);
        if (bin < pairs) {
          influence.add(line * pairs + bin, -deviation);
        }
        if (bin > 0) {
          influence.add(line * pairs + bin - 1, deviation);
        }
      }
      influence.endGame();
      start = gameEnds[game];
    }
    Paired[][] differences = new Paired[lines][pairs];
    for (int line = 0; line < lines; line++) {
      for (int bin = 0; bin < pairs; bin++) {
        boolean defined = clusters(line, bin) >= 2 && clusters(line, bin + 1) >= 2;
        differences[line][bin] = defined
            ? new Paired(mean(line, bin + 1) - mean(line, bin), Math.sqrt(influence.variance(line * pairs + bin)))
            : new Paired(Double.NaN, Double.NaN);
      }
    }
    return differences;
  }

  /** 隣り合う区間の対ごとに、半荘 1 つぶんの影響をまとめる。その二乗を全ての半荘で足し合わせる。 */
  private static final class Influence {

    private final double[] game;
    private final double[] squares;
    private final double[] games;
    private final int[] touched;
    private final boolean[] marked;
    private int touchedCount;

    Influence(int pairs) {
      game = new double[pairs];
      squares = new double[pairs];
      games = new double[pairs];
      touched = new int[pairs];
      marked = new boolean[pairs];
    }

    void add(int pair, double value) {
      if (!marked[pair]) {
        marked[pair] = true;
        touched[touchedCount++] = pair;
      }
      game[pair] += value;
    }

    void endGame() {
      for (int index = 0; index < touchedCount; index++) {
        int pair = touched[index];
        squares[pair] += game[pair] * game[pair];
        games[pair]++;
        game[pair] = 0;
        marked[pair] = false;
      }
      touchedCount = 0;
    }

    /** G/(G − 1) · Σ_g 影響²。 */
    double variance(int pair) {
      double g = games[pair];
      return g / (g - 1) * squares[pair];
    }
  }
}
