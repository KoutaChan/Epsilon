package com.epsilon.agent.eval;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 強さの関係が推移しない組を見つけるための尺度。点ごとの強さ θ を置き、辺ごとの牌山の値の平均が、候補の側の点の θ − 相手の側の点の θ になるように重み付き最小二乗で当てはめる。基準の点の
 * θ は 0 にする。重みは辺ごとの平均の分散の逆数。θ の標準誤差は牌山ごとにまとめた頑健な分散（サンドイッチ型）で求め、同じ牌山を共有する辺の
 * 共分散も含める。θ は牌山の値と同じ単位で、順位なら小さいほど強い。
 */
public final class Ratings {

  private static final double FAMILY_ALPHA = 0.05;
  private static final double SINGLE_LINE = 3;
  private static final double TOLERANCE = 1e-10;

  private final List<Rating> ratings;
  private final List<Residual> residuals;
  private final double chiSquare;
  private final int degrees;
  private final double outlierLine;

  private Ratings(List<Rating> ratings, List<Residual> residuals, double chiSquare, int degrees, double outlierLine) {
    this.ratings = ratings;
    this.residuals = residuals;
    this.chiSquare = chiSquare;
    this.degrees = degrees;
    this.outlierLine = outlierLine;
  }

  /**
   * @param edges 辺。どの点も、辺をたどれば基準の点につながること
   * @param anchor θ を 0 にする基準の点
   * @throws IllegalArgumentException 辺が無いか、基準の点がどの辺にも無いか、基準の点につながらない点があるか、牌山が 2 つ未満の辺か
   *     分散が 0 の辺があるとき
   */
  public static Ratings fit(List<Edge> edges, String anchor) {
    Objects.requireNonNull(anchor, "anchor");
    int count = edges.size();
    if (count == 0) {
      throw new IllegalArgumentException("No edges");
    }
    Map<String, Integer> columns = new LinkedHashMap<>();
    int[] from = new int[count];
    int[] to = new int[count];
    double[] mean = new double[count];
    double[] variance = new double[count];
    double[] weight = new double[count];
    boolean anchored = false;
    for (int e = 0; e < count; e++) {
      Edge edge = edges.get(e);
      anchored |= edge.a().equals(anchor) || edge.b().equals(anchor);
      from[e] = column(columns, edge.a(), anchor);
      to[e] = column(columns, edge.b(), anchor);
      if (edge.perWall().length < 2) {
        throw new IllegalArgumentException("Edge " + edge.a() + " - " + edge.b() + " needs two walls");
      }
      WallMean values = new WallMean();
      for (double value : edge.perWall()) {
        values.add(value);
      }
      mean[e] = values.mean();
      double error = values.error();
      variance[e] = error * error;
      if (!(variance[e] > 0)) {
        throw new IllegalArgumentException("Edge " + edge.a() + " - " + edge.b() + " has zero variance");
      }
      weight[e] = 1 / variance[e];
    }
    if (!anchored) {
      throw new IllegalArgumentException("Anchor " + anchor + " is not on any edge");
    }
    int size = columns.size();
    double[] normal = new double[size * size];
    double[] right = new double[size];
    for (int e = 0; e < count; e++) {
      add(normal, size, from[e], from[e], weight[e]);
      add(normal, size, to[e], to[e], weight[e]);
      add(normal, size, from[e], to[e], -weight[e]);
      add(normal, size, to[e], from[e], -weight[e]);
      if (from[e] >= 0) {
        right[from[e]] += weight[e] * mean[e];
      }
      if (to[e] >= 0) {
        right[to[e]] -= weight[e] * mean[e];
      }
    }
    double[] inverse = invert(normal, size);
    double[] theta = apply(inverse, right, size);

    List<Residual> residuals = new ArrayList<>(count);
    double chiSquare = 0;
    for (int e = 0; e < count; e++) {
      double fitted = at(theta, from[e]) - at(theta, to[e]);
      double residual = mean[e] - fitted;
      chiSquare += weight[e] * residual * residual;
      double leverage = weight[e] * quadratic(inverse, size, from[e], to[e]);
      double spread = 1 - leverage;
      double z = spread > TOLERANCE ? residual / Math.sqrt(variance[e] * spread) : 0;
      Edge edge = edges.get(e);
      residuals.add(new Residual(edge.a(), edge.b(), mean[e], fitted, z));
    }

    double[] meat = clustered(edges, from, to, mean, weight, size);
    double[] covariance = multiply(multiply(inverse, meat, size), inverse, size);

    int[] touching = new int[size + 1];
    for (int e = 0; e < count; e++) {
      touching[from[e] + 1]++;
      touching[to[e] + 1]++;
    }
    List<Rating> ratings = new ArrayList<>(size + 1);
    ratings.add(new Rating(anchor, 0, 0, touching[0]));
    for (Map.Entry<String, Integer> entry : columns.entrySet()) {
      int j = entry.getValue();
      ratings.add(new Rating(entry.getKey(), theta[j], Math.sqrt(Math.max(0, covariance[j * size + j])), touching[j + 1]));
    }
    double line = Normal.upperPoint(FAMILY_ALPHA / (2.0 * count));
    return new Ratings(List.copyOf(ratings), List.copyOf(residuals), chiSquare, count - size, line);
  }

  private static int column(Map<String, Integer> columns, String label, String anchor) {
    if (label.equals(anchor)) {
      return -1;
    }
    Integer column = columns.get(label);
    if (column == null) {
      column = columns.size();
      columns.put(label, column);
    }
    return column;
  }

  private static void add(double[] matrix, int size, int row, int column, double value) {
    if (row >= 0 && column >= 0) {
      matrix[row * size + column] += value;
    }
  }

  private static double at(double[] vector, int index) {
    return index < 0 ? 0 : vector[index];
  }

  /** @return a の成分が 1、b の成分が −1、ほかの成分が 0 のベクトル x での xᵀ M x（負の添字は基準の点で、成分を持たない） */
  private static double quadratic(double[] matrix, int size, int a, int b) {
    double value = 0;
    if (a >= 0) {
      value += matrix[a * size + a];
    }
    if (b >= 0) {
      value += matrix[b * size + b];
    }
    if (a >= 0 && b >= 0) {
      value -= 2 * matrix[a * size + b];
    }
    return value;
  }

  /**
   * 正規方程式の右辺の誤差の分散を、牌山ごとにまとめて求める。牌山 c の寄与は Σ_e x_e w_e (y_ec − ȳ_e) / √(n_e (n_e − 1)) で（x_e は辺 e の両端の点に ±1 を置いた行、w_e は辺の重み、y_ec は辺 e の牌山 c の値、ȳ_e と n_e は辺の平均と牌山の数）、牌山の種が
   * 同じ辺どうしでは、添字が同じ値を 1 つの牌山として足す。辺が 1 本だけなら、その辺の平均の分散に一致する。
   */
  private static double[] clustered(List<Edge> edges, int[] from, int[] to, double[] mean, double[] weight, int size) {
    Map<Long, List<Integer>> groups = new LinkedHashMap<>();
    for (int e = 0; e < edges.size(); e++) {
      groups.computeIfAbsent(edges.get(e).wallSeed(), seed -> new ArrayList<>()).add(e);
    }
    double[] meat = new double[size * size];
    for (List<Integer> group : groups.values()) {
      group.sort(Comparator.comparingInt((Integer e) -> edges.get(e).perWall().length).reversed());
      Map<Integer, Integer> local = new HashMap<>();
      for (int e : group) {
        for (int column : new int[] {from[e], to[e]}) {
          if (column >= 0) {
            local.putIfAbsent(column, local.size());
          }
        }
      }
      int[] global = new int[local.size()];
      local.forEach((column, index) -> global[index] = column);
      int members = group.size();
      int[] localFrom = new int[members];
      int[] localTo = new int[members];
      double[] scale = new double[members];
      double[][] values = new double[members][];
      for (int m = 0; m < members; m++) {
        int e = group.get(m);
        localFrom[m] = from[e] < 0 ? -1 : local.get(from[e]);
        localTo[m] = to[e] < 0 ? -1 : local.get(to[e]);
        values[m] = edges.get(e).perWall();
        scale[m] = weight[e] / Math.sqrt((double) values[m].length * (values[m].length - 1));
      }
      double[] score = new double[global.length];
      int walls = values[0].length;
      for (int c = 0; c < walls; c++) {
        Arrays.fill(score, 0);
        for (int m = 0; m < members && values[m].length > c; m++) {
          double contribution = scale[m] * (values[m][c] - mean[group.get(m)]);
          if (localFrom[m] >= 0) {
            score[localFrom[m]] += contribution;
          }
          if (localTo[m] >= 0) {
            score[localTo[m]] -= contribution;
          }
        }
        for (int r = 0; r < global.length; r++) {
          double left = score[r];
          if (left == 0) {
            continue;
          }
          int row = global[r] * size;
          for (int s = 0; s < global.length; s++) {
            meat[row + global[s]] += left * score[s];
          }
        }
      }
    }
    return meat;
  }

  /** @return 対称な正定値行列の逆行列（Cholesky 分解で求める） */
  private static double[] invert(double[] matrix, int size) {
    double[] lower = new double[size * size];
    for (int j = 0; j < size; j++) {
      double diagonal = matrix[j * size + j];
      for (int k = 0; k < j; k++) {
        diagonal -= lower[j * size + k] * lower[j * size + k];
      }
      if (!(diagonal > TOLERANCE * matrix[j * size + j])) {
        throw new IllegalArgumentException("Ratings are not all connected to the anchor");
      }
      double pivot = Math.sqrt(diagonal);
      lower[j * size + j] = pivot;
      for (int i = j + 1; i < size; i++) {
        double value = matrix[i * size + j];
        for (int k = 0; k < j; k++) {
          value -= lower[i * size + k] * lower[j * size + k];
        }
        lower[i * size + j] = value / pivot;
      }
    }
    double[] inverse = new double[size * size];
    double[] column = new double[size];
    for (int c = 0; c < size; c++) {
      Arrays.fill(column, 0);
      column[c] = 1;
      for (int i = 0; i < size; i++) {
        double value = column[i];
        for (int k = 0; k < i; k++) {
          value -= lower[i * size + k] * column[k];
        }
        column[i] = value / lower[i * size + i];
      }
      for (int i = size - 1; i >= 0; i--) {
        double value = column[i];
        for (int k = i + 1; k < size; k++) {
          value -= lower[k * size + i] * column[k];
        }
        column[i] = value / lower[i * size + i];
      }
      for (int i = 0; i < size; i++) {
        inverse[i * size + c] = column[i];
      }
    }
    return inverse;
  }

  private static double[] apply(double[] matrix, double[] vector, int size) {
    double[] product = new double[size];
    for (int i = 0; i < size; i++) {
      double sum = 0;
      for (int k = 0; k < size; k++) {
        sum += matrix[i * size + k] * vector[k];
      }
      product[i] = sum;
    }
    return product;
  }

  private static double[] multiply(double[] left, double[] right, int size) {
    double[] product = new double[size * size];
    for (int i = 0; i < size; i++) {
      for (int k = 0; k < size; k++) {
        double factor = left[i * size + k];
        if (factor == 0) {
          continue;
        }
        for (int j = 0; j < size; j++) {
          product[i * size + j] += factor * right[k * size + j];
        }
      }
    }
    return product;
  }

  /** @return 点ごとの θ。基準の点が先頭で、ほかは辺に初めて出た順 */
  public List<Rating> ratings() {
    return ratings;
  }

  /** @return 辺ごとの当てはめの残差（辺の順） */
  public List<Residual> residuals() {
    return residuals;
  }

  /** @return χ² を自由度（辺の数 − 当てはめた点の数）で割った値。自由度が 0 以下なら NaN */
  public double chiSquarePerDegree() {
    return degrees > 0 ? chiSquare / degrees : Double.NaN;
  }

  /**
   * @return 外れた辺とみなす |z| のしきい値。当てはめが正しいとき、どれか 1 本の辺でも偶然に超える確率が 5% 以下になる値（Bonferroni の
   *     補正）
   */
  public double outlierLine() {
    return outlierLine;
  }

  /** @return |z| が外れた辺のしきい値を超える辺 */
  public List<Residual> outliers() {
    return residuals.stream().filter(residual -> Math.abs(residual.z()) > outlierLine).toList();
  }

  /** @return |z| が 3 を超える辺の数 */
  public int beyondThree() {
    return (int) residuals.stream().filter(residual -> Math.abs(residual.z()) > SINGLE_LINE).count();
  }

  /** @return 当てはめが正しいときの、|z| が 3 を超える辺の数の期待値 */
  public double beyondThreeExpected() {
    return 2 * Normal.upper(SINGLE_LINE) * residuals.size();
  }

  /**
   * 1 つの卓を表す辺と、その牌山ごとの値。
   *
   * @param a 候補の側の点。牌山の値は、この点から見た値
   * @param b 相手の側の点
   * @param wallSeed 牌山の種。種が同じ辺どうしは、添字が同じなら同じ牌山
   * @param perWall 牌山ごとの値（牌山の番号の順、2 つ以上）。渡した後は書き換えないこと
   */
  public record Edge(String a, String b, long wallSeed, double[] perWall) {
    public Edge {
      Objects.requireNonNull(a, "a");
      Objects.requireNonNull(b, "b");
      Objects.requireNonNull(perWall, "perWall");
      if (a.equals(b)) {
        throw new IllegalArgumentException("Edge joins " + a + " to itself");
      }
    }
  }

  /**
   * @param value θ
   * @param error θ の頑健な標準誤差。基準の点は 0
   * @param edges この点に付く辺の数
   */
  public record Rating(String label, double value, double error, int edges) {}

  /**
   * @param observed 辺の牌山ごとの値の平均
   * @param fitted 候補の側の点の θ − 相手の側の点の θ
   * @param z 残差を、その標準誤差で割った値。標準誤差は当てはめで減る分（てこ比）を除いて求める。1 本の辺だけで θ が決まる点の辺では 0
   */
  public record Residual(String a, String b, double observed, double fitted, double z) {}
}
