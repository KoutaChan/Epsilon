package com.epsilon.prism.luck;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtEngine;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Base64;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

/** 候補間で中心化した特徴と効用の二次モーメント。最後の列は教師の効用。 */
final class RootRidge {
  private static final int BLOCK = 1024;
  final int size;
  final double[] gram;
  private final double[] mean, phi;
  private double[] batch;
  private final ArrayDeque<Block> pending = new ArrayDeque<>();
  private Executor executor;
  private int capacity = 1;
  private int used;

  private record Block(double[] input, CompletableFuture<double[]> product) {}

  void parallelize(Executor executor, int capacity) {
    this.executor = executor;
    this.capacity = capacity;
  }

  RootRidge(int width, int columns, String previous) {
    size = width * columns + 1;
    gram = new double[size * size];
    batch = new double[size * BLOCK];
    mean = new double[size];
    phi = new double[width];
    if (previous != null) {
      var bytes = ByteBuffer.wrap(Base64.getDecoder().decode(previous));
      if (bytes.remaining() != (long) size * (size + 1) / 2 * Double.BYTES)
        throw new IllegalArgumentException("Invalid ledger moment dimensions");
      for (int j = 0; j < size; j++)
        for (int i = 0; i <= j; i++) gram[i + j * size] = bytes.getDouble();
    }
  }

  void add(
      FitRow row,
      RootFeatures root,
      RootFeatures.Layout layout,
      int[] actions,
      int n,
      int[] columns,
      float[] values) {
    if (used + n > BLOCK) submit();
    Arrays.fill(mean, 0);
    for (int j = 0; j < n; j++) {
      root.fill(j, layout, phi);
      int start = (used + j) * size, index = start;
      for (int col : columns) {
        double l = row.ledger(actions[j], col);
        for (double x : phi) batch[index++] = l * x;
      }
      batch[index] = values[actions[j]];
      for (int i = 0; i < size; i++) mean[i] += batch[start + i];
    }
    double w = 1 / Math.sqrt(n);
    for (int i = 0; i < size; i++) mean[i] /= n;
    for (int j = 0; j < n; j++) {
      int start = (used + j) * size;
      for (int i = 0; i < size; i++) batch[start + i] = (batch[start + i] - mean[i]) * w;
    }
    used += n;
  }

  private void submit() {
    if (used == 0) return;
    double[] input = batch;
    int count = used;
    if (capacity == 1) {
      accumulate(product(input, count));
    } else {
      pending.addLast(
          new Block(input, CompletableFuture.supplyAsync(() -> product(input, count), executor)));
      batch = pending.size() == capacity ? mergeFirst() : new double[input.length];
    }
    used = 0;
  }

  private double[] product(double[] input, int count) {
    try (var cpu = NDManager.newBaseManager(Device.cpu(), PtEngine.ENGINE_NAME)) {
      var rows = cpu.create(DoubleBuffer.wrap(input, 0, count * size), new Shape(count, size));
      return rows.transpose().matMul(rows).toDoubleArray();
    }
  }

  private void accumulate(double[] product) {
    for (int j = 0; j < size; j++)
      for (int i = 0; i <= j; i++) gram[i + j * size] += product[i + j * size];
  }

  private double[] mergeFirst() {
    var block = pending.removeFirst();
    accumulate(block.product().join());
    return block.input();
  }

  private void flush() {
    submit();
    while (!pending.isEmpty()) mergeFirst();
  }

  String save() {
    flush();
    var bytes = ByteBuffer.allocate(size * (size + 1) / 2 * Double.BYTES);
    for (int j = 0; j < size; j++) for (int i = 0; i <= j; i++) bytes.putDouble(gram[i + j * size]);
    return Base64.getEncoder().encodeToString(bytes.array());
  }

  record Solution(double[] theta, Double ridge) {}

  /** 訓練世代の二次モーメントを標準化し、次世代の二乗誤差で連続な縮小強度と両端を比べる。 */
  Solution select(RootRidge holdout, RootFeatures.Layout layout) {
    flush();
    holdout.flush();
    int n = size - 1;
    double[] scale = new double[n], q = new double[n * n];
    for (int i = 0; i < n; i++) scale[i] = Math.sqrt(gram[i * size + i]);
    for (int j = 0; j < n; j++)
      for (int i = 0; i <= j; i++) {
        if (scale[i] > 0 && scale[j] > 0) q[i + j * n] = gram[i + j * size] / scale[i] / scale[j];
      }
    double dependenceError = dependenceError(q, scale, layout);
    double[] eigen = SymmetricEigen.decompose(q, n);
    double cutoff = Math.max(dependenceError, n * Math.ulp(1d) * eigen[n - 1]);
    double[] z = new double[n], v = new double[n], g = new double[n * n];
    for (int j = 0; j < n; j++)
      for (int i = 0; i < n; i++) {
        q[i + j * n] = scale[i] > 0 && eigen[j] > cutoff ? q[i + j * n] / scale[i] : 0;
        z[j] += q[i + j * n] * gram[i + n * size];
        v[j] += q[i + j * n] * holdout.gram[i + n * size];
        g[i + j * n] = holdout.gram[Math.min(i, j) + Math.max(i, j) * size];
      }
    double[] tmp = new double[n], mat = new double[n * n];
    for (int j = 0; j < n; j++) {
      multiply(g, q, j * n, tmp);
      for (int i = 0; i < n; i++) mat[i + j * n] = dot(q, i * n, tmp);
    }
    var risk = new Risk(eigen, z, v, mat, cutoff, holdout.gram[n + n * size]);
    double lambda = risk.minimum();
    double[] theta = new double[n];
    if (Double.isFinite(lambda)) {
      double[] a = risk.coordinates(lambda);
      multiply(q, a, 0, theta);
    }
    return new Solution(theta, Double.isFinite(lambda) ? lambda : null);
  }

  /**
   * 定数は全候補種別の指示値の和に等しい。標準化後のこの零空間に Gram 行列を掛けた残差の
   * Frobenius ノルムを、積率を実際に足したときの丸め誤差として求める。
   */
  private static double dependenceError(
      double[] upper, double[] scale, RootFeatures.Layout layout) {
    if (layout.types == 0) return 0;
    int n = scale.length;
    double squared = 0;
    for (int block = 0; block < n; block += layout.width) {
      int constant = block + layout.at(RootFeatures.Field.CONSTANT), first = block + layout.common;
      double norm = scale[constant] * scale[constant];
      for (int j = first; j < first + layout.types; j++) norm += scale[j] * scale[j];
      if (norm == 0) continue;
      for (int i = 0; i < n; i++) {
        double residual =
            upper[Math.min(i, constant) + Math.max(i, constant) * n] * scale[constant];
        for (int j = first; j < first + layout.types; j++)
          residual -= upper[Math.min(i, j) + Math.max(i, j) * n] * scale[j];
        squared += residual * residual / norm;
      }
    }
    return Math.sqrt(squared);
  }

  /** 列優先の正方行列とベクトルの積を求める。 */
  private static void multiply(double[] matrix, double[] vector, int start, double[] result) {
    int n = result.length;
    Arrays.fill(result, 0);
    for (int j = 0; j < n; j++) {
      double factor = vector[start + j];
      if (factor == 0) continue;
      for (int i = 0; i < n; i++) result[i] += factor * matrix[j * n + i];
    }
  }

  private static double dot(double[] column, int start, double[] vector) {
    double sum = 0;
    for (int i = 0; i < vector.length; i++) sum += column[start + i] * vector[i];
    return sum;
  }

  private record Risk(
      double[] eigen, double[] z, double[] v, double[] mat, double cutoff, double rawEnergy) {
    private record Point(double x, double[] a, double loss) {}

    private record Interval(Point left, Point right, double lower) {}

    private double[] coordinates(double lambda) {
      double[] a = new double[z.length];
      for (int i = 0; i < a.length; i++) if (eigen[i] > cutoff) a[i] = z[i] / (eigen[i] + lambda);
      return a;
    }

    private Point evaluate(double x, double reference, double[] gradient) {
      double[] a = coordinates(reference * x / (1 - x));
      multiply(mat, a, 0, gradient);
      double loss = dot(a, 0, gradient) - 2 * dot(a, 0, v);
      for (int i = 0; i < a.length; i++) gradient[i] = 2 * (gradient[i] - v[i]);
      return new Point(x, a, loss);
    }

    /**
     * 検証損失は係数ベクトルの凸二次式なので接平面が下界になる。縮小強度 λ を x = λ/(λ+s) に移し、
     * 区間の両端を結ぶ線分から各係数が離れうる量を引く。各係数は x の一次分数関数で、
     * 離れうる量は両端の差の絶対値と、両端の分母の平方根の差を和で割った絶対値の積になる。
     */
    private double lower(
        Point left, Point right, Point tangent, double[] gradient, double reference) {
      double lo = 0, hi = 0, error = 0;
      for (int i = 0; i < z.length; i++) {
        if (z[i] == 0 || eigen[i] <= cutoff) continue;
        lo += gradient[i] * (left.a[i] - tangent.a[i]);
        hi += gradient[i] * (right.a[i] - tangent.a[i]);
        double a = Math.sqrt((1 - left.x) * eigen[i] + left.x * reference);
        double b = Math.sqrt((1 - right.x) * eigen[i] + right.x * reference);
        error += Math.abs(gradient[i] * (right.a[i] - left.a[i]) * (a - b) / (a + b));
      }
      return tangent.loss + Math.min(lo, hi) - error;
    }

    double minimum() {
      int n = z.length;
      double smallest = Double.POSITIVE_INFINITY;
      for (int i = 0; i < n; i++)
        if (eigen[i] > cutoff && z[i] != 0) smallest = Math.min(smallest, eigen[i]);
      if (!Double.isFinite(smallest) || rawEnergy == 0) return Double.POSITIVE_INFINITY;
      double reference = Math.sqrt(smallest * eigen[n - 1]);
      double[] gradient = new double[n];
      Point left = evaluate(0, reference, gradient), right = evaluate(1, reference, gradient);
      Point best = left.loss < right.loss ? left : right;
      double tolerance = n * Math.ulp(rawEnergy);
      var pending = new PriorityQueue<Interval>(Comparator.comparingDouble(Interval::lower));
      pending.add(new Interval(left, right, Double.NEGATIVE_INFINITY));
      while (!pending.isEmpty() && pending.peek().lower < best.loss - tolerance) {
        var interval = pending.remove();
        left = interval.left;
        right = interval.right;
        double x = (left.x + right.x) / 2;
        if (x == left.x || x == right.x) continue;
        var mid = evaluate(x, reference, gradient);
        if (mid.loss < best.loss) best = mid;
        double lo = lower(left, mid, mid, gradient, reference);
        double hi = lower(mid, right, mid, gradient, reference);
        if (lo < best.loss - tolerance) pending.add(new Interval(left, mid, lo));
        if (hi < best.loss - tolerance) pending.add(new Interval(mid, right, hi));
      }
      return best.loss < -tolerance ? reference * best.x / (1 - best.x) : Double.POSITIVE_INFINITY;
    }
  }
}
