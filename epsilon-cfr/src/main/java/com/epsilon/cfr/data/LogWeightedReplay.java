package com.epsilon.cfr.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.IntFunction;
import java.util.function.IntToDoubleFunction;
import java.util.random.RandomGenerator;

/** 全行の重みに比例して抽選する、変更不可のサンプラー。抽選した行に重要度の重みを重ねて掛ける必要はない。 */
public final class LogWeightedReplay {
  private final int rowCount;
  private final IntFunction<CfrTrainingRow> loader;
  private final int leafBase;
  private final double[] tree;
  private final Metrics metrics;

  public LogWeightedReplay(
      int rowCount, IntToDoubleFunction logWeight, IntFunction<CfrTrainingRow> loader) {
    this.rowCount = rowCount;
    this.loader = loader;
    if (rowCount < 0 || rowCount > (1 << 28))
      throw new IllegalArgumentException("Invalid replay row count");
    int base = 1;
    while (base < rowCount) base <<= 1;
    leafBase = base;
    tree = new double[base * 2];
    Arrays.fill(tree, Double.NEGATIVE_INFINITY);
    int zeros = 0;
    double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < rowCount; i++) {
      double weight = logWeight.applyAsDouble(i);
      CfrDataChecks.logWeight(weight);
      tree[base + i] = weight;
      max = Math.max(max, weight);
      if (weight == Double.NEGATIVE_INFINITY) zeros++;
    }
    for (int i = 0; i < rowCount; i++) {
      if (tree[base + i] != Double.NEGATIVE_INFINITY) tree[base + i] -= max;
    }
    for (int i = base - 1; i > 0; i--) tree[i] = logAdd(tree[i * 2], tree[i * 2 + 1]);
    double sumScaled = 0;
    double sumSquareScaled = 0;
    for (int i = 0; i < rowCount; i++) {
      double weight = max == Double.NEGATIVE_INFINITY ? 0 : Math.exp(tree[base + i]);
      sumScaled += weight;
      sumSquareScaled += weight * weight;
    }
    double ess = sumSquareScaled == 0 ? 0 : sumScaled * sumScaled / sumSquareScaled;
    metrics =
        new Metrics(
            rowCount,
            rowCount - zeros,
            zeros,
            max == Double.NEGATIVE_INFINITY ? max : max + tree[1],
            ess,
            sumScaled == 0 ? 0 : 1 / sumScaled);
  }

  public CfrTrainingRow sample(RandomGenerator random) {
    if (tree[1] == Double.NEGATIVE_INFINITY)
      throw new IllegalStateException("Replay has no positive-weight rows");
    int node = 1;
    while (node < leafBase) {
      double left = tree[node * 2];
      double right = tree[node * 2 + 1];
      if (left == Double.NEGATIVE_INFINITY) node = node * 2 + 1;
      else if (right == Double.NEGATIVE_INFINITY) node *= 2;
      else {
        // 対数のまま比較し、確率の小さい枝も指数関数のアンダーフローなしに抽選する。
        boolean leftSmaller = left <= right;
        double smallerLogProbability = Math.min(left, right) - tree[node];
        boolean small = logUniform(random) < smallerLogProbability;
        node = node * 2 + ((small == leftSmaller) ? 0 : 1);
      }
    }
    return loader.apply(node - leafBase);
  }

  public List<CfrTrainingRow> sample(int count, RandomGenerator random) {
    if (count < 0) throw new IllegalArgumentException("Negative replay batch size");
    List<CfrTrainingRow> batch = new ArrayList<>(count);
    for (int i = 0; i < count; i++) batch.add(sample(random));
    return List.copyOf(batch);
  }

  public Metrics metrics() {
    return metrics;
  }

  public int size() {
    return rowCount;
  }

  /** 検証時に保持された標本を順に参照する。学習時の抽選には{@code sample}を使う。 */
  public CfrTrainingRow rowAt(int index) {
    if (index < 0 || index >= rowCount) throw new IndexOutOfBoundsException(index);
    return loader.apply(index);
  }

  static double logAdd(double first, double second) {
    if (first == Double.NEGATIVE_INFINITY) return second;
    if (second == Double.NEGATIVE_INFINITY) return first;
    double max = Math.max(first, second);
    return max + Math.log1p(Math.exp(Math.min(first, second) - max));
  }

  private static double logUniform(RandomGenerator random) {
    // 先頭に続く0ビットを複数の乱数にわたって数え、2^-53より小さな確率も扱う。
    long leading = 0;
    long bits;
    while ((bits = random.nextLong()) == 0) leading = Math.addExact(leading, 64);
    int zeroBits = Long.numberOfLeadingZeros(bits);
    leading += zeroBits;
    return Math.log1p(random.nextDouble()) - (leading + 1) * Math.log(2);
  }

  public record Metrics(
      int rows,
      int positiveWeightRows,
      int zeroWeightRows,
      double logTotalWeight,
      double effectiveSampleSize,
      double maximumReplayProbability) {}
}
