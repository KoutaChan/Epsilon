package com.epsilon.prism.luck;

import com.epsilon.prism.noise.RootCoefficients.Context;
import com.epsilon.prism.noise.RootCoefficients.Direction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/** 同じ点差をまとめて当てはめた世代別の等張曲線。観測された範囲の中で線形に補間する。 */
public record RootRates(Context context, Direction direction, List<Fold> folds) {
  public RootRates {
    folds = List.copyOf(folds);
  }

  /** 観測した点差と、その位置での補正率。 */
  public record Curve(List<Integer> gaps, List<Double> rates) {
    public Curve {
      gaps = List.copyOf(gaps);
      rates = List.copyOf(rates);
      if (gaps.isEmpty() || gaps.size() != rates.size())
        throw new IllegalArgumentException("Invalid ledger curve dimensions");
      for (int i = 0; i < gaps.size(); i++)
        if ((i > 0 && gaps.get(i) <= gaps.get(i - 1)) || !(rates.get(i) >= 0 && rates.get(i) <= 1))
          throw new IllegalArgumentException("Invalid ledger curve knot");
    }

    double at(int gap) {
      int i = Collections.binarySearch(gaps, gap);
      if (i >= 0) return rates.get(i);
      int hi = -i - 1;
      if (hi == 0 || hi == gaps.size()) return 0;
      int lo = hi - 1;
      double w = (double) (gap - gaps.get(lo)) / (gaps.get(hi) - gaps.get(lo));
      return rates.get(lo) + w * (rates.get(hi) - rates.get(lo));
    }
  }

  /** 1 世代の観測から求めた局面別の曲線。 */
  public record Fold(int cohort, Map<String, Curve> curves) {
    public Fold {
      curves = Map.copyOf(curves);
    }
  }

  private static int key(Context context, LedgerCell.Standing s, int seats) {
    return switch (context) {
      case GLOBAL -> 0;
      case RANK -> s.rank();
      case ROUND -> s.round();
      case RANK_ROUND -> s.round() * seats + s.rank();
    };
  }

  double scale(LedgerCell.Standing standing, int seats) {
    String key = Integer.toString(key(context, standing, seats));
    double result = 1;
    for (Fold fold : folds) {
      Curve c = fold.curves.get(key);
      if (c == null) return 0;
      result = Math.min(result, c.at(standing.gap()));
    }
    return result;
  }

  static Fold fit(
      int cohort,
      Map<LedgerCell.Standing, double[]> moments,
      int seats,
      Context context,
      Direction direction) {
    Map<Integer, TreeMap<Integer, double[]>> partitions = new TreeMap<>();
    moments.forEach(
        (s, sums) -> {
          if (sums[0] > 0) {
            double[] pair =
                partitions
                    .computeIfAbsent(key(context, s, seats), unused -> new TreeMap<>())
                    .computeIfAbsent(s.gap(), unused -> new double[2]);
            pair[0] += sums[0];
            pair[1] += sums[1];
          }
        });
    Map<String, Curve> curves = new TreeMap<>();
    partitions.forEach((key, sums) -> curves.put(Integer.toString(key), fit(sums, direction)));
    return new Fold(cohort, curves);
  }

  private static Curve fit(TreeMap<Integer, double[]> points, Direction dir) {
    int n = points.size(), blocks = 0, index = 0;
    double sign = dir == Direction.DECREASING ? -1 : 1;
    int[] ends = new int[n];
    double[] a = new double[n], b = new double[n], h = new double[n];
    for (double[] pair : points.values()) {
      ends[blocks] = ++index;
      a[blocks] = pair[0];
      b[blocks] = sign * pair[1];
      blocks++;
      while (blocks > 1
          && (dir == Direction.CONSTANT
              || b[blocks - 2] / a[blocks - 2] > b[blocks - 1] / a[blocks - 1])) {
        blocks--;
        a[blocks - 1] += a[blocks];
        b[blocks - 1] += b[blocks];
        ends[blocks - 1] = ends[blocks];
      }
    }
    for (int i = 0; i < blocks; i++)
      Arrays.fill(h, i == 0 ? 0 : ends[i - 1], ends[i], Math.clamp(sign * b[i] / a[i], 0, 1));
    List<Integer> gaps = new ArrayList<>();
    List<Double> rates = new ArrayList<>();
    int i = 0;
    for (int gap : points.keySet()) {
      if (i == 0 || i == n - 1 || h[i] != h[i - 1] || h[i] != h[i + 1]) {
        gaps.add(gap);
        rates.add(h[i]);
      }
      i++;
    }
    return new Curve(gaps, rates);
  }
}
