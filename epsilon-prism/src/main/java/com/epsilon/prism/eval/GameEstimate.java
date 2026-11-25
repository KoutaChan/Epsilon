package com.epsilon.prism.eval;

import com.epsilon.prism.core.DecisionClass;
import java.util.Arrays;
import java.util.Collection;

/** 半荘内の相関を許した判断あたりの平均。分母が空なら平均はなく、半荘が一つ以下なら標準誤差はない。 */
public record GameEstimate(long positions, int hanchan, Double mean, Double se) {

  /** 半荘一つのクラス別の和と件数。全クラスの和にも同じ判断を一度だけ足す。 */
  public static final class Game {
    private final long[] counts = new long[DecisionClass.COUNT + 1];
    private final double[] sums = new double[counts.length];

    public void add(DecisionClass group, double value) {
      if (!Double.isFinite(value))
        throw new IllegalArgumentException("Non-finite observation: " + value);
      counts[group.ordinal()]++;
      sums[group.ordinal()] += value;
      counts[DecisionClass.COUNT]++;
      sums[DecisionClass.COUNT] += value;
    }

    /** 同じ行を測った半荘の差。行ごとの差の和は、同じクラスの二つの和の差に等しい。 */
    public Game difference(Game reference) {
      if (!Arrays.equals(counts, reference.counts))
        throw new IllegalArgumentException("Paired row counts differ");
      var delta = new Game();
      System.arraycopy(counts, 0, delta.counts, 0, counts.length);
      for (int i = 0; i < sums.length; i++) delta.sums[i] = sums[i] - reference.sums[i];
      return delta;
    }
  }

  /** クラスが null なら全判断を使う。G は判断のある半荘の数。半荘 g の件数 n_g と和 S_g に対し分散は G/(G−1) Σ(S_g−n_g μ)² / N²。 */
  public static GameEstimate of(Collection<Game> games, DecisionClass group) {
    int index = group == null ? DecisionClass.COUNT : group.ordinal();
    long n = 0;
    int clusters = 0;
    double total = 0;
    for (Game game : games) {
      if (game.counts[index] > 0) clusters++;
      n += game.counts[index];
      total += game.sums[index];
    }
    if (n == 0) return new GameEstimate(0, 0, null, null);
    double mean = total / n;
    double squares = 0;
    for (Game game : games) {
      double residual = game.sums[index] - game.counts[index] * mean;
      squares += residual * residual;
    }
    return new GameEstimate(
        n,
        clusters,
        mean,
        clusters < 2 ? null : Math.sqrt(squares * clusters / (clusters - 1)) / n);
  }
}
