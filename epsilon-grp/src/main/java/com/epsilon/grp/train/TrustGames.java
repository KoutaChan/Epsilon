package com.epsilon.grp.train;

import java.util.SplittableRandom;

/**
 * 点数だけの Φ を初めて置き換える関門（{@link GateKind#REPLACEMENT}）で、2 SE の判定に要る半荘の数を見積もった値。候補の Φ と
 * 使用中の Φ を、どちらも学習に使っていない半荘で比べる。差は対の MSE の差（候補 − 使用中。席・局末あたり）で、関門と同じ値
 * （{@link GrpMetrics#pairedMse}）。差の標準偏差は、半荘を復元抽出するブートストラップで求める。数える半荘は、半荘が続く局末を
 * 1 つ以上持つものだけ。
 *
 * <p>差の標準偏差は半荘の数の平方根に反比例するので、半荘 G 個で測った標準偏差 SD から、半荘 n 個で測ったときの標準偏差を
 * SD·√(G/n) と見積もる。関門は差が −2 SE より小さいことを求めるので、測った差 d（負）がちょうど −2 SE になる半荘の数は
 * G·(2·SD/d)² になる。
 *
 * @param games 比べた半荘の数
 * @param handEnds 比べた、半荘が続く局末の数
 * @param difference 差と、半荘をクラスタとしたその標準誤差
 * @param bootstrapSd 差のブートストラップの標準偏差
 * @param resamples ブートストラップの回数
 */
public record TrustGames(int games, long handEnds, Paired difference, double bootstrapSd, int resamples) {

  /** 検出力 95% の片側の z。 */
  static final double POWER_Z = 1.6448536269514722;

  /** 半荘あたりの、半荘が続く局末の数。 */
  public double handEndsPerGame() {
    return (double) handEnds / games;
  }

  /** 測った差がちょうど関門のしきい値（−2 SE）になる半荘の数を返す。候補が良くなっていなければ（差が 0 以上）+∞。 */
  public double atLine() {
    return gamesFor(-GateKind.REPLACEMENT.standardErrors());
  }

  /** 真の差が測った差と同じなら、95% の確率で差が関門のしきい値を下回る半荘の数を返す。候補が良くなっていなければ +∞。 */
  public double withPower() {
    return gamesFor(-GateKind.REPLACEMENT.standardErrors() + POWER_Z);
  }

  /** 測った差が、その数の半荘で測ったときの標準偏差の z 倍になる半荘の数を返す。差が 0 以上なら +∞。 */
  private double gamesFor(double z) {
    double mean = difference.mean();
    if (!(mean < 0)) {
      return Double.POSITIVE_INFINITY;
    }
    double ratio = z * bootstrapSd / mean;
    return games * ratio * ratio;
  }

  /**
   * 2 つの Φ の局末の値から測る。
   *
   * @param used 使用中の Φ の値
   * @param candidate 候補の値
   * @param resamples ブートストラップの回数（2 以上）
   * @param seed ブートストラップの乱数の種
   * @throws IllegalArgumentException 値がその局末のものでないとき、回数が 2 未満のとき、局末を持つ半荘が 2 つ未満のとき
   */
  public static TrustGames measure(
      HandEnds ends, HandEndValues used, HandEndValues candidate, int resamples, long seed) {
    GrpMetrics.requireValues(ends, used);
    GrpMetrics.requireValues(ends, candidate);
    if (resamples < 2) {
      throw new IllegalArgumentException("The bootstrap needs at least two resamples: " + resamples);
    }
    int seats = ends.seats;
    double[] sums = new double[ends.games()];
    int[] counts = new int[ends.games()];
    int games = 0;
    for (int game = 0; game < ends.games(); game++) {
      int first = ends.firstHand[game] * seats;
      int last = ends.firstHand[game + 1] * seats;
      if (last == first) {
        continue;
      }
      double sum = 0;
      for (int j = first; j < last; j++) {
        double u = ends.utilities[ends.finalRank[j]];
        double ofCandidate = u - candidate.phi[j];
        double ofUsed = u - used.phi[j];
        sum += ofCandidate * ofCandidate - ofUsed * ofUsed;
      }
      sums[games] = sum;
      counts[games] = last - first;
      games++;
    }
    if (games < 2) {
      throw new IllegalArgumentException("The bootstrap needs at least two games with hand ends, got " + games);
    }
    SplittableRandom random = new SplittableRandom(seed);
    double mean = 0;
    double squares = 0;
    for (int resample = 1; resample <= resamples; resample++) {
      double sum = 0;
      long count = 0;
      for (int draw = 0; draw < games; draw++) {
        int game = random.nextInt(games);
        sum += sums[game];
        count += counts[game];
      }
      double value = sum / count;
      double step = value - mean;
      mean += step / resample;
      squares += step * (value - mean);
    }
    return new TrustGames(
        games,
        ends.handEnds(),
        GrpMetrics.pairedMse(ends, used, candidate),
        Math.sqrt(squares / (resamples - 1)),
        resamples);
  }
}
