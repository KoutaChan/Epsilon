package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.prism.collect.HandValue;
import java.util.Arrays;
import java.util.List;
import java.util.function.Consumer;

/**
 * 点数のなめらかな関数の Φ（{@link PhiEntryKind#SCORE_SMOOTH}）: 4 席の最終の点数を互いに独立な正規分布 N(局末の点数, σ²)、σ = {@value
 * #SIGMA_PER_HAND}·√残りの局数 とみなした最終順位の期待効用 E[u(順位)]。点数の順位（段差の関数）では、全ての候補で判断の席の順位が変わらない判断の
 * ラベルの差がちょうど 0 になる。この Φ は点差に応じてなめらかに動くので、その判断にも差が残る（GRP なしの出発の候補）。
 *
 * <p>σ の係数は PRISM の自己対局（v224 の監査、65,536 半荘・局末 705,246）で、局末から半荘の終わりまでの点の動きの標準偏差を √残りの局数 で
 * 割ったもの（次局が東 1〜南 4 で 5,007〜5,170、全体 5,137）。人の牌譜は使わない。残りの局数は次局から南 4 局までの局の数で、西入は 1。
 *
 * <p>席 i の順位の分布は、自分の最終点 X をガウス・エルミートの {@value #NODES} 点で積分し、X を条件にした他の 3 席の「X より上」の独立な確率
 * から作る。決定的（{@link StrictMath} だけを使うので、JIT や CPU に依らず同じ入力に同じビットの値）で、自分の点に対して単調。半荘が終わった
 * 局末には使わない（最終順位の効用そのもの）。学習の JVM の局末の値は (4 席の点数, 次局) の直接写像のメモ（{@value #MEMO_SLOTS} 枠）を通す。
 */
public final class ScoreSmoothPhi {

  /** 1 局あたりの最終の点の標準偏差の係数（点）。 */
  public static final double SIGMA_PER_HAND = 5140;

  /** ガウス・エルミートの点の数。 */
  public static final int NODES = 20;

  private static final int SEATS = 4;

  /** 学習の JVM の局末の値のメモの枠の数（2 の冪）。 */
  static final int MEMO_SLOTS = 1 << 18;

  /** 標準正規の積分の点 z_k = √2·x_k と重み w_k/√π（和 1）。 */
  private static final double[] Z = new double[NODES];

  private static final double[] W = new double[NODES];

  /** erfc の近似の多項式の係数（u = 1/(1 + |x|/2) の冪の順）。 */
  private static final double[] ERFC = {
    -1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806, 0.27886807, -1.13520398, 1.48851587,
    -0.82215223, 0.17087277
  };

  static {
    hermite(Z, W);
    for (int k = 0; k < NODES; k++) {
      Z[k] *= StrictMath.sqrt(2);
      W[k] /= StrictMath.sqrt(Math.PI);
    }
  }

  private ScoreSmoothPhi() {}

  /**
   * 次局からの残りの局の数（次局から南 4 局まで、西入は 1）。
   *
   * @param kyokuIndex 次局の局番号（東 1 = 0）
   */
  public static int remainingHands(int kyokuIndex) {
    if (kyokuIndex < 0) throw new IllegalArgumentException("Negative kyoku index " + kyokuIndex);
    return Math.max(1, HanchanProgression.SOUTH_END_KYOKU + 1 - kyokuIndex);
  }

  /**
   * 順位分布。
   *
   * @param scores 4 席の局末の点数（精算後）
   * @param kyokuIndex 次局の局番号
   * @param out 席 i・順位 r を [4i + r] に書く 16 の配列
   */
  public static void rankProbabilities(int[] scores, int kyokuIndex, double[] out) {
    if (scores.length != SEATS || out.length != SEATS * SEATS)
      throw new IllegalArgumentException("Four scores and sixteen probabilities required");
    double sigma = SIGMA_PER_HAND * StrictMath.sqrt(remainingHands(kyokuIndex));
    Arrays.fill(out, 0);
    double[] above = new double[SEATS - 1];
    double[] ranks = new double[SEATS];
    for (int seat = 0; seat < SEATS; seat++) {
      for (int k = 0; k < NODES; k++) {
        int other = 0;
        for (int rival = 0; rival < SEATS; rival++) {
          if (rival == seat) continue;
          above[other++] = normalCdf((scores[rival] - scores[seat]) / sigma - Z[k]);
        }
        // 上にいる席の数の分布（3 つの独立なベルヌーイの和）
        Arrays.fill(ranks, 0);
        ranks[0] = 1;
        for (int j = 0; j < SEATS - 1; j++) {
          double p = above[j];
          for (int r = j + 1; r > 0; r--) ranks[r] = ranks[r] * (1 - p) + ranks[r - 1] * p;
          ranks[0] *= 1 - p;
        }
        for (int r = 0; r < SEATS; r++) out[SEATS * seat + r] += W[k] * ranks[r];
      }
    }
  }

  /**
   * 4 席の期待効用。
   *
   * @param scores 4 席の局末の点数（精算後）
   * @param kyokuIndex 次局の局番号
   */
  public static double[] utilities(int[] scores, int kyokuIndex, EpsilonUtilityProfile profile) {
    double[] ranks = new double[SEATS * SEATS];
    rankProbabilities(scores, kyokuIndex, ranks);
    return expected(ranks, profile);
  }

  /**
   * 学習の JVM の局末の値（次局の局番号は公開系列の最後の 1 歩から読む）。ドライバのスレッドから同期して呼ばれるので、同じ (点数, 次局) の
   * 繰り返しは直接写像のメモで引く（値は入力だけで決まるので、メモの当たり外れで値は変わらない）。渡す配列は呼び手ごとの写し。
   */
  static HandValue handValue(EpsilonUtilityProfile profile) {
    return new HandValue() {
      private final int[][] keys = new int[MEMO_SLOTS][];
      private final double[][] values = new double[MEMO_SLOTS][];
      private long requests, computed;

      @Override
      public void request(float[] prefix, int[] scores, Consumer<double[]> completion) {
        requests++;
        int kyoku = kyokuOf(prefix);
        int[] key = {scores[0], scores[1], scores[2], scores[3], kyoku};
        int slot = (Arrays.hashCode(key) * 0x9E3779B9 >>> 14) & (MEMO_SLOTS - 1);
        if (!Arrays.equals(keys[slot], key)) {
          computed++;
          values[slot] = utilities(scores, kyoku, profile);
          keys[slot] = key;
        }
        completion.accept(values[slot].clone());
      }

      @Override
      public int pending() {
        return 0;
      }

      @Override
      public Stats stats() {
        return new Stats(requests, requests - computed, computed, 0, 0);
      }
    };
  }

  /** 記録した局末の評価。 */
  static PhiValues values(List<HandEndInput> inputs, EpsilonUtilityProfile profile) {
    double[][] utilities = new double[inputs.size()][];
    double[][] ranks = new double[inputs.size()][];
    for (int i = 0; i < inputs.size(); i++) {
      HandStart next = inputs.get(i).next();
      ranks[i] = new double[SEATS * SEATS];
      rankProbabilities(next.scores(), next.kyokuIndex(), ranks[i]);
      utilities[i] = expected(ranks[i], profile);
    }
    return new PhiValues(utilities, ranks);
  }

  /** 次局 root の公開系列の最後の 1 歩の局番号。 */
  static int kyokuOf(float[] prefix) {
    int step = prefix.length - EpsilonGrpFeature.FEATURE_SIZE;
    if (step < 0 || prefix.length % EpsilonGrpFeature.FEATURE_SIZE != 0)
      throw new IllegalArgumentException("GRP prefix length " + prefix.length);
    return Math.round(prefix[step + EpsilonGrpFeature.KYOKU_INDEX] * EpsilonGrpFeature.KYOKU_SCALE);
  }

  private static double[] expected(double[] ranks, EpsilonUtilityProfile profile) {
    double[] utilities = new double[SEATS];
    for (int seat = 0; seat < SEATS; seat++) {
      double sum = 0;
      for (int r = 0; r < SEATS; r++) sum += ranks[SEATS * seat + r] * profile.utilityForRankDouble(r);
      utilities[seat] = sum;
    }
    return utilities;
  }

  /** 標準正規の分布関数（erfc の Chebyshev 近似、相対誤差 1.2e-7。t に対して単調）。 */
  static double normalCdf(double t) {
    double x = -t / StrictMath.sqrt(2);
    double z = Math.abs(x);
    double u = 1 / (1 + 0.5 * z);
    double poly = 0;
    for (int i = ERFC.length - 1; i > 0; i--) poly = (poly + ERFC[i]) * u;
    double erfc = u * StrictMath.exp(-z * z + ERFC[0] + poly);
    return 0.5 * (x >= 0 ? erfc : 2 - erfc);
  }

  /** ガウス・エルミートの点と重み（重み関数 e^{−x²}、ニュートン法）。 */
  private static void hermite(double[] x, double[] w) {
    int n = x.length;
    double z = 0;
    for (int i = 0; i < (n + 1) / 2; i++) {
      if (i == 0) z = StrictMath.sqrt(2.0 * n + 1) - 1.85575 * StrictMath.pow(2.0 * n + 1, -0.16667);
      else if (i == 1) z -= 1.14 * StrictMath.pow(n, 0.426) / z;
      else if (i == 2) z = 1.86 * z - 0.86 * x[0];
      else if (i == 3) z = 1.91 * z - 0.91 * x[1];
      else z = 2 * z - x[i - 2];
      double derivative = 0;
      for (int iteration = 0; iteration < 100; iteration++) {
        double p1 = StrictMath.pow(Math.PI, -0.25);
        double p2 = 0;
        for (int j = 0; j < n; j++) {
          double p3 = p2;
          p2 = p1;
          p1 = z * StrictMath.sqrt(2.0 / (j + 1)) * p2 - StrictMath.sqrt((double) j / (j + 1)) * p3;
        }
        derivative = StrictMath.sqrt(2.0 * n) * p2;
        double previous = z;
        z = previous - p1 / derivative;
        if (StrictMath.abs(z - previous) <= 1e-14) break;
      }
      x[i] = z;
      x[n - 1 - i] = -z;
      w[i] = 2 / (derivative * derivative);
      w[n - 1 - i] = w[i];
    }
  }
}
