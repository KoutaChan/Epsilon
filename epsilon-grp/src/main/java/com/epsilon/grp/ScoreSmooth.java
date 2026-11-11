package com.epsilon.grp;

import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.OptionalDouble;

/**
 * 点数からなめらかに求める順位の周辺確率。各席の最終の点が互いに独立な正規分布 N(局末の点, σ²) に従うとみなし（σ = 係数·√残りの
 * 局の数）、最終順位の分布を出す。点数だけで順位を決める（階段状の関数）と、どの候補を選んでも判断する席の順位が変わらない判断では、
 * ラベルの差がちょうど 0 になる。この分布は点差に応じてなめらかに動くので、そうした判断にも差が残る。
 *
 * <p>係数は、PRISM の四人打ちの自己対局で、局末から半荘の終わりまでに動いた点の標準偏差を測り、√残りの局の数で割ったもの。残りの局の
 * 数は次局から南場の最後の局までを数え、西入なら 1。三人打ちの係数は持たないので、三人打ちでは使えない。
 *
 * <p>席 i の順位の分布は、自分の最終の点 X についてガウス・エルミートの 20 点で積分して求める。X を決めたとき、他の席が X より上に
 * いる確率は席ごとに独立とする。StrictMath だけを使うので、同じ入力にはビット単位で同じ値を返す。分布は自分の点に対して単調に動く。
 */
public final class ScoreSmooth {

  /** 四人打ちの係数。残り 1 局あたりの、最終の点の標準偏差（点）。 */
  static final double SIGMA_PER_HAND_4P = 5140;

  /** ガウス・エルミートの点の数。 */
  static final int NODES = 20;

  /** 標準正規分布で積分するときの点 z_k = √2·x_k と、重み w_k/√π（和は 1）。 */
  private static final double[] Z = new double[NODES];

  private static final double[] W = new double[NODES];

  /** erfc を近似する多項式の係数（u = 1/(1 + |x|/2) の冪の順）。 */
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

  private ScoreSmooth() {}

  /**
   * そのルールで使う σ の係数。
   *
   * @return 残り 1 局あたりの、最終の点の標準偏差（点）。係数を持たないルール（三人打ち）では空
   */
  public static OptionalDouble sigmaPerHand(Variant variant) {
    return switch (variant) {
      case TENHOU_4P -> OptionalDouble.of(SIGMA_PER_HAND_4P);
      case TENHOU_3P -> OptionalDouble.empty();
    };
  }

  /** σ の係数を持たないルールでは IllegalArgumentException を投げる。 */
  static void requireMeasured(Variant variant) {
    if (sigmaPerHand(variant).isEmpty()) {
      throw new IllegalArgumentException("The score-smooth sigma is not measured for " + variant);
    }
  }

  /**
   * 残りの局の数。次局から南場の最後の局までを数え、西入なら 1。σ はこの数の平方根に比例するので、係数を測るときもこの数で割る。
   *
   * @param round 次局の番号（東 1 局が 0）
   */
  public static int remainingRounds(Variant variant, int round) {
    return Math.max(1, variant.lastRound() + 1 - round);
  }

  /**
   * 局末の卓から、絶対席ごとの最終順位の周辺確率を書く。
   *
   * @param out 出力先（長さ S²。並びは [絶対席 × S ＋ 順位]）
   */
  static void marginals(GrpState state, double[] out) {
    Variant variant = state.variant();
    requireMeasured(variant);
    int seats = variant.seats();
    double sigma = SIGMA_PER_HAND_4P * StrictMath.sqrt(remainingRounds(variant, state.round()));
    Arrays.fill(out, 0);
    double[] above = new double[seats - 1];
    double[] ranks = new double[seats];
    for (int seat = 0; seat < seats; seat++) {
      for (int k = 0; k < NODES; k++) {
        int other = 0;
        for (int rival = 0; rival < seats; rival++) {
          if (rival != seat) {
            above[other++] = normalCdf((state.score(rival) - state.score(seat)) / sigma - Z[k]);
          }
        }
        // 上にいる席の数の分布（独立なベルヌーイの和）
        Arrays.fill(ranks, 0);
        ranks[0] = 1;
        for (int j = 0; j < seats - 1; j++) {
          double p = above[j];
          for (int r = j + 1; r > 0; r--) {
            ranks[r] = ranks[r] * (1 - p) + ranks[r - 1] * p;
          }
          ranks[0] *= 1 - p;
        }
        for (int r = 0; r < seats; r++) {
          out[seats * seat + r] += W[k] * ranks[r];
        }
      }
    }
  }

  /** 標準正規分布の累積分布関数。erfc の Chebyshev 近似で、相対誤差は 1.2e-7。t に対して単調。 */
  static double normalCdf(double t) {
    double x = -t / StrictMath.sqrt(2);
    double z = Math.abs(x);
    double u = 1 / (1 + 0.5 * z);
    double poly = 0;
    for (int i = ERFC.length - 1; i > 0; i--) {
      poly = (poly + ERFC[i]) * u;
    }
    double erfc = u * StrictMath.exp(-z * z + ERFC[0] + poly);
    return 0.5 * (x >= 0 ? erfc : 2 - erfc);
  }

  /** ガウス・エルミートの点と重みを、ニュートン法で求める（重み関数は e^{−x²}）。 */
  private static void hermite(double[] x, double[] w) {
    int n = x.length;
    double z = 0;
    for (int i = 0; i < (n + 1) / 2; i++) {
      if (i == 0) {
        z = StrictMath.sqrt(2.0 * n + 1) - 1.85575 * StrictMath.pow(2.0 * n + 1, -0.16667);
      } else if (i == 1) {
        z -= 1.14 * StrictMath.pow(n, 0.426) / z;
      } else if (i == 2) {
        z = 1.86 * z - 0.86 * x[0];
      } else if (i == 3) {
        z = 1.91 * z - 0.91 * x[1];
      } else {
        z = 2 * z - x[i - 2];
      }
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
        if (StrictMath.abs(z - previous) <= 1e-14) {
          break;
        }
      }
      x[i] = z;
      x[n - 1 - i] = -z;
      w[i] = 2 / (derivative * derivative);
      w[n - 1 - i] = w[i];
    }
  }
}
