package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.util.Objects;

/**
 * 価値の効用を、等しい幅の 101 個のビンの分布で表す（HL-Gauss）。教師の分布は、効用を平均とする正規分布をビンごとに積分し、和が 1 に
 * なるよう割ったもの。標準偏差はビンの幅の 0.75 倍。ビンが覆う範囲は、順位の効用の最小値から最大値までに、両側へ標準偏差の 6 倍の余白を
 * 足したもので、これを 101 等分する。分布から効用へは、ビンの中心を確率で重み付けた平均で戻す。
 */
public final class HlGauss {

  public static final int BINS = 101;

  /** 事前分布の確率の下限。対数を取る前に、これより小さい確率をこの値にする。教師の分布には使わない。 */
  public static final float PRIOR_FLOOR = 1e-8f;

  private static final double SIGMA_PER_BIN = 0.75;
  private static final double PADDING_SIGMAS = 6;

  private final double sigma;
  private final float[] edges = new float[BINS + 1];
  private final float[] centers = new float[BINS];

  /** ビンの端 [1, 102]（FP32）。{@link #initialize} で作る。 */
  private NDArray edgeRow;

  /**
   * @param variant ルール（席の数だけ順位の効用がある）
   * @param utility 順位の効用
   */
  public HlGauss(Variant variant, RankUtility utility) {
    Objects.requireNonNull(utility, "utility");
    double low = Double.POSITIVE_INFINITY;
    double high = Double.NEGATIVE_INFINITY;
    for (int rank = 0; rank < variant.seats(); rank++) {
      low = Math.min(low, utility.value(variant, rank));
      high = Math.max(high, utility.value(variant, rank));
    }
    double width = (high - low) / (BINS - 2 * PADDING_SIGMAS * SIGMA_PER_BIN);
    sigma = SIGMA_PER_BIN * width;
    double minimum = low - PADDING_SIGMAS * sigma;
    for (int edge = 0; edge <= BINS; edge++) {
      edges[edge] = (float) (minimum + edge * width);
      if (edge < BINS) {
        centers[edge] = (float) (minimum + (edge + 0.5) * width);
      }
    }
  }

  /**
   * 教師の分布の計算に使うビンの端を、マネージャーのデバイスに 1 回だけ作る。{@link #targets} の前に呼ぶ。
   *
   * @param manager 端を持つマネージャー（デバイスもこのマネージャーのもの）
   */
  public void initialize(NDManager manager) {
    edgeRow = manager.create(edges, new Shape(1, BINS + 1));
  }

  /**
   * 効用から教師の分布を作る。
   *
   * @param utilities 効用 [行]（{@link #initialize} に渡したマネージャーと同じデバイス）
   * @return ビンの確率 [行, 101]（FP32、勾配は流さない）
   */
  public NDArray targets(NDArray utilities) {
    NDArray targets = utilities.toType(DataType.FLOAT32, false).stopGradient().reshape(-1, 1);
    NDArray cumulative = targets.sub(edgeRow).neg().div((float) (Math.sqrt(2.0) * sigma)).erf();
    NDArray mass = cumulative.get(":, 1:").sub(cumulative.get(":, 0:{}", BINS)).maximum(0f);
    return mass.div(mass.sum(new int[] {1}, true)).stopGradient();
  }

  /**
   * ホストに読み出したロジットの 1 行を、効用に戻す。
   *
   * @param logits ロジットの配列
   * @param offset 行の先頭の位置
   * @return 効用
   */
  public float decode(float[] logits, int offset) {
    double maximum = Double.NEGATIVE_INFINITY;
    for (int bin = 0; bin < BINS; bin++) {
      maximum = Math.max(maximum, logits[offset + bin]);
    }
    double sum = 0;
    double weighted = 0;
    for (int bin = 0; bin < BINS; bin++) {
      double probability = Math.exp(logits[offset + bin] - maximum);
      sum += probability;
      weighted += probability * centers[bin];
    }
    return (float) (weighted / sum);
  }
}
