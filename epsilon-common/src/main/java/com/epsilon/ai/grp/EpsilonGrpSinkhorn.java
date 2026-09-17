package com.epsilon.ai.grp;

import ai.djl.ndarray.NDArray;

/** 4行4列のロジットを、行和と列和が1になる席別の順位確率へ正規化する。計算は対数空間で行う。 */
public final class EpsilonGrpSinkhorn {

  /** 行・列正規化を交互に適用する回数。 */
  public static final int ITERATIONS = 32;

  /** 行と列の平均成分を除いたロジットに適用する、滑らかなクリッピングの幅。 */
  public static final float INTERACTION_CLIP = 2.0f;

  private EpsilonGrpSinkhorn() {}

  /**
   * {@code [B,16]} ロジットを {@code [B,4,4]} の周辺確率の対数に正規化する。
   *
   * @param flatLogits 席ごとにまとめた未正規化ロジット
   * @return 行和・列和が1になる 4x4 周辺確率の自然対数
   */
  public static NDArray logMarginals(NDArray flatLogits) {
    long batch = requireFlatBatch(flatLogits);
    return normalizeLogWeights(boundedInteraction(flatLogits.reshape(batch, 4, 4)));
  }

  /**
   * 正の4行4列の事前分布に残差を加えた対数重みを、追加のクリッピングなしで正規化する。
   *
   * <p>{@link
   * #logMarginals(NDArray)}は行・列の平均成分を除き、残った成分の大きさを制限する。このメソッドはGRPの順位確率を事前分布として使う経路向けで、残差が0のときに元の事前分布を保つため、その変形を行わない。
   *
   * @param flatLogWeights 席ごとにまとめた対数重み。形状は{@code [B,16]}
   * @return 行和・列和が1になる4x4 周辺確率の自然対数
   */
  public static NDArray logMarginalsFromLogWeights(NDArray flatLogWeights) {
    long batch = requireFlatBatch(flatLogWeights);
    return normalizeLogWeights(flatLogWeights.reshape(batch, 4, 4));
  }

  private static NDArray normalizeLogWeights(NDArray logWeights) {
    NDArray logMarginals = logWeights;
    for (int iteration = 0; iteration < ITERATIONS; iteration++) {
      logMarginals = logMarginals.logSoftmax(2);
      logMarginals = logMarginals.logSoftmax(1);
    }
    // 席別の周辺確率は後続処理で確率として直接使うため、まず各行を厳密に正規化する。
    NDArray probabilities = logMarginals.logSoftmax(2).exp();
    // 行和=1なら全要素和=4。各列の不足分を4行へ等分すると、行和を保ったまま
    // 列和も厳密に1へ補正できる。滑らかなクリッピングと32回反復により補正前残差は十分小さく、
    // 正の要素を保つ。
    NDArray columnCorrection =
        probabilities.sum(new int[] {1}, true).neg().add(1.0f).div(EpsilonGrpRanks.SEAT_COUNT);
    return probabilities.add(columnCorrection).log();
  }

  /**
   * {@code [B,16]} ロジットを同形状の席ごとにまとめた周辺確率にする。
   *
   * @param flatLogits 席ごとにまとめた未正規化ロジット
   * @return 行和と列和が1になる席別の順位確率
   */
  public static NDArray marginals(NDArray flatLogits) {
    long batch = requireFlatBatch(flatLogits);
    return logMarginals(flatLogits).exp().reshape(batch, EpsilonGrpRanks.MATRIX_SIZE);
  }

  /**
   * 一つの入力に対応する16個のロジットを CPU 上で席別の順位周辺確率へ変換する。
   *
   * <p>テンソル実装と同じ滑らかなクリッピング、反復回数、最終列補正を使い、テストと軽量な後処理で利用する。
   *
   * @param flatLogits 席ごとにまとめた16 ロジット
   * @return 行和と列和が1になる16個の確率
   */
  public static float[] marginals(float[] flatLogits) {
    if (flatLogits == null || flatLogits.length != EpsilonGrpRanks.MATRIX_SIZE) {
      throw new IllegalArgumentException("GRP logits size must be 16");
    }
    return marginalsBatch(flatLogits);
  }

  /**
   * 連続した16 ロジットのバッチをCPUで正規化する。作業領域は全行で再利用する。
   *
   * @param flatLogits 席ごとにまとめたロジットをバッチ順に連結した配列
   * @return 入力と同じ配置の周辺確率
   */
  static float[] marginalsBatch(float[] flatLogits) {
    if (flatLogits == null || flatLogits.length % EpsilonGrpRanks.MATRIX_SIZE != 0) {
      throw new IllegalArgumentException("GRP logits batch size must be a multiple of 16");
    }
    float[] out = new float[flatLogits.length];
    double[] logMarginals = new double[EpsilonGrpRanks.MATRIX_SIZE];
    double[] rowMeans = new double[4];
    double[] columnMeans = new double[4];
    for (int offset = 0; offset < flatLogits.length; offset += EpsilonGrpRanks.MATRIX_SIZE) {
      double globalMean = 0.0;
      for (int i = 0; i < 4; i++) {
        rowMeans[i] = 0.0;
        columnMeans[i] = 0.0;
      }
      for (int seat = 0; seat < 4; seat++) {
        for (int rank = 0; rank < 4; rank++) {
          int index = EpsilonGrpRanks.index(seat, rank);
          double value = flatLogits[offset + index];
          if (!Double.isFinite(value)) {
            throw new IllegalArgumentException("GRP logits must be finite");
          }
          logMarginals[index] = value;
          rowMeans[seat] += value / 4.0;
          columnMeans[rank] += value / 4.0;
          globalMean += value / 16.0;
        }
      }
      for (int seat = 0; seat < 4; seat++) {
        for (int rank = 0; rank < 4; rank++) {
          int index = EpsilonGrpRanks.index(seat, rank);
          double interaction =
              logMarginals[index] - rowMeans[seat] - columnMeans[rank] + globalMean;
          logMarginals[index] = INTERACTION_CLIP * Math.tanh(interaction / INTERACTION_CLIP);
        }
      }
      for (int iteration = 0; iteration < ITERATIONS; iteration++) {
        for (int seat = 0; seat < 4; seat++) {
          normalizeLogGroup(logMarginals, seat * 4, 1);
        }
        for (int rank = 0; rank < 4; rank++) {
          normalizeLogGroup(logMarginals, rank, 4);
        }
      }
      for (int seat = 0; seat < 4; seat++) {
        normalizeLogGroup(logMarginals, seat * 4, 1);
      }
      // 最後の指数関数の計算は同じ配列上で行い、列平均用の作業領域を列和の計算に再利用する。
      for (int rank = 0; rank < 4; rank++) columnMeans[rank] = 0.0;
      for (int seat = 0; seat < 4; seat++) {
        for (int rank = 0; rank < 4; rank++) {
          int index = EpsilonGrpRanks.index(seat, rank);
          logMarginals[index] = Math.exp(logMarginals[index]);
          columnMeans[rank] += logMarginals[index];
        }
      }
      for (int seat = 0; seat < 4; seat++) {
        for (int rank = 0; rank < 4; rank++) {
          int index = EpsilonGrpRanks.index(seat, rank);
          double corrected = logMarginals[index] + (1.0 - columnMeans[rank]) / 4.0;
          if (!(corrected > 0.0) || !Double.isFinite(corrected)) {
            throw new IllegalStateException("GRP Sinkhorn marginal repair became non-positive");
          }
          out[offset + index] = (float) corrected;
        }
      }
    }
    return out;
  }

  private static void normalizeLogGroup(double[] values, int offset, int stride) {
    double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < 4; i++) max = Math.max(max, values[offset + i * stride]);
    double sum = 0.0;
    for (int i = 0; i < 4; i++) sum += Math.exp(values[offset + i * stride] - max);
    double normalizer = max + Math.log(sum);
    for (int i = 0; i < 4; i++) values[offset + i * stride] -= normalizer;
  }

  private static NDArray boundedInteraction(NDArray logits) {
    NDArray rowMean = logits.sum(new int[] {2}, true).div(4.0f);
    NDArray columnMean = logits.sum(new int[] {1}, true).div(4.0f);
    NDArray globalMean = logits.sum(new int[] {1, 2}, true).div(16.0f);
    return logits
        .sub(rowMean)
        .sub(columnMean)
        .add(globalMean)
        .div(INTERACTION_CLIP)
        .tanh()
        .mul(INTERACTION_CLIP);
  }

  private static long requireFlatBatch(NDArray flatLogits) {
    if (flatLogits == null
        || flatLogits.getShape().dimension() != 2
        || flatLogits.getShape().get(1) != EpsilonGrpRanks.MATRIX_SIZE) {
      throw new IllegalArgumentException("GRP logits must have shape [B,16]");
    }
    return flatLogits.getShape().get(0);
  }
}
