package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;

/**
 * ネットワークの出力。
 *
 * @param scores s [行, 手の容量]（FP32）。列 k は行の k 番目の手で、合法手でない列と PASS の列は 0
 * @param legal 合法手かどうか [行, 手の容量]（BOOLEAN）
 * @param actions 行ごとの手の数（ホストに置く）
 * @param discards B が値を出す打牌の手かどうか [行 × 手の容量]（ホストに置く。行の平均のアンカー {@link PrismLoss#rowMean} が読む）
 * @param auxiliary 補助のヘッドの出力（補助のヘッドを計算しない順伝播では null）
 */
public record Outputs(NDArray scores, NDArray legal, int[] actions, boolean[] discards, Auxiliary auxiliary) {

  /**
   * 補助のヘッドの出力（FP32 のロジットと値）。
   *
   * @param candidates 候補のヘッド [N_c, TargetSchema.candidateWidth]（候補がなければ null）
   * @param reading 読みのヘッド [N_r, 3, TargetSchema.readingWidth]（読みの行がなければ null）
   */
  public record Auxiliary(NDArray candidates, NDArray reading) {}
}
