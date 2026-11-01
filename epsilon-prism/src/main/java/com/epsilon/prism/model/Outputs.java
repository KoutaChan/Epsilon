package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;

/**
 * 網の出力。
 *
 * @param scores s [行, 手の容量]（FP32。列 k は行の k 番目の手、合法でない枠と PASS は 0）
 * @param legal 合法手の印 [行, 手の容量]（BOOLEAN）
 * @param actions 行ごとの手の数（ホスト）
 * @param discards 値を B が出す打牌の手の印 [行 × 手の容量]（ホスト。行の平均の錨 {@link PrismLoss#rowMean} が読む）
 * @param auxiliary 補助の頭の出力（補助の頭を計算しない順伝播では null）
 */
public record Outputs(NDArray scores, NDArray legal, int[] actions, boolean[] discards, Auxiliary auxiliary) {

  /**
   * 補助の頭の出力（FP32 のロジット・値）。
   *
   * @param candidates 候補の頭 [N_c, CandidateTarget.WIDTH]（候補がなければ null）
   * @param reading 読みの頭 [N_r, 3, ReadingTarget.WIDTH]（読みの行がなければ null）
   */
  public record Auxiliary(NDArray candidates, NDArray reading) {}
}
