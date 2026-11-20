package com.epsilon.prism.kernel;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import com.epsilon.prism.kernel.CollectionPlanOptions.SegmentLayout;

/**
 * 遷移ごとの牌の数え上げと区分付き候補 Attention を学習する。
 *
 * <p>遷移ごとの行の番号（数え上げの {@code rows}、候補 Attention の {@code groups}）は 0 以上、バッチの行数未満で、減らない並び（同じ行の
 * 遷移が続けて並ぶ）であること（{@link #requireGroupedRows}）。順伝播は遷移ごとに行の番号を読み、逆伝播は行ごとの遷移の区間をこの並びの
 * 二分探索で求める。並びが崩れていると、逆伝播で勾配が欠けるか別の行に入る。
 */
public interface CandidateKernels {
  /**
   * 遷移ごとの行の番号が、候補の演算の並びの約束を守るかを確かめる。
   *
   * @param transitionRows 遷移ごとの行の番号
   * @param rows バッチの行数
   * @throws IllegalArgumentException 行の番号が範囲の外にあるか、前の遷移の行の番号より小さいとき
   */
  static void requireGroupedRows(int[] transitionRows, int rows) {
    int previous = 0;
    for (int t = 0; t < transitionRows.length; t++) {
      int row = transitionRows[t];
      if (row < previous || row >= rows) {
        throw new IllegalArgumentException(
            String.format("Transition %d is in row %d after row %d of %d", t, row, previous, rows));
      }
      previous = row;
    }
  }

  /** 遷移順のクエリ、行別の記憶、補正表と各キーの区分コード。{@code groups} は遷移ごとの行の番号（並びはクラスの説明）。 */
  record CandidateInput(
      NDArray query,
      NDArray memory,
      NDArray base,
      NDArray weights,
      NDArray codes,
      NDArray groups,
      int tiles,
      int seats,
      SegmentLayout layout,
      float scale) {
    int[] dimensions() {
      int[] dimensions = new int[8 + SegmentLayout.INTS];
      dimensions[0] = Math.toIntExact(query.getShape().get(0));
      dimensions[1] = 1 + tiles + seats;
      dimensions[2] = tiles;
      dimensions[3] = seats;
      dimensions[4] = Math.toIntExact(base.getShape().get(0));
      dimensions[5] = Math.toIntExact(base.getShape().get(1));
      dimensions[6] = Math.toIntExact(memory.getShape().get(0));
      dimensions[7] = Math.toIntExact(query.getShape().get(1));
      layout.encode(dimensions, 8);
      return dimensions;
    }

    public Shape probabilities() {
      return new Shape(query.getShape().get(0), query.getShape().get(1), 1 + tiles + seats);
    }
  }

  /** ゲートと BF16 射影の積を牌番号順に FP32 の積和で足し、結果を BF16 に丸める。{@code rows} は遷移ごとの行の番号（並びはクラスの説明）。 */
  void count(NDArray projected, NDArray gates, NDArray rows, NDArray ids, NDArray output);

  /** 行ごとに遷移をたどり、牌の射影の BF16 勾配とゲートの FP32 勾配を書く。{@code rows} は順伝播と同じ並び。 */
  void countGradient(
      NDArray projected,
      NDArray gates,
      NDArray rows,
      NDArray ids,
      NDArray gradient,
      NDArray gradProjected,
      NDArray gradGate);

  /** 遷移・ヘッド・キーごとの確率を FP32 で保存する。 */
  void candidateAttention(CandidateInput input, NDArray output, NDArray probabilities);

  /** クエリ・記憶・補正表の勾配を求める。 */
  void candidateAttentionGradient(
      CandidateInput input,
      NDArray probabilities,
      NDArray upstream,
      NDArray gradQuery,
      NDArray gradMemory,
      NDArray gradBase,
      NDArray gradWeights);
}
