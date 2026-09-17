package com.epsilon.ai.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;

/** マスクが有効な行だけを取り出して処理し、元の配列配置へ戻すためのテンソル演算。 */
public final class EpsilonMaskedRows {

  private EpsilonMaskedRows() {}

  /**
   * マスクを一次元に並べ、有効な要素のインデックスを返す。
   *
   * @param mask 正値を有効とみなす任意形状のマスク
   * @return 一次元化後の有効な行のインデックス
   */
  public static NDArray indices(NDArray mask) {
    return mask.reshape(-1).gt(0.0f).nonzero().reshape(-1).stopGradient();
  }

  /**
   * 先頭軸から指定行を順序どおり集める。
   *
   * <p>計算ライブラリの行選択演算を使うため、行インデックスを末尾形状へ展開しない。重複インデックスの勾配は通常の行選択と同様に加算される。
   *
   * @param rows {@code [N,...]} の入力テンソル
   * @param rowIndices 集める行インデックス
   * @return {@code [P,...]} の抽出した行テンソル
   */
  public static NDArray gather(NDArray rows, NDArray rowIndices) {
    return NDArrays.gatherRows(rows, rowIndices);
  }

  /**
   * {@code rows=[P,W]} を指定インデックスへ配置し、未指定行を0にする。
   *
   * @param rows 抽出した行テンソル
   * @param rowIndices 各入力行の出力インデックス
   * @param rowCount 出力の総行数
   * @return {@code [rowCount,W]} の出力テンソル
   */
  public static NDArray scatter(NDArray rows, NDArray rowIndices, long rowCount) {
    return NDArrays.scatterRows(rows, rowIndices, rowCount);
  }
}
