package com.epsilon.nano.ai.decision.input;

import ai.djl.ndarray.NDArray;

/**
 * CPU 側で実在する遷移だけを集めた推論入力と、その位置情報をデバイス上に保持する。
 *
 * <p>2本の連続バッファがすべてのビューの参照先を保持する。学習用の教師値と、未使用位置を含む固定長の遷移入力は持たない。
 */
public record DecisionInferenceDeviceBatch(
    NDArray categoricalSlab,
    NDArray numericSlab,
    DecisionInferenceInputs inputs,
    NDArray playerMemoryPresentIndices,
    NDArray transitionPresentIndices,
    PolicyExecutionIndices policyExecutionIndices) {

  public int rowCount() {
    return inputs.rowCount();
  }

  public DecisionBucket bucket() {
    return inputs.bucket();
  }

  public NDArray legalActionMask() {
    return inputs.legalActionMask();
  }

  /** ホストから転送した二本の元の連続バッファのバイト数。数値のデバイス変換精度には依存しない。 */
  public long inputByteCount() {
    return Math.addExact(
        Math.multiplyExact(categoricalSlab.getShape().size(), Short.BYTES),
        Math.multiplyExact(numericSlab.getShape().size(), Float.BYTES));
  }

  /**
   * GPU推論の疎な方策分岐を、密なバッチ上の位置へ戻せる形で保持する。
   *
   * <p>先頭のRIICHI インデックスだけは{@code [row, action]}を平坦化した位置、残りはバッチ行である。全インデックスを一つのデバイス
   * テンソルへまとめ、分岐ごとの小さなホストからデバイスへの転送を発生させない。
   */
  public record PolicyExecutionIndices(
      NDArray packedIndices,
      int riichiActionCount,
      int callRowCount,
      int ronRowCount,
      int kanRowCount,
      int kyushuRowCount,
      int tsumoRowCount) {

    public NDArray riichiActions() {
      return slice(0, riichiActionCount);
    }

    public NDArray callRows() {
      return slice(riichiActionCount, callRowCount);
    }

    public NDArray ronRows() {
      return slice(riichiActionCount + callRowCount, ronRowCount);
    }

    public NDArray kanRows() {
      return slice(riichiActionCount + callRowCount + ronRowCount, kanRowCount);
    }

    public NDArray kyushuRows() {
      return slice(riichiActionCount + callRowCount + ronRowCount + kanRowCount, kyushuRowCount);
    }

    public NDArray tsumoRows() {
      return slice(
          riichiActionCount + callRowCount + ronRowCount + kanRowCount + kyushuRowCount,
          tsumoRowCount);
    }

    private NDArray slice(int offset, int count) {
      return packedIndices.get("{}:{}", offset, offset + count);
    }
  }
}
