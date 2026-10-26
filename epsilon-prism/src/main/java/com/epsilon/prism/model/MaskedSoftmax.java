package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

/**
 * マスクの上の softmax（{@link PrismLoss}・{@link Distill} が使う）。全て FP32 の [行, 容量]。
 *
 * <p>マスクの外の枠は掛け算でなく where で外すので、そこの出力が何でも（NaN や Inf でも）値にも勾配にも混ざらない。安定化に引く行の最大値は勾配を
 * 切る（softmax は行の定数で変わらないので勾配は同じ）。
 *
 * @param pi マスクの上の確率（外は 0）
 * @param logPi マスクの上の log 確率（外は 0）
 * @param logSumExp 行の log Σ_マスク exp(z)。[行, 1]（マスクが空の行は 0）
 */
record MaskedSoftmax(NDArray pi, NDArray logPi, NDArray logSumExp) {
  /** where で外す枠の値（exp で 0 になる。FP32 で有限）。 */
  private static final float FLOOR = -1e30f;

  private static final int[] ACTIONS = {1};

  /** マスクを真偽値にする（真偽値でなければ正の枠を真とする）。 */
  static NDArray bool(NDArray mask) {
    return mask.getDataType() == DataType.BOOLEAN ? mask : mask.gt(0);
  }

  /**
   * z のマスクの上の softmax。
   *
   * @param scope 作った配列を付ける管理元
   * @param z ロジット。[行, 容量]、FP32
   * @param mask 真偽値の [行, 容量]
   */
  static MaskedSoftmax of(NDManager scope, NDArray z, NDArray mask) {
    Shape shape = z.getShape();
    NDArray maskF = mask.toType(DataType.FLOAT32, false);
    // マスクが空でない行で 1、空の行で 0。[行, 1]
    NDArray nonEmpty = maskF.sum(ACTIONS, true).minimum(1f);
    NDArray zeros = scope.zeros(shape, DataType.FLOAT32);
    // 行の最大値（勾配を切る）。空の行は 0 にして値を有限に保つ
    NDArray peak =
        NDArrays.where(mask, z, scope.full(shape, FLOOR))
            .max(ACTIONS, true)
            .stopGradient()
            .mul(nonEmpty);
    // マスクの上は z − 最大値（≤ 0）、外は 0（exp は 1 だが maskF で消す）。外の z には勾配が流れない
    NDArray shifted = NDArrays.where(mask, z.sub(peak), zeros);
    NDArray e = shifted.exp().mul(maskF);
    // マスクが空でない行は和 ≥ 1（最大の手が exp(0) = 1）。空の行は 1 を足して log を有限にする
    NDArray sum = e.sum(ACTIONS, true).add(nonEmpty.neg().add(1f));
    NDArray logSum = sum.log();
    return new MaskedSoftmax(
        e.div(sum), shifted.sub(logSum).mul(maskF), logSum.add(peak).mul(nonEmpty));
  }
}
