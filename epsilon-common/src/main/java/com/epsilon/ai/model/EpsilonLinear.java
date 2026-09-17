package com.epsilon.ai.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/** 線形層の既存パラメーターを使い、メモリ上で連続していない入力領域での丸め方を保って射影を計算する。 */
public final class EpsilonLinear {
  private EpsilonLinear() {}

  /**
   * 連続化した {@code [batch,tokens,width]} を、元のバッチ間に隙間を持つ切り出した領域と同じ丸め境界で射影する。
   *
   * <p>GPU凍結推論の複数バッチだけ積を丸めてからバイアスを加える。単バッチの切り出した領域は元から連続なので、学習・CPUと同様に既存ブロックの順伝播を使う。パラメーターは借用し、必要なバイアス変換だけを入力の順伝播
   * メモリ管理オブジェクトへ所有させる。
   */
  public static NDArray projectBatchSlices(
      Linear block,
      ParameterStore parameterStore,
      NDArray input,
      boolean training,
      PairList<String, Object> runtimeParameters) {
    if (training || !input.getDevice().isGpu() || input.getShape().get(0) == 1) {
      return block
          .forward(parameterStore, new NDList(input), training, runtimeParameters)
          .singletonOrThrow();
    }
    NDArray weight =
        parameterStore.getValue(
            block.getDirectParameters().get("weight"), input.getDevice(), false);
    NDArray bias =
        parameterStore.getValue(block.getDirectParameters().get("bias"), input.getDevice(), false);
    NDArray projected = Linear.linear(input, weight).singletonOrThrow();
    if (bias.getDataType() != projected.getDataType()) {
      bias = bias.toType(projected.getDataType(), false);
      bias.attach(input.getManager());
    }
    return projected.addi(bias);
  }
}
