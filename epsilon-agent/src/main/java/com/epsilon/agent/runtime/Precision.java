package com.epsilon.agent.runtime;

import ai.djl.ndarray.types.DataType;

/**
 * ネットワークの計算の型。どこで丸めるかはネットワークが型の変換で明示するので、autocast は使わない。FP32 は CPU の Reviewer と
 * テストが使い、BF16 は GPU での収集・学習・評価の卓が使う。
 */
public enum Precision {
  FLOAT32(DataType.FLOAT32),
  /** 残差ストリームと線形の層の出力を BF16 に丸める。どこを FP32 で計算するかは、系列ごとのネットワークが決める。 */
  BFLOAT16(DataType.BFLOAT16);

  private final DataType compute;

  Precision(DataType compute) {
    this.compute = compute;
  }

  /** 残差ストリームと線形の層の出力に使う型。 */
  public DataType compute() {
    return compute;
  }
}
