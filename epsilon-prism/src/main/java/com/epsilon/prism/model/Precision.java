package com.epsilon.prism.model;

import ai.djl.ndarray.types.DataType;

/**
 * eager の網の計算の型。
 *
 * <p>BFLOAT16 では、Linear・RMSNorm の出力・残差・SiLU・注意の出力を BF16 に 1 回ずつ丸め、注意の中（q·k・偏り・softmax・p·v）・候補の
 * 交差注意・集約・頭は FP32 で計算する。HIP は同じ所で丸める。FLOAT32 は全てを FP32 で計算する（CPU の Reviewer と試験）。
 *
 * <p>丸める所は網が型の変換で明示するので、autocast は開かずに呼ぶ。
 */
public enum Precision {
  /** 全て FP32。 */
  FLOAT32(DataType.FLOAT32),
  /** 決まった所（上の説明）で BF16 に丸める。 */
  BFLOAT16(DataType.BFLOAT16);

  private final DataType compute;

  Precision(DataType compute) {
    this.compute = compute;
  }

  /**
   * 流れ・Linear の出力の型を返す。
   *
   * @return 計算の型
   */
  public DataType compute() {
    return compute;
  }
}
