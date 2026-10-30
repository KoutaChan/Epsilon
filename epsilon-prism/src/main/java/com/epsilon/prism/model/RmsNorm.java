package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.nn.Parameter;

/**
 * 背骨の最後の RMSNorm（N_final）。ほかの部品の RMSNorm の利得はその部品が持ち、計算は {@link Context#norm} が同じ式で行う（統計・
 * 利得は FP32、出力は計算の型、ε は {@link PrismShape#epsilon()}）。
 */
public final class RmsNorm extends Module {

  private final Parameter gain;
  private final double epsilon;

  /**
   * 利得を登録する。
   *
   * @param shape 網の形
   */
  public RmsNorm(PrismShape shape) {
    gain = gain("gain", shape.hidden());
    epsilon = shape.epsilon();
  }

  /**
   * 正規化する。
   *
   * @param context 文脈
   * @param input 入力 [..., H]
   * @return 正規化した値
   */
  public NDArray apply(Context context, NDArray input) {
    return context.norm(input, gain, epsilon);
  }
}
