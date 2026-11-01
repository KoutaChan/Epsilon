package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.nn.Parameter;
import com.epsilon.prism.model.FitProbe.Stage;

/**
 * 背骨の最後の RMSNorm（N_final）。ほかの部品の RMSNorm の利得はその部品が持ち、計算は {@link Context#norm} が同じ式で行う（統計・
 * 利得は FP32、出力は計算の型、ε は {@link PrismShape#epsilon()}）。背骨の正規化なので、注意の記録があれば記録の演算で計算する
 * （{@link Context#backboneNorm}）。
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
   * 正規化する。input は背骨の最後の層の出力で、この正規化のほかに読まれないこと（{@link Context#backboneNorm}）。
   *
   * @param context 文脈
   * @param input 入力 [..., H]
   * @param lastLayer 背骨の最後の層の番号（記録の逆伝播はその層の O 射影と FFN の段で計る）
   * @return 正規化した値
   */
  public NDArray apply(Context context, NDArray input, int lastLayer) {
    return context.backboneNorm(input, gain, epsilon, Stage.O_FFN_BACKWARD, lastLayer);
  }
}
