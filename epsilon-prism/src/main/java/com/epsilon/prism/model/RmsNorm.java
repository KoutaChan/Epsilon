package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Parameter;
import com.epsilon.prism.hip.HipFfn;
import com.epsilon.prism.model.EncoderLayer.FixedTokens;
import com.epsilon.prism.model.EncoderLayer.SharedTokens;
import com.epsilon.prism.model.EncoderLayer.TypedTokens;
import com.epsilon.prism.model.FitProbe.Stage;

/**
 * バックボーンの最後の RMSNorm（N_final）。ほかの部品の RMSNorm のゲインはその部品が持ち、計算は {@link Context#norm} が同じ式で行う
 * （統計とゲインは FP32、出力は計算に使う型、ε は {@link PrismShape#epsilon()}）。バックボーンの正規化なので、Attention の記録があれば
 * 記録用の演算で計算する（{@link Context#backboneNorm}）。
 *
 * <p>最後の層が型ごとの FFN を持つ形（{@link PrismShape#layers()}）では、局・席のトークンと牌のトークンがそれぞれのゲインを持つ。両者を
 * 別々に正規化してから並べる。
 */
public final class RmsNorm extends Module {

  /** 全てのトークンのゲイン（型ごとのゲインなら null）。 */
  private final Parameter gain;

  /** 局・席のトークンのゲイン（型ごとのゲインでなければ null）。 */
  private final Parameter contextGain;

  /** 牌のトークンのゲイン（型ごとのゲインでなければ null）。 */
  private final Parameter tileGain;

  private final double epsilon;

  /**
   * ゲインを登録する。
   *
   * @param shape ネットワークの形
   */
  public RmsNorm(PrismShape shape) {
    if (shape.layers().getLast().ffn() instanceof HipFfn.Typed) {
      gain = null;
      contextGain = gain("contextGain", shape.hidden());
      tileGain = gain("tileGain", shape.hidden());
    } else {
      gain = gain("gain", shape.hidden());
      contextGain = null;
      tileGain = null;
    }
    epsilon = shape.epsilon();
  }

  /**
   * 1 つのゲインで正規化する（全てのトークンでゲインを共有するバックボーン）。入力は最後の層の出力で、この正規化のほかから読まれては
   * ならない（{@link Context#backboneNorm}）。型ごとのゲインを持つときは {@link #apply(Context, FixedTokens, int)} を使う。
   *
   * @param context 文脈
   * @param input 入力 [..., H]
   * @param lastLayer 最後の層の番号（記録の逆伝播は、この層の O 射影と FFN の段に入れて計る）
   * @return 正規化した値
   * @throws IllegalStateException 型ごとのゲインを持つとき
   */
  public NDArray apply(Context context, NDArray input, int lastLayer) {
    if (gain == null) {
      throw new IllegalStateException("the final RMSNorm has typed gains; pass the last layer's FixedTokens");
    }
    return context.backboneNorm(input, gain, epsilon, Stage.O_FFN_BACKWARD, lastLayer);
  }

  /**
   * バックボーンの最後の層の出力を正規化する。その出力は、この正規化のほかから読まれてはならない（{@link Context#backboneNorm}）。
   *
   * @param context 文脈
   * @param tokens バックボーンの最後の層の出力
   * @param lastLayer バックボーンの最後の層の番号（記録の逆伝播は、この層の O 射影と FFN の段に入れて計る）
   * @return 正規化した局・席・牌のトークン [行, 局・席・牌, H]
   * @throws IllegalArgumentException 出力の形がゲインの持ち方と合わないとき
   */
  NDArray apply(Context context, FixedTokens tokens, int lastLayer) {
    return switch (tokens) {
      case SharedTokens shared when gain != null -> apply(context, shared.tokens(), lastLayer);
      case TypedTokens typed when gain == null -> typed(context, typed, lastLayer);
      default ->
          throw new IllegalArgumentException(
              "the final RMSNorm "
                  + (gain == null ? "has typed gains" : "has one gain")
                  + " but the last layer gave "
                  + tokens.getClass().getSimpleName());
    };
  }

  /** 局・席のトークンと牌のトークンを、それぞれのゲインで正規化して並べる。 */
  private NDArray typed(Context context, TypedTokens tokens, int lastLayer) {
    NDArray seats =
        context.backboneNorm(tokens.context(), contextGain, epsilon, Stage.O_FFN_BACKWARD, lastLayer);
    NDArray tiles =
        context.backboneNorm(tokens.tiles(), tileGain, epsilon, Stage.O_FFN_BACKWARD, lastLayer);
    return NDArrays.concat(new NDList(seats, tiles), 1);
  }
}
