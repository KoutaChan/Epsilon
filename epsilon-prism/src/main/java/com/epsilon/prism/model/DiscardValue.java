package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.input.InputFormat;

/**
 * 打牌の値 B。HIP では K6 カーネルの最後で計算する。
 *
 * <pre>
 * u^B  = W_B·N_B(z_t) + c_B                         （計算の型で求めて FP32 に変える）
 * B(t) = Σ_g wB_g·silu(u^B_g) + Σ_k uBx_k·x^B_t,k   （FP32。x^B は B 用の厳密な入力の列）
 * </pre>
 *
 * <p>wB と uBx は FP32 の最後の層で、0 で初期化する。バイアス b_B は持たない。持っても、sg（勾配を止める演算）の下では
 * 損失に効かず、値が漂うだけになる。
 */
final class DiscardValue extends Module {

  private final double epsilon;
  private final Parameter norm;
  private final Parameter hidden;
  private final Parameter hiddenBias;
  private final Parameter output;
  private final Parameter exact;

  /**
   * パラメーターを登録する。
   *
   * @param shape ネットワークの形
   * @param format 入力の形式（B 用の厳密な列の数をここから読む）
   */
  DiscardValue(PrismShape shape, InputFormat format) {
    epsilon = shape.epsilon();
    norm = gain("norm", shape.hidden());
    hidden = weight("hidden", new Shape(shape.gate(), shape.hidden()));
    hiddenBias = bias("hiddenBias", shape.gate());
    output = zeros("output", new Shape(shape.gate()));
    int exacts = format.width(Tensor.DISCARD_EXACT);
    exact = zeros("exact", new Shape(exacts));
  }

  /**
   * B を計算する。
   *
   * @param context 計算の文脈
   * @param transitions 遷移の表現 z [遷移, H]（計算の型）
   * @param discardExact B 用の厳密な入力 [遷移, 列]（FP32）
   * @return B [遷移]（FP32）
   */
  NDArray forward(Context context, NDArray transitions, NDArray discardExact) {
    NDArray units =
        context.toFloat(
            context.linear(context.norm(transitions, norm, epsilon), hidden, hiddenBias));
    NDArray value = Context.silu(units).mul(context.fp32(output)).sum(new int[] {1});
    return value.add(discardExact.mul(context.fp32(exact)).sum(new int[] {1}));
  }
}
