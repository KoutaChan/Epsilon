package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.prism.input.InputSchema.DiscardExact;
import com.epsilon.prism.model.WidthPlan.Axis;
import com.epsilon.prism.model.WidthPlan.Role;

/**
 * 打牌の値 B（HIP の K6 の最後）。
 *
 * <pre>
 * u^B  = W_B·N_B(z_t) + c_B                         （計算の型の出力を FP32 にする）
 * B(t) = Σ_g wB_g·silu(u^B_g) + Σ_k uBx_k·x^B_t,k   （FP32。x^B は B の道 DiscardExact）
 * </pre>
 *
 * <p>wB・uBx は 0 で始める FP32 の最後の層。b_B は持たない（sg の下では損失に効かないまま漂うため）。
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
   * @param shape 網の形
   */
  DiscardValue(PrismShape shape) {
    epsilon = shape.epsilon();
    norm = gain("norm", shape.hidden());
    hidden =
        weight(
            "hidden",
            new Shape(shape.gate(), shape.hidden()),
            Axis.of(Role.INNER_OUT),
            Axis.of(Role.STREAM_IN));
    hiddenBias = bias("hiddenBias", shape.gate(), Axis.of(Role.INNER_OUT));
    output = zeros("output", new Shape(shape.gate()), Axis.of(Role.INNER_IN));
    exact = zeros("exact", new Shape(DiscardExact.values().length), Axis.of(Role.FIXED));
  }

  /**
   * B を計算する。
   *
   * @param context 文脈
   * @param transitions 遷移の表現 z [遷移, H]（計算の型）
   * @param discardExact B の道 [遷移, 8]（FP32）
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
