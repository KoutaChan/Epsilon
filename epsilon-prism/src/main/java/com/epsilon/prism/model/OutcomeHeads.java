package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.model.WidthPlan.Axis;
import com.epsilon.prism.model.WidthPlan.Role;

/**
 * 候補の結果の頭（学習のときだけ計算し、推論の重みの束に入れない）。
 *
 * <pre>
 * u_a = SiLU(W_o·[N_o(e_a); h] + b_o)      2H → 128
 * 出力 = W·u_a + b                          FP32、{@link CandidateTarget#WIDTH} 列（区画は CandidateTarget の順）
 * </pre>
 *
 * <p>e_a は価値グラフと同じ手の表現（集約の重みは sg）なので、勾配は e_a・z・h を通って候補層と背骨に返るだけで、B と門の重みには
 * 届かない。s は出力を読まない。
 */
final class OutcomeHeads extends Module {

  static final int HIDDEN = 128;

  private final double epsilon;
  private final Parameter norm;
  private final Parameter hidden;
  private final Parameter hiddenBias;
  private final Parameter output;
  private final Parameter outputBias;

  /**
   * パラメーターを登録する。
   *
   * @param shape 網の形
   */
  OutcomeHeads(PrismShape shape) {
    epsilon = shape.epsilon();
    norm = gain("norm", shape.hidden());
    hidden =
        weight(
            "hidden",
            new Shape(HIDDEN, 2L * shape.hidden()),
            Axis.of(Role.FIXED),
            Axis.of(Role.STREAM_IN, 2));
    hiddenBias = bias("hiddenBias", HIDDEN, Axis.of(Role.FIXED));
    output =
        weight("output", new Shape(CandidateTarget.WIDTH, HIDDEN), Axis.of(Role.FIXED), Axis.of(Role.FIXED));
    outputBias = bias("outputBias", CandidateTarget.WIDTH, Axis.of(Role.FIXED));
  }

  /**
   * 候補の頭の出力を計算する。
   *
   * @param context 文脈
   * @param chosen e_a [候補, H]（FP32）
   * @param states 候補の行の h [候補, H]（計算の型）
   * @return 出力 [候補, CandidateTarget.WIDTH]（FP32）
   */
  NDArray forward(Context context, NDArray chosen, NDArray states) {
    NDArray input = NDArrays.concat(new NDList(context.norm(chosen, norm, epsilon), states), 1);
    NDArray units = Context.silu(context.linear(input, hidden, hiddenBias));
    return context.linear(context.toFloat(units), context.fp32(output), context.fp32(outputBias));
  }
}
