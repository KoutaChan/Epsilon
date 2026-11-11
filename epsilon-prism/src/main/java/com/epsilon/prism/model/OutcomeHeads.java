package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;

/**
 * 候補の結果を当てる補助のヘッド。学習のときだけ計算し、推論で使う重みには入れない。
 *
 * <pre>
 * u_a = SiLU(W_o·[N_o(e_a); h] + b_o)      2H → 128
 * 出力 = W·u_a + b                          FP32、{@link TargetSchema#candidateWidth} 列（領域は CandidateTarget の順）
 * </pre>
 *
 * <p>e_a は価値グラフと同じ手の表現で、集約の重みは勾配を止めている（sg）。そのため勾配は e_a・z・h を通って候補層とバックボーンに戻る
 * だけで、B とゲートの重みには届かない。s はこのヘッドの出力を読まない。
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
   * @param shape ネットワークの形
   * @param heads 補助のヘッドの形
   */
  OutcomeHeads(PrismShape shape, TargetSchema heads) {
    epsilon = shape.epsilon();
    norm = gain("norm", shape.hidden());
    hidden = weight("hidden", new Shape(HIDDEN, 2L * shape.hidden()));
    hiddenBias = bias("hiddenBias", HIDDEN);
    output = weight("output", new Shape(heads.candidateWidth(), HIDDEN));
    outputBias = bias("outputBias", heads.candidateWidth());
  }

  /**
   * 候補のヘッドの出力を計算する。
   *
   * @param context 順伝播の文脈
   * @param chosen e_a [候補, H]（FP32）
   * @param states 候補の行の h [候補, H]（計算の型）
   * @return 出力 [候補, TargetSchema.candidateWidth]（FP32）
   */
  NDArray forward(Context context, NDArray chosen, NDArray states) {
    NDArray input = NDArrays.concat(new NDList(context.norm(chosen, norm, epsilon), states), 1);
    NDArray units = Context.silu(context.linear(input, hidden, hiddenBias));
    return context.linear(context.toFloat(units), context.fp32(output), context.fp32(outputBias));
  }
}
