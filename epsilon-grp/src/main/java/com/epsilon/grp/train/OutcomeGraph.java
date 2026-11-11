package com.epsilon.grp.train;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import com.epsilon.grp.HandOutcomes;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.OutcomeTensor;
import com.epsilon.grp.OutcomeTerm;
import java.util.SplittableRandom;
import java.util.function.IntFunction;

/**
 * OutcomeNetwork と同じ計算を、DJL（PyTorch、CPU、float32）で組んだもの。重みは OutcomeTensor の宣言の順に並び、W の形は
 * [入力][出力] にする。順伝播は OutcomeNetwork と同じ式で、ロジットから対数確率の項を HandOutcomes の項の順に作る（終わり方の
 * log softmax、打点の組ごとの log softmax、立直の log σ(±z)）。1 行の損失は周辺尤度 −log Σ_k exp(Σ_j 項[k][j]) で、k は
 * 記録を再現できる結果（OutcomeBatch に並べたもの）を全て取る。
 */
final class OutcomeGraph implements TrainingLoop.Graph<OutcomeBatch>, AutoCloseable {

  private static final OutcomeTensor[] TENSORS = OutcomeTensor.values();
  private static final int[] OUTCOMES_AXIS = {1};
  private static final int[] TERMS_AXIS = {2};

  private final AdamParameters parameters;
  private final int mains;
  private final int grades;
  private final int gradeGroups;

  private OutcomeGraph(OutcomeShape shape, float[] weights) {
    int[] starts = OutcomeTensor.starts(shape);
    if (weights.length != starts[TENSORS.length]) {
      throw new IllegalArgumentException(shape + " needs " + starts[TENSORS.length] + " weights, got " + weights.length);
    }
    mains = OutcomeTerm.MAIN.width(shape.variant());
    grades = OutcomeTerm.GRADE.width(shape.variant());
    gradeGroups = grades / HandOutcomes.BASES;
    parameters = new AdamParameters(weights, starts, forms(shape), hidden());
  }

  /** 重みは OutcomeTensor の宣言の順に並べて渡す。 */
  static OutcomeGraph of(OutcomeShape shape, float[] weights) {
    return new OutcomeGraph(shape, weights);
  }

  /**
   * ゼロから学習するときの初期の重みを、OutcomeTensor の順に並べて返す。隠れ層の重みは He の初期値、ほかは 0 にする。そのため
   * 学習の始めは、終わり方と打点の確率が一様、立直の確率が 1/2 になる。
   */
  static float[] scratch(OutcomeShape shape, SplittableRandom random) {
    return AdamParameters.scratch(OutcomeTensor.starts(shape), forms(shape), hidden(), random);
  }

  @Override
  public float[] weights() {
    return parameters.weights();
  }

  @Override
  public TrainingStep step(OutcomeBatch batch, double learningRate, double hiddenDecay, double clipNorm) {
    return parameters.step((scope, weights) -> loss(scope, batch, weights), batch.size, learningRate, hiddenDecay,
        clipNorm);
  }

  @Override
  public double weightedNll(OutcomeBatch batch) {
    return parameters.weightedNll((scope, weights) -> loss(scope, batch, weights));
  }

  /** バッチの対数確率の項を、[行][項] の行優先で返す。勾配は作らない。 */
  float[] terms(OutcomeBatch batch) {
    return parameters.evaluate((scope, weights) -> terms(scope, batch, weights));
  }

  @Override
  public void close() {
    parameters.close();
  }

  /**
   * 対数確率の項を [行][項] で返す。項は HandOutcomes の順で、終わり方の log softmax、打点の組ごとの log softmax、log σ(z)、
   * log σ(−z) と並ぶ。
   */
  private NDArray terms(NDManager scope, OutcomeBatch batch, IntFunction<NDArray> weights) {
    if (batch.size == 0) {
      throw new IllegalArgumentException("An empty batch");
    }
    NDArray x = AdamParameters.rows(scope, batch.inputs, new Shape(batch.size, batch.width));
    NDArray first = Activation.relu(
        x.matMul(weights.apply(OutcomeTensor.W1.ordinal())).add(weights.apply(OutcomeTensor.B1.ordinal())));
    NDArray second = Activation.relu(
        first.matMul(weights.apply(OutcomeTensor.W2.ordinal())).add(weights.apply(OutcomeTensor.B2.ordinal())));
    NDArray logits =
        second.matMul(weights.apply(OutcomeTensor.W3.ordinal())).add(weights.apply(OutcomeTensor.B3.ordinal()));
    NDArray main = logits.get(":, :{}", mains).logSoftmax(1);
    NDArray grade = logits.get(":, {}:{}", mains, mains + grades)
        .reshape(batch.size, gradeGroups, HandOutcomes.BASES)
        .logSoftmax(2)
        .reshape(batch.size, grades);
    NDArray riichi = logits.get(":, {}:", mains + grades);
    NDArray declared = Activation.softPlus(riichi.neg()).neg();
    NDArray quiet = Activation.softPlus(riichi).neg();
    return main.concat(grade, 1).concat(declared, 1).concat(quiet, 1);
  }

  /** 行ごとの w·(−log Σ_k exp(Σ_j 項)) を返す。項の空きは値が 0 の列を読み、結果の無い k はマスクで外す。 */
  private NDArray loss(NDManager scope, OutcomeBatch batch, IntFunction<NDArray> weights) {
    int columns = batch.outcomes * batch.stride;
    NDArray indices = AdamParameters.rows(scope, batch.terms, new Shape(batch.size, columns));
    NDArray mask = AdamParameters.rows(scope, batch.mask, new Shape(batch.size, batch.outcomes));
    NDArray rowWeights = AdamParameters.rows(scope, batch.weights, new Shape(batch.size));
    NDArray padding = scope.zeros(new Shape(batch.size, 1));
    NDArray logs = terms(scope, batch, weights).concat(padding, 1)
        .gather(indices, 1)
        .reshape(batch.size, batch.outcomes, batch.stride)
        .sum(TERMS_AXIS)
        .add(mask);
    NDArray top = logs.max(OUTCOMES_AXIS, true).stopGradient();
    return logs.sub(top).exp().sum(OUTCOMES_AXIS).log().add(top.squeeze(1)).neg().mul(rowWeights);
  }

  private static Shape[] forms(OutcomeShape shape) {
    Shape[] forms = new Shape[TENSORS.length];
    for (OutcomeTensor tensor : TENSORS) {
      forms[tensor.ordinal()] = switch (tensor) {
        case W1, W2, W3 -> new Shape(tensor.rows(shape), tensor.columns(shape));
        case B1, B2, B3 -> new Shape(tensor.columns(shape));
      };
    }
    return forms;
  }

  private static boolean[] hidden() {
    boolean[] hidden = new boolean[TENSORS.length];
    for (OutcomeTensor tensor : TENSORS) {
      hidden[tensor.ordinal()] = tensor.hidden();
    }
    return hidden;
  }
}
