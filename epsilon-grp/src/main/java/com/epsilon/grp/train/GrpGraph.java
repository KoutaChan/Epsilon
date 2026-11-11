package com.epsilon.grp.train;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import com.epsilon.grp.GrpShape;
import com.epsilon.grp.GrpTensor;
import java.util.SplittableRandom;
import java.util.function.IntFunction;

/**
 * GrpNetwork を DJL（PyTorch、CPU、float32）で組み直したもの。重みは GrpTensor の宣言の順で、W は [入力][出力]、W3 は
 * [ヘッド][隠れ層][S!]、B3 は [ヘッド][1][S!]（バッチの行へブロードキャストする形）。順伝播は GrpNetwork と同じ式で、段階ごとの
 * 出力ヘッドは行ごとの one-hot で選ぶ。行の損失は目標 t との交差エントロピー −Σ_π t(π)·log softmax(z)_π。t が実際の順列 π* の
 * one-hot なら、同時 NLL −log softmax(z)_{π*} そのものになる。
 */
final class GrpGraph implements TrainingLoop.Graph<RowBatch>, AutoCloseable {

  private static final GrpTensor[] TENSORS = GrpTensor.values();
  private static final int[] ROWS_AXIS = {1};
  private static final int[] HEADS_AXIS = {0};

  private final AdamParameters parameters;

  private GrpGraph(GrpShape shape, float[] weights) {
    int[] starts = GrpTensor.starts(shape);
    if (weights.length != starts[TENSORS.length]) {
      throw new IllegalArgumentException(shape + " needs " + starts[TENSORS.length] + " weights, got " + weights.length);
    }
    parameters = new AdamParameters(weights, starts, forms(shape), hidden());
  }

  /** 重みは GrpTensor の宣言の順に並べて渡す。 */
  static GrpGraph of(GrpShape shape, float[] weights) {
    return new GrpGraph(shape, weights);
  }

  /** ゼロから学習するときの初期の重み（GrpTensor の順）。隠れ層の重みは He の初期値、ほかは 0。 */
  static float[] scratch(GrpShape shape, SplittableRandom random) {
    return AdamParameters.scratch(GrpTensor.starts(shape), forms(shape), hidden(), random);
  }

  @Override
  public float[] weights() {
    return parameters.weights();
  }

  @Override
  public TrainingStep step(RowBatch batch, double learningRate, double hiddenDecay, double clipNorm) {
    return parameters.step((scope, weights) -> loss(scope, batch, weights), batch.size, learningRate, hiddenDecay,
        clipNorm);
  }

  @Override
  public double weightedNll(RowBatch batch) {
    return parameters.weightedNll((scope, weights) -> loss(scope, batch, weights));
  }

  /** バッチのロジットを、[行][S!] の行優先で返す。勾配は作らない。 */
  float[] logits(RowBatch batch) {
    return parameters.evaluate((scope, weights) -> logits(scope, batch, weights));
  }

  /** バッチの Σ w·CE / 行数 について、今の重みでの勾配を GrpTensor の順に返す。重みは変えない。 */
  float[] gradient(RowBatch batch) {
    return parameters.gradient((scope, weights) -> loss(scope, batch, weights), batch.size);
  }

  @Override
  public void close() {
    parameters.close();
  }

  /** 行ごとの w·(−Σ_π t(π)·log softmax(z)_π)。 */
  private static NDArray loss(NDManager scope, RowBatch batch, IntFunction<NDArray> weights) {
    NDArray targets = AdamParameters.rows(scope, batch.targets, new Shape(batch.size, batch.outputs));
    NDArray rowWeights = AdamParameters.rows(scope, batch.weights, new Shape(batch.size));
    return logits(scope, batch, weights).logSoftmax(1).mul(targets).sum(ROWS_AXIS).neg().mul(rowWeights);
  }

  /** h1 = ReLU(x·W1 ＋ b1)、h2 = ReLU(h1·W2 ＋ b2)、z = Σ_k m_k ⊙ (h2·W3_k ＋ b3_k)。m は出力ヘッド k を選ぶ one-hot。 */
  private static NDArray logits(NDManager scope, RowBatch batch, IntFunction<NDArray> weights) {
    if (batch.size == 0) {
      throw new IllegalArgumentException("An empty batch");
    }
    NDArray x = AdamParameters.rows(scope, batch.inputs, new Shape(batch.size, batch.width));
    NDArray heads = AdamParameters.rows(scope, batch.headMask, new Shape(batch.size, batch.headCount))
        .transpose()
        .expandDims(2);
    NDArray first = Activation.relu(
        x.matMul(weights.apply(GrpTensor.W1.ordinal())).add(weights.apply(GrpTensor.B1.ordinal())));
    NDArray second = Activation.relu(
        first.matMul(weights.apply(GrpTensor.W2.ordinal())).add(weights.apply(GrpTensor.B2.ordinal())));
    return second.expandDims(0)
        .matMul(weights.apply(GrpTensor.W3.ordinal()))
        .add(weights.apply(GrpTensor.B3.ordinal()))
        .mul(heads)
        .sum(HEADS_AXIS);
  }

  private static Shape[] forms(GrpShape shape) {
    Shape[] forms = new Shape[TENSORS.length];
    for (GrpTensor tensor : TENSORS) {
      forms[tensor.ordinal()] = switch (tensor) {
        case W1, W2 -> new Shape(tensor.rows(shape), tensor.columns(shape));
        case B1, B2 -> new Shape(tensor.columns(shape));
        case W3 -> new Shape(shape.heads(), tensor.rows(shape), tensor.columns(shape));
        case B3 -> new Shape(shape.heads(), 1, tensor.columns(shape));
      };
    }
    return forms;
  }

  private static boolean[] hidden() {
    boolean[] hidden = new boolean[TENSORS.length];
    for (GrpTensor tensor : TENSORS) {
      hidden[tensor.ordinal()] = tensor.hidden();
    }
    return hidden;
  }
}
