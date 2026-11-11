package com.epsilon.grp;

import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Objects;

/**
 * GRP のネットワーク。h1 = ReLU(W1·x ＋ b1)、h2 = ReLU(W2·h1 ＋ b2)、z = W3·h2 ＋ b3、p = softmax(z)。p は最終順位の S! 通りの
 * 順列それぞれの確率。席は相対席で、順列の番号は RankPermutations に従う。段階ごとの出力ヘッドを持つときは、次局の段階
 * （GrpPhase）の W3・b3 を使う。重みは書き換えず、作業領域はスレッドごとに持つ。
 *
 * <p>層の計算には Dense を使う。隠れ層は float、z と softmax は double で求め、指数は StrictMath で求める。値は卓だけで決まり、
 * ビット単位で再現する。
 */
public final class GrpNetwork implements GrpModel {

  private final GrpShape shape;
  private final GrpLayout layout;
  private final RankPermutations permutations;
  private final float[] weights;

  /** テンソルごとの、重みの配列の中での開始位置（GrpTensor の宣言の順）。 */
  private final int[] starts;

  private final int width;
  private final int hidden;
  private final int outputs;

  private GrpNetwork(GrpShape shape, float[] weights) {
    this.shape = shape;
    this.weights = weights;
    layout = shape.layout();
    permutations = RankPermutations.of(shape.variant());
    starts = GrpTensor.starts(shape);
    width = layout.width();
    hidden = shape.hidden();
    outputs = permutations.count();
  }

  /**
   * 重みのコピーを持つネットワークを作る。重みの数が形に合わないか、有限でない値があれば IllegalArgumentException を投げる。
   *
   * @param weights GrpTensor の宣言の順に並べた重み
   */
  public static GrpNetwork of(GrpShape shape, float[] weights) {
    int expected = GrpTensor.starts(shape)[GrpTensor.values().length];
    if (weights.length != expected) {
      throw new IllegalArgumentException(shape + " needs " + expected + " weights, got " + weights.length);
    }
    for (int index = 0; index < weights.length; index++) {
      if (!Float.isFinite(weights[index])) {
        throw new IllegalArgumentException("Weight " + index + " is not finite: " + weights[index]);
      }
    }
    return new GrpNetwork(shape, weights.clone());
  }

  public GrpShape shape() {
    return shape;
  }

  @Override
  public Variant variant() {
    return shape.variant();
  }

  public GrpLayout layout() {
    return layout;
  }

  /** 重みのコピーを返す。並びは GrpTensor の宣言の順。 */
  public float[] weights() {
    return weights.clone();
  }

  /**
   * 順列ごとの確率を書く。
   *
   * @param out 書き先（長さ S!）
   */
  public void permutations(GrpState state, GrpWorkspace workspace, double[] out) {
    permutations(check(workspace).encoder.load(state), workspace, out);
  }

  /**
   * 半荘が続く卓の、順列ごとの確率を書く。
   *
   * @param out 書き先（長さ S!）
   */
  public void permutations(TableState table, GrpWorkspace workspace, double[] out) {
    requireOutput(out, outputs);
    forward(table, check(workspace));
    System.arraycopy(workspace.probabilities, 0, out, 0, outputs);
  }

  /**
   * 順列 k の対数確率 log p_k = z_k − max z − log Σ exp(z − max z) を返す。p_k が double で 0 にアンダーフローしても、
   * 有限の値を返す。
   *
   * @param permutation 順列の番号 k
   */
  public double logProbability(GrpState state, int permutation, GrpWorkspace workspace) {
    Objects.checkIndex(permutation, outputs);
    forward(check(workspace).encoder.load(state), workspace);
    return workspace.logits[permutation] - workspace.logitMax - StrictMath.log(workspace.exponentSum);
  }

  /**
   * 絶対席ごとの Φ_i = Σ_π p(π)·u_{π(i)} を書く。
   *
   * @param out 書き先（長さ S）
   */
  public void phi(GrpState state, RankUtility utility, GrpWorkspace workspace, double[] out) {
    phi(check(workspace).encoder.load(state), utility, workspace, out);
  }

  /**
   * 半荘が続く卓の、絶対席ごとの Φ を書く。π の番号の小さい順に足す（{@code RankPermutations.phi}）。
   *
   * @param out 書き先（長さ S）
   */
  public void phi(TableState table, RankUtility utility, GrpWorkspace workspace, double[] out) {
    requireOutput(out, shape.variant().seats());
    forward(table, check(workspace));
    permutations.phi(workspace.probabilities, table.dealer(), utility, out);
  }

  /** p を求めて作業領域に書く。 */
  private void forward(TableState table, GrpWorkspace workspace) {
    workspace.encoder.encode(table, workspace.input);
    int[] at = starts;
    Dense.relu(
        weights, at[GrpTensor.W1.ordinal()], at[GrpTensor.B1.ordinal()], workspace.input, width, workspace.hidden1, hidden);
    Dense.relu(
        weights, at[GrpTensor.W2.ordinal()], at[GrpTensor.B2.ordinal()], workspace.hidden1, hidden, workspace.hidden2,
        hidden);
    int head = shape.phaseHeads() ? GrpPhase.of(shape.variant(), table.round()).ordinal() : 0;
    Dense.logits(
        weights,
        at[GrpTensor.W3.ordinal()] + head * hidden * outputs,
        at[GrpTensor.B3.ordinal()] + head * outputs,
        workspace.hidden2,
        hidden,
        workspace.logits,
        outputs);
    softmax(workspace);
  }

  private GrpWorkspace check(GrpWorkspace workspace) {
    if (workspace.shape != shape && !workspace.shape.equals(shape)) {
      throw new IllegalArgumentException("Workspace for " + workspace.shape + " used with " + shape);
    }
    return workspace;
  }

  /** p_k = exp(z_k − max z) / Σ exp(z_l − max z)。 */
  private static void softmax(GrpWorkspace workspace) {
    double[] z = workspace.logits;
    double[] p = workspace.probabilities;
    double max = z[0];
    for (int k = 1; k < z.length; k++) {
      max = Math.max(max, z[k]);
    }
    double sum = 0;
    for (int k = 0; k < z.length; k++) {
      p[k] = StrictMath.exp(z[k] - max);
      sum += p[k];
    }
    for (int k = 0; k < z.length; k++) {
      p[k] /= sum;
    }
    workspace.logitMax = max;
    workspace.exponentSum = sum;
  }

  private static void requireOutput(double[] out, int length) {
    if (out.length != length) {
      throw new IllegalArgumentException("Output needs " + length + " values, got " + out.length);
    }
  }

  @Override
  public String toString() {
    return "GrpNetwork[" + shape + "]";
  }
}
