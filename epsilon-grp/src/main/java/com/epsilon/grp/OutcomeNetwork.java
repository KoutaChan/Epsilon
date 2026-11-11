package com.epsilon.grp;

import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;

/**
 * 次局の結果の確率 q(o|s) を出すネットワーク（純 Java）。h1 = ReLU(W1·x ＋ b1)、h2 = ReLU(W2·h1 ＋ b2)、z = W3·h2 ＋ b3 で、z は
 * HandOutcomes の項の位置に並ぶロジット。終わり方の項の softmax が終わり方の確率に、打点の項の組ごとの softmax が打点の確率に、
 * 「立直した」の項の σ(z) が相対席と立場ごとの立直の確率になる。1 つの結果の q は、終わり方の確率に、和了者ごとの打点の確率と、
 * 立直の有無を確率に入れる席ごとの立直の確率を掛けたもの。どの項を掛けるかは HandOutcomes が決める。
 *
 * <p>層の計算には Dense を使う。隠れ層は float、ロジットから先は double で求め、指数と対数は StrictMath で求める。値は卓だけで
 * 決まり、ビット単位で再現する。重みは書き換えず、作業領域はスレッドごとに持つ。
 */
public final class OutcomeNetwork {

  private final OutcomeShape shape;
  private final float[] weights;

  /** テンソルごとの、重みの配列の中での開始位置（OutcomeTensor の宣言の順）。 */
  private final int[] starts;

  private final int width;
  private final int hidden;
  private final int logits;
  private final int mainStart;
  private final int mains;
  private final int gradeStart;
  private final int gradeGroups;
  private final int riichiStart;
  private final int noRiichiStart;
  private final int riichiCount;

  private OutcomeNetwork(OutcomeShape shape, float[] weights) {
    this.shape = shape;
    this.weights = weights;
    starts = OutcomeTensor.starts(shape);
    HandOutcomes outcomes = shape.outcomes();
    width = shape.width();
    hidden = shape.hidden();
    logits = outcomes.logits();
    mainStart = outcomes.termStart(OutcomeTerm.MAIN);
    mains = outcomes.mainCount();
    gradeStart = outcomes.termStart(OutcomeTerm.GRADE);
    gradeGroups = OutcomeTerm.GRADE.width(shape.variant()) / HandOutcomes.BASES;
    riichiStart = outcomes.termStart(OutcomeTerm.RIICHI);
    noRiichiStart = outcomes.termStart(OutcomeTerm.NO_RIICHI);
    riichiCount = OutcomeTerm.RIICHI.width(shape.variant());
  }

  /**
   * 重みのコピーを持つネットワークを作る。重みの数が形に合わないか、有限でない値があれば IllegalArgumentException を投げる。
   *
   * @param weights OutcomeTensor の宣言の順に並べた重み
   */
  public static OutcomeNetwork of(OutcomeShape shape, float[] weights) {
    int expected = OutcomeTensor.starts(shape)[OutcomeTensor.values().length];
    if (weights.length != expected) {
      throw new IllegalArgumentException(shape + " needs " + expected + " weights, got " + weights.length);
    }
    for (int index = 0; index < weights.length; index++) {
      if (!Float.isFinite(weights[index])) {
        throw new IllegalArgumentException("Weight " + index + " is not finite: " + weights[index]);
      }
    }
    return new OutcomeNetwork(shape, weights.clone());
  }

  public OutcomeShape shape() {
    return shape;
  }

  public Variant variant() {
    return shape.variant();
  }

  /** 重みのコピーを返す。並びは OutcomeTensor の宣言の順。 */
  public float[] weights() {
    return weights.clone();
  }

  /** 半荘が続く卓の、各項の対数確率と確率を作業領域に書く。 */
  public void forward(GrpState state, OutcomeWorkspace workspace) {
    forward(check(workspace).encoder.load(state), workspace);
  }

  /** 半荘が続く卓の、各項の対数確率と確率を作業領域に書く。 */
  public void forward(TableState table, OutcomeWorkspace workspace) {
    check(workspace).encoder.encode(table, workspace.input);
    Dense.relu(
        weights,
        starts[OutcomeTensor.W1.ordinal()],
        starts[OutcomeTensor.B1.ordinal()],
        workspace.input,
        width,
        workspace.hidden1,
        hidden);
    Dense.relu(
        weights,
        starts[OutcomeTensor.W2.ordinal()],
        starts[OutcomeTensor.B2.ordinal()],
        workspace.hidden1,
        hidden,
        workspace.hidden2,
        hidden);
    Dense.logits(
        weights,
        starts[OutcomeTensor.W3.ordinal()],
        starts[OutcomeTensor.B3.ordinal()],
        workspace.hidden2,
        hidden,
        workspace.logits,
        logits);
    double[] z = workspace.logits;
    logSoftmax(z, mainStart, mains, workspace);
    for (int group = 0; group < gradeGroups; group++) {
      logSoftmax(z, gradeStart + group * HandOutcomes.BASES, HandOutcomes.BASES, workspace);
    }
    double[] terms = workspace.terms;
    double[] probabilities = workspace.probabilities;
    for (int index = 0; index < riichiCount; index++) {
      int declared = riichiStart + index;
      int quiet = noRiichiStart + index;
      double value = z[declared];
      double small = StrictMath.exp(-Math.abs(value));
      double log1p = StrictMath.log1p(small);
      double large = 1 / (1 + small);
      if (value >= 0) {
        terms[declared] = -log1p;
        terms[quiet] = -value - log1p;
        probabilities[declared] = large;
        probabilities[quiet] = small * large;
      } else {
        terms[declared] = value - log1p;
        terms[quiet] = -log1p;
        probabilities[declared] = small * large;
        probabilities[quiet] = large;
      }
    }
  }

  private OutcomeWorkspace check(OutcomeWorkspace workspace) {
    if (workspace.shape != shape && !workspace.shape.equals(shape)) {
      throw new IllegalArgumentException("Workspace for " + workspace.shape + " used with " + shape);
    }
    return workspace;
  }

  /**
   * z の 1 つの区間について log softmax と softmax を求め、作業領域の対数確率と確率の同じ位置に書く。
   *
   * @param from 区間の開始位置
   * @param count 区間の長さ
   */
  private static void logSoftmax(double[] z, int from, int count, OutcomeWorkspace workspace) {
    double max = z[from];
    for (int k = 1; k < count; k++) {
      max = Math.max(max, z[from + k]);
    }
    double sum = 0;
    for (int k = 0; k < count; k++) {
      double exponent = StrictMath.exp(z[from + k] - max);
      workspace.probabilities[from + k] = exponent;
      sum += exponent;
    }
    double log = StrictMath.log(sum);
    for (int k = 0; k < count; k++) {
      workspace.terms[from + k] = z[from + k] - max - log;
      workspace.probabilities[from + k] /= sum;
    }
  }

  @Override
  public String toString() {
    return "OutcomeNetwork[" + shape + "]";
  }
}
