package com.epsilon.grp;

/**
 * OutcomeNetwork の作業領域。最後に求めた各項の対数確率（HandOutcomes の項の位置に並ぶ）と、確率（対数確率の exp）を持つ。
 * 形ごとに作り、同時に使えるのは 1 スレッドだけ。
 */
public final class OutcomeWorkspace {

  final OutcomeShape shape;
  final GrpEncoder encoder;
  final float[] input;
  final float[] hidden1;
  final float[] hidden2;
  final double[] logits;
  final double[] terms;
  final double[] probabilities;

  private OutcomeWorkspace(OutcomeShape shape, GrpEncoder encoder) {
    this.shape = shape;
    this.encoder = encoder;
    HandOutcomes outcomes = shape.outcomes();
    input = new float[shape.width()];
    hidden1 = new float[shape.hidden()];
    hidden2 = new float[shape.hidden()];
    logits = new double[outcomes.logits()];
    terms = new double[outcomes.terms()];
    probabilities = new double[outcomes.terms()];
  }

  public static OutcomeWorkspace of(OutcomeShape shape) {
    return new OutcomeWorkspace(shape, GrpEncoder.of(shape.layout()));
  }

  public OutcomeShape shape() {
    return shape;
  }

  /** 最後に求めた、指定した項の対数確率を返す。 */
  public double term(int index) {
    return terms[index];
  }

  /** 最後に求めた、指定した項の確率（対数確率の exp）を返す。 */
  public double probability(int index) {
    return probabilities[index];
  }
}
