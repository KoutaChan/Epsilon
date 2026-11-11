package com.epsilon.grp;

/** GrpNetwork の作業領域。入力を書く GrpEncoder と層ごとの配列を持つ。形ごとに作り、同時に使えるのは 1 スレッドだけ。 */
public final class GrpWorkspace {

  final GrpShape shape;
  final GrpEncoder encoder;
  final float[] input;
  final float[] hidden1;
  final float[] hidden2;
  final double[] logits;
  final double[] probabilities;

  /** 最後に求めた softmax の max z と Σ exp(z − max z)。 */
  double logitMax;
  double exponentSum;

  private GrpWorkspace(GrpShape shape, GrpEncoder encoder) {
    this.shape = shape;
    this.encoder = encoder;
    input = new float[shape.width()];
    hidden1 = new float[shape.hidden()];
    hidden2 = new float[shape.hidden()];
    logits = new double[shape.outputs()];
    probabilities = new double[shape.outputs()];
  }

  public static GrpWorkspace of(GrpShape shape) {
    return new GrpWorkspace(shape, GrpEncoder.of(shape.layout()));
  }

  public GrpShape shape() {
    return shape;
  }
}
