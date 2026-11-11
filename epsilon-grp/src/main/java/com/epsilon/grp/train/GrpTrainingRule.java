package com.epsilon.grp.train;

import java.util.Objects;

/**
 * 学習 1 回分の設定。呼び出し元が、自分の学習の設定から作る。最適化は Adam（β1 0.9、β2 0.999）で、半エポックごとに検証用の行の
 * 損失を測る。
 *
 * @param init 始め方（ゼロから学習するか、WARM か）
 * @param learningRate 始めの学習率（正の有限な値）
 * @param hiddenDecay 隠れ層の重みだけに掛ける重み減衰（0 以上の有限な値。0 なら掛けない）
 * @param maxEpochs エポックの上限（1 以上）
 * @param batchRows 1 ステップの行数（1 以上）
 * @param clipNorm 勾配全体のノルムの上限（正の値）。超えたときだけ勾配を縮める
 * @param halvingPatience 検証用の行の損失がこの回数続けて良くならなければ、学習率を半分にする（1 以上）
 * @param stopPatience 検証用の行の損失がこの回数続けて良くならなければ、学習を止める（{@code halvingPatience} 以上）
 * @param seed 並べ替えと、ゼロから学習するときの初期値を決める乱数の種
 */
public record GrpTrainingRule(
    GrpInit init,
    double learningRate,
    double hiddenDecay,
    int maxEpochs,
    int batchRows,
    double clipNorm,
    int halvingPatience,
    int stopPatience,
    long seed) {

  public GrpTrainingRule {
    Objects.requireNonNull(init, "init");
    if (!(learningRate > 0) || !Double.isFinite(learningRate)) {
      throw new IllegalArgumentException("The learning rate must be positive and finite: " + learningRate);
    }
    if (!(hiddenDecay >= 0) || !Double.isFinite(hiddenDecay)) {
      throw new IllegalArgumentException("The hidden decay must be non-negative and finite: " + hiddenDecay);
    }
    if (maxEpochs < 1 || batchRows < 1) {
      throw new IllegalArgumentException("Epochs and batch rows must be positive: " + maxEpochs + ", " + batchRows);
    }
    if (!(clipNorm > 0)) {
      throw new IllegalArgumentException("The clipping norm must be positive: " + clipNorm);
    }
    if (halvingPatience < 1 || stopPatience < halvingPatience) {
      throw new IllegalArgumentException(
          "Patience must satisfy 1 <= halving <= stop: " + halvingPatience + ", " + stopPatience);
    }
  }

  /**
   * 始めのネットワークが始め方に合うかを確かめる。ゼロから学習するならネットワークが無いこと、WARM なら学習する形と同じ形の
   * ネットワークがあることが条件。
   *
   * @param startShape 始めのネットワークの形。ゼロから学習するなら null
   * @param shape 学習する形
   * @throws IllegalArgumentException 合わないとき
   */
  public <S> void requireStart(S startShape, S shape) {
    if ((startShape == null) != (init == GrpInit.SCRATCH)) {
      throw new IllegalArgumentException("A start network is required exactly for WARM: " + init);
    }
    if (startShape != null && !startShape.equals(shape)) {
      throw new IllegalArgumentException("Start " + startShape + " does not match " + shape);
    }
  }
}
