package com.epsilon.prism.learn;

import com.epsilon.prism.core.DecisionClass;

/**
 * 1 回の当てはめ（生徒 {@link StudentFit} か蒸留 {@link DistillFit}）の設定。学習率は始めの 5% で 0 から lrPeak へ線形に上げ、cosine で lrFloor へ
 * 下げる（{@link com.epsilon.prism.model.LearningRateSchedule#of}）。
 *
 * @param steps 歩数
 * @param batchSize 1 歩の行数
 * @param gradientClip 勾配のノルムの上限（0 なら切らない）
 * @param lambdaDiscard 打牌の類の温度 λ
 * @param lambdaCommit 確約の類の温度 λ
 * @param kappa 差分整合の C の外の項の係数 κ（蒸留では使わない）
 * @param uniformReference 差分整合の KL の基準を C の上の一様分布にするか（false なら行の規則事前 p0_C）
 */
public record FitSettings(
    int steps,
    int batchSize,
    double lrPeak,
    double lrFloor,
    float gradientClip,
    double lambdaDiscard,
    double lambdaCommit,
    double kappa,
    boolean uniformReference) {
  /** 学習率の warmup の割合。 */
  public static final double WARMUP = 0.05;

  /** 類 c の温度 λ_c。 */
  public float lambda(DecisionClass c) {
    return (float) (c.commitment() ? lambdaCommit : lambdaDiscard);
  }

  /** 損失に渡す KL の基準（一様なら全て 1、C の上で正規化されて 1/|C| になる。そうでなければ行の p0）。 */
  public float[] reference(float[] prior) {
    if (!uniformReference) return prior;
    float[] uniform = new float[prior.length];
    java.util.Arrays.fill(uniform, 1f);
    return uniform;
  }

  /** 最適の s に足される λ_c·log p0 の係数（一様の基準なら 0。保留の指標の debiased はこれを引く）。 */
  public float tilt(DecisionClass c) {
    return uniformReference ? 0f : lambda(c);
  }
}
