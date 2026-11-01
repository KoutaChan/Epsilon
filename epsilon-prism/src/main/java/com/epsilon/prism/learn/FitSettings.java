package com.epsilon.prism.learn;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.model.LearningRateSchedule;
import com.epsilon.prism.model.PrismLoss;

/**
 * 1 回の当てはめ（{@link Fit}）の学習器の設定。学習率は始めの 5% で 0 から lrPeak へ線形に上げ、cosine で lrFloor へ
 * 下げる（{@link LearningRateSchedule#of}）。損失は差分整合（κ = 0、KL の基準は C の上の一様分布。{@link
 * PrismLoss}）。
 *
 * @param steps 歩数
 * @param batchSize 1 歩の行数
 * @param gradientClip 勾配のノルムの上限（0 なら切らない）
 * @param lambdaDiscard 打牌の類の温度 λ
 * @param lambdaCommit 確約の類の温度 λ
 */
public record FitSettings(
    int steps,
    int batchSize,
    double lrPeak,
    double lrFloor,
    float gradientClip,
    double lambdaDiscard,
    double lambdaCommit) {
  /** 学習率の warmup の割合。 */
  public static final double WARMUP = 0.05;

  /** 類 c の温度 λ_c。 */
  public float lambda(DecisionClass c) {
    return (float) (c.commitment() ? lambdaCommit : lambdaDiscard);
  }
}
