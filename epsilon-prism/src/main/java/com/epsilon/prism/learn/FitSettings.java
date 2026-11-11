package com.epsilon.prism.learn;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.model.LearningRateSchedule;
import com.epsilon.prism.model.PrismLoss;

/**
 * 1 回の学習（{@link Fit}）の設定。学習率は最初の 5% のステップで 0 からピークまで線形に上げ、その後 cosine で下限まで
 * 下げる（{@link LearningRateSchedule#of}）。損失は差分整合で、KL の基準は候補の上の一様分布にする（{@link PrismLoss}）。
 *
 * @param steps ステップ数
 * @param batchSize 1 ステップの行数
 * @param lrPeak 学習率のピーク
 * @param lrFloor 学習率の下限
 * @param gradientClip 勾配のノルムの上限（0 なら切らない）
 * @param lambdaDiscard 打牌のクラスの温度
 * @param lambdaCommit 確約のクラスの温度
 */
public record FitSettings(
    int steps,
    int batchSize,
    double lrPeak,
    double lrFloor,
    float gradientClip,
    double lambdaDiscard,
    double lambdaCommit) {
  /** 学習率を上げていく区間が全ステップに占める割合。 */
  public static final double WARMUP = 0.05;

  /** 判断のクラス c の温度を返す。 */
  public float lambda(DecisionClass c) {
    return (float) (c.commitment() ? lambdaCommit : lambdaDiscard);
  }
}
