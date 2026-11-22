package com.epsilon.prism.learn;

import com.epsilon.prism.core.AgariOrder;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.model.LearningRateSchedule;
import com.epsilon.prism.model.PairLoss;
import com.epsilon.prism.model.PrismLoss;

/**
 * 1 回の学習（{@link Fit}）の設定。学習率は最初の 5% のステップで 0 からピークまで線形に上げ、その後 cosine で下限まで
 * 下げる（{@link LearningRateSchedule#of}）。報告する損失は学習の目的関数によらず差分整合で求める（{@link PrismLoss}）。
 *
 * @param steps ステップ数
 * @param batchSize 1 ステップの行数
 * @param lrPeak 学習率のピーク
 * @param lrFloor 学習率の下限
 * @param gradientClip 勾配のノルムの上限（0 なら切らない）
 * @param lambdaDiscard 打牌のクラスの温度
 * @param lambdaCommit 確約のクラスの温度
 * @param loss 手の差の学習に使う目的関数
 * @param classWeight クラスごとの損失の重み
 * @param adamEpsilon AdamW の分母に加える ε の扱い
 * @param rowAnchor 行の平均のアンカーの重みの決め方
 * @param rowAnchorDiscards 行の平均のアンカーで平均を取る打牌の集合
 * @param agariOrder 証明できる和了の順序を教師・学習・推論に反映するか
 */
public record FitSettings(
    int steps,
    int batchSize,
    double lrPeak,
    double lrFloor,
    float gradientClip,
    double lambdaDiscard,
    double lambdaCommit,
    PairLoss.Mode loss,
    PairLoss.ClassWeight classWeight,
    AdamEpsilon adamEpsilon,
    PrismLoss.RowAnchor rowAnchor,
    PrismLoss.RowAnchorDiscards rowAnchorDiscards,
    AgariOrder.Mode agariOrder) {

  /** 指定のない目的関数・判断クラスの重み・ε の扱い・アンカー・和了順序には既定値を使う。 */
  public FitSettings {
    if (adamEpsilon == null) adamEpsilon = AdamEpsilon.FIXED;
    if (rowAnchor == null) rowAnchor = PrismLoss.RowAnchor.GAIN;
    if (rowAnchorDiscards == null) rowAnchorDiscards = PrismLoss.RowAnchorDiscards.MEASURED;
    if (loss == null) loss = PairLoss.Mode.PRISM;
    if (classWeight == null) classWeight = PairLoss.ClassWeight.INVERSE_LAMBDA;
    if (agariOrder == null) agariOrder = AgariOrder.Mode.LEGACY;
  }

  /** 和了順序の制約を掛けない設定。 */
  public FitSettings(
      int steps,
      int batchSize,
      double lrPeak,
      double lrFloor,
      float gradientClip,
      double lambdaDiscard,
      double lambdaCommit,
      PairLoss.Mode loss,
      PairLoss.ClassWeight classWeight,
      AdamEpsilon adamEpsilon,
      PrismLoss.RowAnchor rowAnchor) {
    this(
        steps,
        batchSize,
        lrPeak,
        lrFloor,
        gradientClip,
        lambdaDiscard,
        lambdaCommit,
        loss,
        classWeight,
        adamEpsilon,
        rowAnchor,
        null,
        null);
  }

  /** アンカーに教師のゲインを掛ける設定。 */
  public FitSettings(
      int steps,
      int batchSize,
      double lrPeak,
      double lrFloor,
      float gradientClip,
      double lambdaDiscard,
      double lambdaCommit,
      PairLoss.Mode loss,
      PairLoss.ClassWeight classWeight,
      AdamEpsilon adamEpsilon) {
    this(
        steps,
        batchSize,
        lrPeak,
        lrFloor,
        gradientClip,
        lambdaDiscard,
        lambdaCommit,
        loss,
        classWeight,
        adamEpsilon,
        null,
        null,
        null);
  }

  /** 学習率を上げていく区間が全ステップに占める割合。 */
  public static final double WARMUP = 0.05;

  /** 判断のクラス c の温度を返す。 */
  public float lambda(DecisionClass c) {
    return (float) (c.commitment() ? lambdaCommit : lambdaDiscard);
  }
}
