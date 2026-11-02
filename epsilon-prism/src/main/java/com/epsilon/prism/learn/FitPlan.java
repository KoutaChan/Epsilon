package com.epsilon.prism.learn;

import com.epsilon.prism.Limits;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.LearningRateSchedule;
import com.epsilon.prism.model.PrismShape;
import java.util.Objects;

/**
 * 1 回の当てはめの計画。学習器の数（歩数・バッチ・学習率・勾配の上限・温度 λ_c）は学習器の設定（{@link FitSettings}。run の設定から
 * 作る本番の形）をそのまま持ち、網の形（前線の腕の有無を含む）・補助の頭の重み・微小バッチのトークンの上限・種を足す。版の
 * スナップショットに学習の設定として書く。
 *
 * @param learner 学習器の設定（歩数は行の数と周回数から決めたもの）
 * @param shape 白紙の網の形（{@link PrismShape#frontier()} が前線の腕なら、行ごとに前線も書く）
 * @param auxiliary 補助の頭の重み（α_o・α_r）
 * @param tokenBudget 微小バッチの詰めたトークン（行の数 × 最も長い行の S）の上限
 * @param seed 網の初期値と行の並べ替えの種
 */
public record FitPlan(
    FitSettings learner,
    PrismShape shape,
    AuxiliaryLoss auxiliary,
    int tokenBudget,
    long seed) {

  /** 微小バッチのトークンの上限の既定。 */
  public static final int TOKEN_BUDGET = 160_000;

  /**
   * 計画を確かめる。
   *
   * @throws IllegalArgumentException トークンの上限が最も長い 1 行より小さいとき
   */
  public FitPlan {
    Objects.requireNonNull(learner, "learner");
    Objects.requireNonNull(shape, "shape");
    Objects.requireNonNull(auxiliary, "auxiliary");
    if (tokenBudget < BatchPlan.FIXED_TOKENS + Limits.MAX_EVENT_TOKENS) {
      throw new IllegalArgumentException(
          "token budget cannot hold the longest row: " + tokenBudget);
    }
  }

  /**
   * 当てはめの学習率（始めの {@link FitSettings#WARMUP} で lrPeak まで上げ、cosine で lrFloor へ）。
   *
   * @return 学習率
   */
  public LearningRateSchedule schedule() {
    return LearningRateSchedule.of(
        learner.lrPeak(), learner.lrFloor(), FitSettings.WARMUP, learner.steps());
  }
}
