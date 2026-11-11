package com.epsilon.prism.learn;

import com.epsilon.prism.Limits;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.LearningRateSchedule;
import com.epsilon.prism.model.PrismShape;
import java.util.Objects;

/**
 * 1 回の学習の計画。学習器の値（ステップ数・バッチ・学習率・勾配の上限・温度 λ_c）は、学習器の設定（{@link FitSettings}。run の
 * 設定から作る本番の形）をそのまま持つ。それに、ネットワークの形、補助のヘッドの重み、マイクロバッチのトークンの上限、乱数の種を足す。
 * バージョンのスナップショットに、学習の設定として書く。
 *
 * @param learner 学習器の設定（ステップ数は、行の数とエポック数から決めたもの）
 * @param shape 学習するネットワークの形
 * @param auxiliary 補助のヘッドの重み（α_o・α_r）
 * @param tokenBudget マイクロバッチに詰めるトークン（行の数 × 最も長い行の S）の上限
 * @param seed ネットワークの初期値と行の並べ替えに使う乱数の種
 */
public record FitPlan(
    FitSettings learner,
    PrismShape shape,
    AuxiliaryLoss auxiliary,
    int tokenBudget,
    long seed) {

  /** マイクロバッチのトークンの上限の既定値。 */
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
    if (tokenBudget < Limits.MAX_ROW_TOKENS) {
      throw new IllegalArgumentException(
          "token budget cannot hold the longest row: " + tokenBudget);
    }
  }

  /**
   * 学習率のスケジュールを返す。始めの {@link FitSettings#WARMUP} の割合のステップで lrPeak まで上げ、その後 cosine で lrFloor まで
   * 下げる。
   *
   * @return 学習率のスケジュール
   */
  public LearningRateSchedule schedule() {
    return LearningRateSchedule.of(
        learner.lrPeak(), learner.lrFloor(), FitSettings.WARMUP, learner.steps());
  }
}
