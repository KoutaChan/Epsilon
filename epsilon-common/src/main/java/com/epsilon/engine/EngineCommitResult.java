package com.epsilon.engine;

import java.util.List;

/**
 * 選択した行動を局面へ反映した後の進行状態と、各行動が実行されたかどうかを表す。エンジンが再利用するため、次の行動反映を超えて保持してはならない。
 *
 * <dl>
 *   <dt>{@code step}
 *   <dd>行動を反映した後に到達した次のエンジン境界
 *   <dt>{@code decisionOutcomes}
 *   <dd>各選択の実行結果と学習上の役割を格納した借用一覧。次の行動反映まで有効
 * </dl>
 */
public final class EngineCommitResult {

  private GameStepResult step;
  private List<EngineDecisionOutcome> decisionOutcomes = List.of();

  EngineCommitResult() {}

  EngineCommitResult bind(GameStepResult step, List<EngineDecisionOutcome> decisionOutcomes) {
    this.step = step;
    this.decisionOutcomes = decisionOutcomes;
    return this;
  }

  public GameStepResult step() {
    return step;
  }

  public List<EngineDecisionOutcome> decisionOutcomes() {
    return decisionOutcomes;
  }
}
