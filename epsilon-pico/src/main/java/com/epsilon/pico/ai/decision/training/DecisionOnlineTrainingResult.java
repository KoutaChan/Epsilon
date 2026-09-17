package com.epsilon.pico.ai.decision.training;

import java.util.Objects;

/** 一連の順伝播と逆伝播から得た、方策と価値関数それぞれの学習結果。 */
public record DecisionOnlineTrainingResult(
    DecisionTrainingResult actorMetrics,
    DecisionTrainingResult valueMetrics,
    float meanActorUpdateKl) {

  public DecisionOnlineTrainingResult {
    Objects.requireNonNull(actorMetrics, "actorMetrics");
    Objects.requireNonNull(valueMetrics, "valueMetrics");
    if (actorMetrics.optimizerSteps() != valueMetrics.optimizerSteps()) {
      throw new IllegalStateException(
          "Fused Actor/Value optimizer-step mismatch: actor="
              + actorMetrics.optimizerSteps()
              + " value="
              + valueMetrics.optimizerSteps());
    }
    if (!actorMetrics.rejection().equals(valueMetrics.rejection())) {
      throw new IllegalStateException("Fused Actor/Value rejection mismatch");
    }
  }

  public int optimizerSteps() {
    return actorMetrics.optimizerSteps();
  }

  public boolean rejected() {
    return actorMetrics.rejected();
  }
}
