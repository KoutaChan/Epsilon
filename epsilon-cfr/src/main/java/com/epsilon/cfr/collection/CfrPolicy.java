package com.epsilon.cfr.collection;

import com.epsilon.cfr.data.ObservationPayload;
import com.epsilon.runtime.DecisionInferenceIngress;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/** 現在の本人観測に同じ凍結方策を適用する。主経路と分岐で方策を変えず、結果は入力順に返す。この段階では探索を加えない。 */
@FunctionalInterface
public interface CfrPolicy {
  List<Prediction> evaluate(List<DecisionContext> batch);

  static CfrPolicy uniform() {
    ObservationPayload emptyObservation = ObservationPayload.copyOf(new byte[0]);
    return rows ->
        rows.stream()
            .map(
                row -> {
                  double[] probabilities = new double[row.legalActions().size()];
                  Arrays.fill(probabilities, 1d / probabilities.length);
                  return new Prediction(probabilities, emptyObservation);
                })
            .toList();
  }

  default boolean needsInferenceWork() {
    return true;
  }

  default DecisionInferenceIngress.Attempt tryAcquireInferenceIngress() {
    return DecisionInferenceIngress.Attempt.acquired(DecisionInferenceIngress.direct(this));
  }

  /** 推論枠の予約を引き受ける。失敗した場合も呼び出し側へ所有権を戻さない。 */
  default CompletableFuture<List<Prediction>> submitBatch(
      List<DecisionContext> batch, DecisionInferenceIngress ingress) {
    ingress.handoff(this).move();
    try {
      return CompletableFuture.completedFuture(evaluate(batch));
    } catch (RuntimeException | Error failure) {
      return CompletableFuture.failedFuture(failure);
    }
  }
}
