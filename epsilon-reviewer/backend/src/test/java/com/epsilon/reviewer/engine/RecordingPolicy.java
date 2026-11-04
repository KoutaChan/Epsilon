package com.epsilon.reviewer.engine;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.reviewer.engine.review.ReviewModel;
import com.epsilon.spi.ReviewEvaluation;
import java.util.ArrayList;
import java.util.List;

/** 受け取った入力を順に覚え、入力から決まる価値を返す試験のモデル。同じ入力には同じ評価を返すので、候補の順も比べられる。 */
final class RecordingPolicy implements ReviewModel {
  final List<PolicyInput> inputs = new ArrayList<>();

  @Override
  public ReviewEvaluation evaluate(PublicObservation observation, List<Action> legalActions) {
    PolicyInput input = PolicyInput.of(observation, legalActions);
    inputs.add(input);
    float[] values = new float[legalActions.size()];
    for (int i = 0; i < values.length; i++)
      values[i] = Math.floorMod(31 * input.hashCode() + legalActions.get(i).toIndex(), 997) * 1e-5f;
    return ReviewEvaluation.valued(
        values, ReviewEvaluation.softmax(values, ReviewEvaluation.DISPLAY_TEMPERATURE));
  }

  @Override
  public double temperature() {
    return ReviewEvaluation.DISPLAY_TEMPERATURE;
  }
}
