package com.epsilon.reviewer.engine;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.FixtureFamily;
import com.epsilon.reviewer.engine.review.ReviewModel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * テスト用のモデル。受け取った O の写しを順に取っておき、O だけから決まる価値を返す。既定は四人打ち。同じ O には同じ評価を返すので、
 * 候補の順も比べられる。
 */
final class RecordingPolicy implements ReviewModel {
  /** 受け取った O の写し。受け取った順に並ぶ。 */
  final List<long[]> inputs = new ArrayList<>();

  private final Variant variant;
  private final ObservationBatch batch;

  RecordingPolicy() {
    this(Variant.TENHOU_4P);
  }

  RecordingPolicy(Variant variant) {
    this.variant = variant;
    batch = new ObservationBatch(variant, 256);
  }

  @Override
  public Evaluation evaluate(Observation observation) {
    batch.clear();
    int index = batch.add(observation);
    int start = batch.start(index);
    long[] words = Arrays.copyOfRange(batch.words(), start, start + batch.longs(index));
    inputs.add(words);
    int hash = Arrays.hashCode(words);
    int count = observation.legalActions();
    float[] values = new float[count];
    int greedy = 0;
    for (int i = 0; i < count; i++) {
      values[i] = Math.floorMod(31 * hash + observation.action(i).code(), 997) * 1e-5f;
      if (values[i] > values[greedy]) greedy = i;
    }
    double[] weights = new double[count];
    double sum = 0;
    for (int i = 0; i < count; i++) {
      weights[i] = Math.exp((values[i] - values[greedy]) / temperature());
      sum += weights[i];
    }
    float[] probabilities = new float[count];
    for (int i = 0; i < count; i++) probabilities[i] = (float) (weights[i] / sum);
    return new Evaluation(probabilities, values, greedy);
  }

  /** {@code index} 番目に受け取った O を、比べるフィールドに読み直す。 */
  ObservedInput input(int index) {
    return ObservedInput.of(variant, inputs.get(index));
  }

  @Override
  public boolean valued() {
    return true;
  }

  @Override
  public double temperature() {
    return FixtureFamily.TEMPERATURE;
  }

  @Override
  public Variant variant() {
    return variant;
  }
}
