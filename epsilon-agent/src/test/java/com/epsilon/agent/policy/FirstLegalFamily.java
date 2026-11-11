package com.epsilon.agent.policy;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.mahjong.rules.Variant;
import java.nio.file.Path;
import java.util.Arrays;

/** テストで使う {@code RULE_BOT} 系列の実装。いつも最初の合法手を選び、評価の確率は一様にする。ServiceLoader が META-INF/services から見つける。 */
public final class FirstLegalFamily implements ModelFamily {

  @Override
  public Series series() {
    return Series.RULE_BOT;
  }

  /** 渡されたランタイムは使わない。 */
  @Override
  public Policy open(Path checkpoint, Variant variant, PolicyOptions options, NetworkRuntime runtime) {
    return new FirstLegal(variant, series().label(checkpoint));
  }

  private record FirstLegal(Variant variant, String label) implements InlinePolicy {

    @Override
    public int act(Observation observation) {
      requireVariant(observation);
      return 0;
    }

    @Override
    public Evaluation evaluate(Observation observation) {
      requireVariant(observation);
      float[] probabilities = new float[observation.legalActions()];
      Arrays.fill(probabilities, 1f / probabilities.length);
      return new Evaluation(probabilities, null, act(observation));
    }

    @Override
    public void close() {}

    private void requireVariant(Observation observation) {
      if (observation.variant() != variant) {
        throw new IllegalArgumentException(
            "Variant mismatch: expected " + variant + " but found " + observation.variant());
      }
    }
  }
}
