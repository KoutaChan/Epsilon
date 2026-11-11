package com.epsilon.classic.rulebot;

import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.mahjong.rules.Variant;
import java.nio.file.Path;
import java.util.Objects;

/** ルールベースの打ち手の系列（{@link Series#RULE_BOT}）。チェックポイントもデバイスも使わない。ServiceLoader が META-INF/services から見つける。 */
public final class RuleBotFamily implements ModelFamily {

  @Override
  public Series series() {
    return Series.RULE_BOT;
  }

  /**
   * ルールベースの打ち手を作る。
   *
   * @param checkpoint 使わないので null を渡す
   * @param options 使わない
   * @param runtime 使わない
   * @throws IllegalArgumentException チェックポイントを渡したとき
   */
  @Override
  public Policy open(Path checkpoint, Variant variant, PolicyOptions options, NetworkRuntime runtime) {
    if (checkpoint != null) {
      throw new IllegalArgumentException("The rule bot takes no checkpoint: " + checkpoint);
    }
    return new RuleBot(Objects.requireNonNull(variant, "variant"));
  }
}
