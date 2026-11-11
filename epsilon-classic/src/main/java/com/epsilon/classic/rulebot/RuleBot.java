package com.epsilon.classic.rulebot;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.InlinePolicy;
import com.epsilon.agent.policy.Series;
import com.epsilon.mahjong.rules.Variant;
import java.util.Objects;

/**
 * 公開の情報だけで、牌効率・打点・鳴き・押し引きを決めるルールベースの打ち手。読むのは判断の記録（自分の手・副露・河・立直者の現物・
 * ドラ表示牌・持ち点・巡目・山の残り）だけで、乱数は使わない。
 *
 * <p>手順:
 *
 * <ul>
 *   <li>ツモ和了かロンができれば和了し、北を抜けるなら抜く。
 *   <li>他家の打牌への応答では、脅威（立直した他家と、副露の多い他家）があれば鳴かない。宣言牌への応答では、宣言した席も立直した他家として
 *       扱う。脅威が無ければ、鳴いた後に役が見込めて手が進む鳴きだけをする。
 *   <li>自分の手番では、終盤の最下位でなく、手が遠ければ（通常形・七対子・国士のどれも 3 向聴以上）九種九牌で流す。打牌は、打った後の
 *       向聴と攻めの点で比べる。脅威があれば押すか降りるかを先に決め、その後で立直と槓を選ぶ。
 * </ul>
 *
 * <p>席の数・終盤の局・山の枚数で変わる値は、ルールから求める。作業領域はスレッドごとに持つので、複数のスレッドから同時に呼んでよい。
 * ウォームアップの後は、手を選ぶときにメモリを確保しない。
 */
public final class RuleBot implements InlinePolicy {

  private final Variant variant;
  private final String label;
  private final ThreadLocal<Rules> rules;

  public RuleBot(Variant variant) {
    this.variant = Objects.requireNonNull(variant, "variant");
    label = Series.RULE_BOT.label(null);
    rules = ThreadLocal.withInitial(() -> new Rules(variant));
  }

  @Override
  public Variant variant() {
    return variant;
  }

  @Override
  public String label() {
    return label;
  }

  @Override
  public int act(Observation observation) {
    if (observation.variant() != variant) {
      throw new IllegalArgumentException(
          "Variant mismatch: expected " + variant + " but found " + observation.variant());
    }
    return rules.get().choose(observation);
  }

  /** 選んだ手の確率を 1、ほかの手を 0 にした評価を返す。価値は出さない。 */
  @Override
  public Evaluation evaluate(Observation observation) {
    int greedy = act(observation);
    float[] probabilities = new float[observation.legalActions()];
    probabilities[greedy] = 1f;
    return new Evaluation(probabilities, null, greedy);
  }

  @Override
  public void close() {}
}
