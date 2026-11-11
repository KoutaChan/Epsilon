package com.epsilon.prism.collect;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.DeviationTarget;
import com.epsilon.prism.core.Openness;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 中立ロールインの台と決まり（{@link RollIn#support}・{@link RollInRule}）のテスト。根のクラスを持つのが写しの件数だけであることも確かめる。 */
public class RollInRuleTest {

  /** ANY では候補の配列そのもの、OTHER では打った手を除いた昇順の配列を返す。打った手のほかに候補が無ければ例外を投げる。 */
  @Test
  public void supportExcludesThePlayedActionOnlyForOther() {
    int[] candidates = {1, 3, 4, 7};
    Assert.assertSame(RollIn.support(candidates, 3, DeviationTarget.ANY), candidates);
    Assert.assertEquals(RollIn.support(candidates, 3, DeviationTarget.OTHER), new int[] {1, 4, 7});
    Assert.assertEquals(RollIn.support(candidates, 7, DeviationTarget.OTHER), new int[] {1, 3, 4});
    // 打った手が候補の外（一様分布を混ぜた分で選んだ手）なら、台は候補の全て
    Assert.assertEquals(RollIn.support(candidates, 2, DeviationTarget.OTHER), candidates);
    Assert.assertEquals(candidates, new int[] {1, 3, 4, 7});
    Assert.expectThrows(IllegalStateException.class, () -> RollIn.support(new int[] {5}, 5, DeviationTarget.OTHER));
    Assert.assertEquals(RollIn.support(new int[] {5}, 5, DeviationTarget.ANY), new int[] {5});
  }

  /**
   * 打った手を除いた台の q は和が 1 で、台の中の順位と、一様に混ぜる ε_q だけで決まる（除いた手は引かれない）。打った手の s が最大でも、
   * 残りで s が最大の手が順位 0 になる。
   */
  @Test
  public void otherSupportIsADistributionOverTheRest() {
    int[] candidates = {0, 2, 5};
    float[] scores = new float[6];
    scores[0] = 0.4f;
    scores[2] = 0.9f;
    scores[5] = 0.1f;
    int[] support = RollIn.support(candidates, 2, DeviationTarget.OTHER);
    double[] q = RollIn.distribution(support, scores, null, 0.2);
    Assert.assertEquals(q[0] + q[1], 1.0, 1e-12);
    Assert.assertEquals(q[0], 0.8 / 1.5 + 0.1, 1e-12);
    Assert.assertEquals(q[1], 0.8 * 0.5 / 1.5 + 0.1, 1e-12);
    for (double u : new double[] {0, 0.3, 0.6, 0.999999}) {
      RollIn.Deviation d = RollIn.draw(support, scores, null, 0.2, u);
      Assert.assertNotEquals(d.action(), 2);
      Assert.assertEquals(d.probability(), (float) (d.action() == 0 ? q[0] : q[1]));
    }
    // ε_q = 1 なら台の上の一様分布になる
    for (double value : RollIn.distribution(support, scores, null, 1.0)) Assert.assertEquals(value, 0.5);
  }

  /** 決まりの値の範囲と、後の判断のクラスごとの ρ_br の表（クラスごとに 1 つ。作った後に元の配列を変えても変わらない）。 */
  @Test
  public void rulesCheckTheirValues() {
    double[] later = new double[DecisionClass.COUNT];
    later[DecisionClass.CALL.ordinal()] = 0.7;
    RollInRule rule = new RollInRule(0.15, 0.1, DeviationTarget.OTHER, later, 4);
    later[DecisionClass.CALL.ordinal()] = 0.1;
    Assert.assertEquals(rule.laterRate(DecisionClass.CALL), 0.7);
    Assert.assertEquals(rule.laterRate(DecisionClass.THREAT), 0.0);
    RollInRule uniform = RollInRule.uniform(0.03, 0.1, DeviationTarget.ANY, 0.3, 2);
    for (DecisionClass c : DecisionClass.values()) Assert.assertEquals(uniform.laterRate(c), 0.3);
    Assert.assertEquals(RollInRule.OFF.rate(), 0.0);
    Assert.assertEquals(RollInRule.OFF.maxLabels(), 0);
    Assert.expectThrows(IllegalArgumentException.class, () -> RollInRule.uniform(1.5, 0.1, DeviationTarget.ANY, 0.3, 2));
    Assert.expectThrows(IllegalArgumentException.class, () -> RollInRule.uniform(0.1, -0.1, DeviationTarget.ANY, 0.3, 2));
    Assert.expectThrows(IllegalArgumentException.class, () -> RollInRule.uniform(0.1, 0.1, DeviationTarget.ANY, Double.NaN, 2));
    Assert.expectThrows(IllegalArgumentException.class, () -> RollInRule.uniform(0.1, 0.1, DeviationTarget.ANY, 0.3, -1));
    Assert.expectThrows(NullPointerException.class, () -> RollInRule.uniform(0.1, 0.1, null, 0.3, 2));
    Assert.expectThrows(
        IllegalArgumentException.class, () -> new RollInRule(0.1, 0.1, DeviationTarget.ANY, new double[3], 2));
  }

  /** 収集の率の表は、手の形ごとにも根のクラスごとにも、クラスの数だけ要る。 */
  @Test
  public void samplingNeedsATablePerHandAndRoot() {
    RollInRule[] rules = new RollInRule[DecisionClass.COUNT];
    Arrays.fill(rules, RollInRule.OFF);
    double[][] rates = new double[Openness.COUNT][DecisionClass.COUNT];
    boolean[] measureAll = new boolean[DecisionClass.COUNT];
    new Sampling(0, rates, measureAll, 2, 0, rules);
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> new Sampling(0, new double[1][DecisionClass.COUNT], measureAll, 2, 0, rules));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> new Sampling(0, new double[][] {new double[DecisionClass.COUNT], new double[2]}, measureAll, 2, 0, rules));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> new Sampling(0, rates, measureAll, 2, 0, new RollInRule[DecisionClass.COUNT - 1]));
    RollInRule[] missing = rules.clone();
    missing[DecisionClass.KAN.ordinal()] = null;
    Assert.expectThrows(NullPointerException.class, () -> new Sampling(0, rates, measureAll, 2, 0, missing));
    double[][] negative = new double[Openness.COUNT][DecisionClass.COUNT];
    negative[Openness.OPEN.ordinal()][0] = -0.1;
    Assert.expectThrows(IllegalArgumentException.class, () -> new Sampling(0, negative, measureAll, 2, 0, rules));
  }

  /** 根のクラスを持つのは写しの件数だけ。 */
  @Test
  public void onlyCopiesCountARootClass() {
    new DecisionCounts(Role.COPY, DecisionClass.CALL);
    new DecisionCounts(Role.MAIN, null);
    new DecisionCounts(Role.FORK, null);
    Assert.expectThrows(IllegalArgumentException.class, () -> new DecisionCounts(Role.COPY, null));
    Assert.expectThrows(IllegalArgumentException.class, () -> new DecisionCounts(Role.MAIN, DecisionClass.CALL));
    Assert.expectThrows(IllegalArgumentException.class, () -> new DecisionCounts(Role.FORK, DecisionClass.CALL));
  }

  /** 写しの件数は、逸脱した席の判断を根のクラスの列に、全ての判断を写しの判断の列に足す。主経路の件数は、手の形とクラスの列に足す。 */
  @Test
  public void countsGoToTheirColumns() {
    CollectionMetrics metrics = new CollectionMetrics();
    DecisionCounts copy = new DecisionCounts(Role.COPY, DecisionClass.CALL);
    copy.countModel();
    copy.countModel();
    copy.countDeviator();
    copy.flush(metrics);
    DecisionCounts main = new DecisionCounts(Role.MAIN, null);
    main.countMain(DecisionClass.THREAT, Openness.OPEN);
    main.countMain(DecisionClass.THREAT, Openness.CLOSED);
    main.countMain(DecisionClass.THREAT, Openness.OPEN);
    main.flush(metrics);
    main.flush(metrics);
    Assert.assertEquals(metrics.copyDecisions.sum(), 2);
    Assert.assertEquals(metrics.deviatorDecisions[DecisionClass.CALL.ordinal()].sum(), 1);
    Assert.assertEquals(metrics.mainDecisions[CollectionMetrics.byHand(Openness.OPEN, DecisionClass.THREAT)].sum(), 2);
    Assert.assertEquals(metrics.mainDecisions[CollectionMetrics.byHand(Openness.CLOSED, DecisionClass.THREAT)].sum(), 1);
  }
}
