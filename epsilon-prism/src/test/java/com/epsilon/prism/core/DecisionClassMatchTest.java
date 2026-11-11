package com.epsilon.prism.core;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.ObservedDecisions;
import com.epsilon.prism.TableWalk;
import java.util.EnumSet;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 判断の種別とクラスは、どの判断でも、卓の席から見た値（収集の側）と O から求めた値（学習・評価の側）で一致する。合法手の並びも一致する。 */
public class DecisionClassMatchTest {

  private static final Variant FOUR = Variant.TENHOU_4P;

  @Test
  public void seatViewAndObservationGiveTheSameClass() {
    SeatFacts facts = new SeatFacts(FOUR);
    EnumSet<Kind> kinds = EnumSet.noneOf(Kind.class);
    EnumSet<DecisionClass> classes = EnumSet.noneOf(DecisionClass.class);
    for (TableWalk.Policy policy : TableWalk.Policy.values()) {
      for (ObservedDecisions.Decision decision :
          ObservedDecisions.collect(71 + policy.ordinal(), 3, policy, Integer.MAX_VALUE)) {
        Observation o = decision.observation();
        int[] legal = new int[o.legalActions()];
        for (int a = 0; a < legal.length; a++) legal[a] = o.action(a).code();
        Assert.assertEquals(legal, decision.legal());
        Assert.assertEquals(o.decisionKind(), decision.kind());
        Assert.assertEquals(o.kyoku(), decision.round());
        Assert.assertEquals(Stage.of(FOUR, o.remainingWall()), Stage.of(FOUR, decision.liveLeft()));
        facts.load(o);
        Kind table = Kind.classify(decision.kind(), decision.legal());
        Kind observed = Kind.classify(o.decisionKind(), legal);
        Assert.assertEquals(observed, table);
        DecisionClass expected =
            DecisionClass.of(FOUR, table, Stage.of(FOUR, decision.liveLeft()), decision.threats(), decision.round());
        Assert.assertEquals(
            DecisionClass.of(FOUR, observed, Stage.of(FOUR, o.remainingWall()), Threats.count(o, facts), o.kyoku()),
            expected);
        kinds.add(table);
        classes.add(expected);
      }
    }
    Assert.assertTrue(kinds.size() >= 6, "kinds " + kinds);
    Assert.assertTrue(classes.size() >= 7, "classes " + classes);
  }
}
