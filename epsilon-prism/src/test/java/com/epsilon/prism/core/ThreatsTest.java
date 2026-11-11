package com.epsilon.prism.core;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.ObservedDecisions;
import com.epsilon.prism.TableWalk;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 公開の情報で数えた脅威の数は、どの判断でも、卓の席から見て数えても、記録 O から求めて数えても同じになる。 */
public class ThreatsTest {

  @Test
  public void seatViewAndObservationAgreeAtEveryDecision() {
    SeatFacts facts = new SeatFacts(Variant.TENHOU_4P);
    int threatened = 0, several = 0;
    for (TableWalk.Policy policy : TableWalk.Policy.values()) {
      for (ObservedDecisions.Decision decision :
          ObservedDecisions.collect(31 + policy.ordinal(), 3, policy, Integer.MAX_VALUE)) {
        Observation o = decision.observation();
        facts.load(o);
        Assert.assertEquals(Threats.count(o, facts), decision.threats());
        Assert.assertFalse(Threats.isThreat(o, facts, 0), "the observer is never a threat");
        if (decision.threats() > 0) threatened++;
        if (decision.threats() > 1) several++;
      }
    }
    Assert.assertTrue(threatened > 100, "threatened " + threatened);
    Assert.assertTrue(several > 10, "several " + several);
  }
}
