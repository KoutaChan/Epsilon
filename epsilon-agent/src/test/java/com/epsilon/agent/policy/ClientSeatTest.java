package com.epsilon.agent.policy;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.RandomPlay;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

/** {@link ClientSeat} は判断する席ごとに、その席から見た局面を方策に渡し、方策が貪欲に選んだ手（合法手の何番目か）をそのまま返す。 */
public class ClientSeatTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final int STEPS = 5_000;

  @Test
  public void returnsTheGreedyMoveForTheSlotsObservation() {
    LastLegal policy = new LastLegal();
    ClientSeat seat = new ClientSeat(policy);
    RandomPlay play = new RandomPlay(VARIANT, 31);
    int decided = 0;
    for (int step = 0; step < STEPS; step++) {
      if (play.boundary() == Boundary.DECISIONS) {
        Decisions decisions = play.engine.decisions();
        for (int slot = 0; slot < decisions.size(); slot++) {
          int observer = decisions.seat(slot);
          int chosen = seat.decide(play.engine.view(observer), decisions, slot);
          Assert.assertEquals(chosen, decisions.legalCount(slot) - 1);
          Assert.assertEquals(policy.observer, observer);
          decided++;
        }
      }
      play.step();
    }
    Assert.assertTrue(decided > 0);
  }

  /** いつも最後の合法手を選ぶ方策。最後に評価した局面がどの席から見たものかを覚えておく。 */
  private static final class LastLegal implements InlinePolicy {

    private int observer = -1;

    @Override
    public Variant variant() {
      return VARIANT;
    }

    @Override
    public String label() {
      return "last-legal";
    }

    @Override
    public int act(Observation observation) {
      return observation.legalActions() - 1;
    }

    @Override
    public Evaluation evaluate(Observation observation) {
      observer = observation.observer();
      float[] probabilities = new float[observation.legalActions()];
      Arrays.fill(probabilities, 1f / probabilities.length);
      return new Evaluation(probabilities, null, act(observation));
    }

    @Override
    public void close() {}
  }
}
