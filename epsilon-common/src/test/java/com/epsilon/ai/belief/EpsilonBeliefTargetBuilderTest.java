package com.epsilon.ai.belief;

import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import org.testng.Assert;
import org.testng.annotations.Test;

/** Beliefの教師データが観測者を基準とする席順に並び、元の局面を変更しても変わらないことを検証する。 */
public class EpsilonBeliefTargetBuilderTest {

  @Test
  public void teacherUsesObserverRelativeSeatsAndKeepsAnIndependentSnapshot() {
    GameState state = new GameState(17);
    hand(state, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 18);
    hand(state, 3, 0, 1, 2, 9, 10, 11, 18, 19, 20, 27, 27, 27, 28);
    hand(state, 0, 3, 4, 5, 12, 13, 14, 21, 22, 23, 28, 28, 28, 29);
    hand(state, 1, 6, 7, 8, 15, 16, 17, 24, 25, 26, 29, 29, 29, 30);
    EpsilonBeliefTarget target = new EpsilonBeliefTargetBuilder().build(state, 2);
    state.hand(3).clear();

    Assert.assertEquals(target.opponentHandCounts()[27], 3.0f);
    Assert.assertEquals(target.opponentHandCounts()[34 + 28], 3.0f);
    Assert.assertEquals(target.opponentHandCounts()[68 + 29], 3.0f);
    Assert.assertEquals(target.hiddenTileCounts()[0], 3.0f);
    Assert.assertEquals(target.hiddenTileCounts()[27], 4.0f);
    Assert.assertEquals(target.opponentShanten(), new float[] {0, 0, 0});
    Assert.assertEquals(target.opponentTenpai(), new float[] {1, 1, 1});
    float[] waits = new float[102];
    waits[28] = 1;
    waits[34 + 29] = 1;
    waits[68 + 30] = 1;
    Assert.assertEquals(target.opponentWaitMask(), waits);
  }

  private static void hand(GameState state, int seat, int... tiles) {
    Hand hand = state.hand(seat);
    for (int tile : tiles) {
      hand.add(tile);
    }
  }
}
