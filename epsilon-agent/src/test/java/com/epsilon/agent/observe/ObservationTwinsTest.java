package com.epsilon.agent.observe;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.ActionSection;
import com.epsilon.mahjong.tile.Tiles;
import org.testng.annotations.Test;

/**
 * 同値な打牌の決まりが、全ての手の組で PRISM の候補の決まりと一致することを確かめる。PRISM は、候補の集合・共通乱数・逸脱の数え方の
 * どれでも、同じ手か、種類（打牌か立直の打牌）・牌・赤の選び方が同じ打牌を 1 つの候補と数える。
 */
public class ObservationTwinsTest {

  /**
   * 同値になる手の組の数（順序つきで、同じ手どうしは除く）。打牌と立直の打牌のそれぞれに、34 種類と赤の 5 の 3 色の分だけ、
   * 手出しとツモ切りの組がある。
   */
  private static final int TWIN_PAIRS = 2 * 2 * (Tiles.KINDS + Tiles.NUMBER_SUITS);

  @Test
  public void twinsFollowTheCandidateRuleOnEveryPair() {
    int pairs = 0;
    for (int first = 0; first < ActionSection.SPACE; first++) {
      Action action = Action.of(first);
      for (int second = 0; second < ActionSection.SPACE; second++) {
        Action other = Action.of(second);
        boolean twins = Observation.twins(action, other);
        assertEquals(action == other || twins, sameCandidate(action, other), action + " / " + other);
        if (twins && action != other) {
          pairs++;
        }
      }
    }
    assertEquals(pairs, TWIN_PAIRS);
  }

  @Test
  public void onlyTheTsumogiriMayDiffer() {
    Action handFive = Action.of(Action.discard(Tiles.M5, false, false, false));
    assertTrue(Observation.twins(handFive, Action.of(Action.discard(Tiles.M5, false, true, false))));
    assertTrue(Observation.twins(handFive, handFive));
    assertFalse(Observation.twins(handFive, Action.of(Action.discard(Tiles.M5, true, true, false))));
    assertFalse(Observation.twins(handFive, Action.of(Action.discard(Tiles.M5, false, true, true))));
    assertFalse(Observation.twins(handFive, Action.of(Action.discard(Tiles.M6, false, true, false))));
    Action redRiichi = Action.of(Action.discard(Tiles.P5, true, false, true));
    assertTrue(Observation.twins(redRiichi, Action.of(Action.discard(Tiles.P5, true, true, true))));
    Action pon = Action.of(Action.code(Action.Type.PON, Tiles.M5, Action.TileSelection.NON_AKA));
    assertFalse(Observation.twins(pon, pon));
  }

  /** PRISM の候補の決まり。同じ手か、種類（打牌か立直の打牌）・牌・赤の選び方が同じ打牌なら、同じ候補になる。 */
  private static boolean sameCandidate(Action x, Action y) {
    if (x.equals(y)) {
      return true;
    }
    boolean discard = x.type() == Action.Type.DAHAI || x.type() == Action.Type.RIICHI_DAHAI;
    return discard && x.type() == y.type() && x.tile() == y.tile() && x.selection().red() == y.selection().red();
  }
}
