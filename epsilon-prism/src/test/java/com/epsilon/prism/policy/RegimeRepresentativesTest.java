package com.epsilon.prism.policy;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.prism.core.Threats;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 公開情報だけの安全度と体制の代表（当てはまる体制の両側を必ず返す）。 */
public class RegimeRepresentativesTest {
  private static final int M1 = 0, M2 = 1, M4 = 3, M5 = 4, M6 = 5, M7 = 6, M9 = 8;
  private static final int P2 = 10, S2 = 19, S5 = 22, TON = 27, HAKU = 31;

  /** 席 1 が 4m・2p を切って立直し、立直後に席 2 が 5s を通した局面。席 0 は東を 2 枚、白を 1 枚持つ。 */
  private static GameState riichiState() {
    GameState state = new GameState(17);
    state.commitDiscard(1, M4, false);
    state.commitDiscard(1, P2, false);
    state.setRiichi(1, true);
    state.commitDiscard(2, S5, false);
    for (int tile : new int[] {TON, TON, HAKU, M1, M2, M5, M6, M7, M9, S2, S2, S2, P2})
      state.hand(0).add(tile);
    return state;
  }

  @Test
  public void dangerClassesFollowGenbutsuSujiAndVisibleHonours() {
    GameState state = riichiState();
    Assert.assertEquals(PublicSafety.dangerClass(state, 0, 1, M4), 0);
    Assert.assertEquals(PublicSafety.dangerClass(state, 0, 1, P2), 0);
    Assert.assertEquals(PublicSafety.dangerClass(state, 0, 1, S5), 0, "通った牌は立直者の現物");
    Assert.assertEquals(PublicSafety.dangerClass(state, 0, 1, M1), 1);
    Assert.assertEquals(PublicSafety.dangerClass(state, 0, 1, M7), 2);
    Assert.assertEquals(PublicSafety.dangerClass(state, 0, 1, S2), 2, "立直後に通った 5s の筋");
    Assert.assertEquals(PublicSafety.dangerClass(state, 0, 1, M9), 2);
    Assert.assertEquals(PublicSafety.dangerClass(state, 0, 1, TON), 2);
    Assert.assertEquals(PublicSafety.dangerClass(state, 0, 1, HAKU), 3);
    Assert.assertEquals(PublicSafety.dangerClass(state, 0, 1, M2), 4);
    Assert.assertEquals(PublicSafety.dangerClass(state, 0, 1, M5), 5);
    Assert.assertEquals(PublicSafety.dangerClass(state, 0, 1, M6), 5);
  }

  @Test
  public void riichiAndThreatRegimesReturnBothSides() {
    GameState state = riichiState();
    List<Action> legal =
        List.of(
            Action.dahai(M5),
            Action.dahai(TON),
            Action.dahai(P2),
            Action.riichiDahai(M1),
            Action.dahai(M1));
    float[] scores = {0.1f, 0f, -0.2f, 0.3f, 0.05f};
    float[] prior = {0.4f, 0.1f, 0.1f, 0.2f, 0.2f};
    int threats = Threats.count(state, 0);
    Assert.assertEquals(threats, 1);
    Assert.assertEquals(PublicSafety.safest(state, 0, legal, prior), 2);
    Assert.assertEquals(
        RegimeRepresentatives.of(state, 0, legal, scores, prior, threats), new int[] {3, 0, 0, 2});
  }

  @Test
  public void callAndKanRegimes() {
    GameState state = new GameState(17);
    List<Action> calls = List.of(Action.chiSequence(M1, M2), Action.pon(M2), Action.pass());
    Assert.assertEquals(
        RegimeRepresentatives.of(state, 0, calls, new float[] {0.2f, 0.5f, 0.1f}, null, 0),
        new int[] {1, 2});
    List<Action> kans =
        List.of(Action.ankan(M4), Action.dahai(M4), Action.dahai(M6), Action.tsumoAgari());
    Assert.assertEquals(
        RegimeRepresentatives.of(state, 0, kans, new float[] {0.1f, 0.4f, 0.2f, 0.9f}, null, 0),
        new int[] {0, 3});
    List<Action> plain = List.of(Action.dahai(M1), Action.dahai(M2));
    Assert.assertEquals(
        RegimeRepresentatives.of(state, 0, plain, new float[] {0f, 1f}, null, 0), new int[0]);
  }

  @Test
  public void safestFallsBackToPriorWithoutThreats() {
    GameState state = new GameState(17);
    List<Action> legal = List.of(Action.dahai(M1), Action.dahai(M5), Action.pass());
    Assert.assertEquals(PublicSafety.safest(state, 0, legal, new float[] {0.2f, 0.7f, 0.1f}), 1);
    Assert.assertEquals(
        PublicSafety.safest(state, 0, List.of(Action.pass()), new float[] {1f}), -1);
  }
}
