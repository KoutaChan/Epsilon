package com.epsilon.prism.collect;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.prism.core.Threats;
import com.epsilon.prism.core.Twins;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 候補集合 C(o)（{@link CandidateSet#build}）は観測・π_t の s・p0・ハッシュの硬貨だけで決まり、隠れた牌（相手の手牌と山）に依らない（論文 命題 6
 * の仮定）。優先の順・ε_C の 1 手・打った手の双子の置き換えも確かめる。
 */
public class CandidateSetTest {
  private static final int M1 = 0, M2 = 1, M4 = 3, M5 = 4, M6 = 5, M7 = 6, M9 = 8;
  private static final int P2 = 10, S2 = 19, S5 = 22, TON = 27, HAKU = 31;

  /**
   * 席 1 が 4m・2p を切って立直し、立直後に席 2 が 5s を通した局面（公開情報は wallSeed に依らない）。席 0 の手牌は同じで、席 2・3 の伏せた手牌は
   * hidden で変える。
   */
  private static GameState state(long wallSeed, int hidden) {
    GameState state = new GameState(wallSeed);
    state.commitDiscard(1, M4, false);
    state.commitDiscard(1, P2, false);
    state.setRiichi(1, true);
    state.commitDiscard(2, S5, false);
    for (int tile : new int[] {TON, TON, HAKU, M1, M2, M5, M6, M7, M9, S2, S2, S2, P2})
      state.hand(0).add(tile);
    for (int i = 0; i < 13; i++) {
      state.hand(2).add((hidden + 3 * i) % 34);
      state.hand(3).add((hidden * 7 + 5 * i) % 34);
    }
    return state;
  }

  private static final List<Action> RIICHI_LEGAL =
      List.of(
          Action.dahai(M5),
          Action.dahai(TON),
          Action.dahai(P2),
          Action.riichiDahai(M1),
          Action.dahai(M1),
          Action.dahai(M9));
  private static final float[] RIICHI_SCORES = {0.10f, 0.00f, -0.20f, 0.30f, 0.05f, -0.10f};
  private static final float[] RIICHI_PRIOR = {0.30f, 0.10f, 0.10f, 0.20f, 0.20f, 0.10f};

  private static CandidateSet riichi(GameState state, int played, double exploreCoin) {
    return CandidateSet.build(
        state,
        0,
        RIICHI_LEGAL,
        RIICHI_SCORES,
        RIICHI_PRIOR,
        Threats.count(state, 0),
        played,
        6,
        0.1,
        exploreCoin,
        0.5);
  }

  /** 全ての合法手の候補集合: 打った手を含み、どの合法手もどれかの候補の双子で、測る手は候補そのもの。 */
  @Test
  public void allActionsCoversEveryLegalAction() {
    for (int played = 0; played < RIICHI_LEGAL.size(); played++) {
      CandidateSet all = CandidateSet.allActions(RIICHI_LEGAL, played);
      int playedAction = played;
      Assert.assertTrue(Arrays.stream(all.support()).anyMatch(a -> a == playedAction));
      Assert.assertEquals(all.measured(), all.support());
      for (int a = 0; a < RIICHI_LEGAL.size(); a++) {
        int action = a;
        Assert.assertTrue(
            Arrays.stream(all.support()).anyMatch(c -> Twins.same(RIICHI_LEGAL, c, action)),
            "action " + a);
      }
      Assert.assertFalse(all.explored());
    }
  }

  /**
   * s の 1 位（立直の 3）→ 体制の代表（立直 3・ダマ 0・効率 0・安全 2 の現物 2p）→ p0 の 1 位（0）→ s の 2 位（0）→ p0 の 2 位（3）→ s の 3 位（4）。
   * 隠れた牌と山を変えても、同じ入力なら何度作っても同じ C。
   */
  @Test
  public void dependsOnlyOnTheObservationScoresPriorAndCoins() {
    GameState first = state(17, 1), second = state(99, 11);
    Assert.assertEquals(Threats.count(first, 0), 1);
    Assert.assertEquals(Threats.count(second, 0), 1);
    CandidateSet built = riichi(first, 0, 0.9);
    Assert.assertEquals(built.support(), new int[] {0, 2, 3, 4});
    Assert.assertEquals(built.measured(), new int[] {0, 2, 3, 4});
    Assert.assertFalse(built.explored());
    Assert.assertTrue(built.playedInside(0));
    for (CandidateSet again : new CandidateSet[] {riichi(first, 0, 0.9), riichi(second, 0, 0.9)}) {
      Assert.assertEquals(again.support(), built.support());
      Assert.assertEquals(again.measured(), built.measured());
      Assert.assertEquals(again.explored(), built.explored());
    }
    // 床 ε_b で C の外の手を打っても C は同じで、測る手に打った手が加わる
    CandidateSet outside = riichi(second, 5, 0.9);
    Assert.assertEquals(outside.support(), built.support());
    Assert.assertEquals(outside.measured(), new int[] {0, 2, 3, 4, 5});
    Assert.assertFalse(outside.playedInside(5));
    Assert.assertEquals(outside.supportBytes(), new byte[] {0, 2, 3, 4});
  }

  /** ε_C の硬貨が当たれば上限を 1 枠空け、ここまでの手と双子でない手から floor(draw × 数) 番目を 1 手足す。 */
  @Test
  public void exploreCoinAddsOneUniformAction() {
    GameState state = new GameState(5);
    List<Action> legal =
        List.of(
            Action.dahai(M1),
            Action.dahai(M2),
            Action.dahai(M4),
            Action.dahai(M5),
            Action.dahai(M6),
            Action.dahai(M7));
    float[] scores = {0.5f, 0.4f, 0.3f, 0.2f, 0.1f, 0.0f};
    float[] prior = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.5f};
    CandidateSet plain = CandidateSet.build(state, 0, legal, scores, prior, 0, 2, 3, 0.1, 0.9, 0.6);
    Assert.assertEquals(plain.support(), new int[] {0, 1, 5});
    Assert.assertFalse(plain.explored());
    // 上限 2 で {0, 5}、残りの {1, 2, 3, 4} から floor(0.6 × 4) = 2 番目の 3
    CandidateSet explored =
        CandidateSet.build(state, 0, legal, scores, prior, 0, 2, 3, 0.1, 0.05, 0.6);
    Assert.assertEquals(explored.support(), new int[] {0, 3, 5});
    Assert.assertEquals(explored.measured(), new int[] {0, 2, 3, 5});
    Assert.assertTrue(explored.explored());
    Assert.assertFalse(explored.playedInside(2));
  }

  /** 手出しとツモ切りの双子は 1 つの候補で、打った手がその双子なら候補を打った手に置き換える（分岐が要らない）。 */
  @Test
  public void twinOfThePlayedActionReplacesTheCandidate() {
    GameState state = new GameState(5);
    List<Action> legal =
        List.of(Action.dahai(M5, false, false), Action.dahai(M5, false, true), Action.dahai(M1));
    float[] scores = {0.30f, 0.29f, 0.00f};
    float[] prior = {0.4f, 0.4f, 0.2f};
    CandidateSet tedashi = CandidateSet.build(state, 0, legal, scores, prior, 0, 0, 6, 0.1, 0.9, 0);
    Assert.assertEquals(tedashi.support(), new int[] {0, 2});
    CandidateSet tsumogiri =
        CandidateSet.build(state, 0, legal, scores, prior, 0, 1, 6, 0.1, 0.9, 0);
    Assert.assertEquals(tsumogiri.support(), new int[] {1, 2});
    Assert.assertEquals(tsumogiri.measured(), new int[] {1, 2});
    Assert.assertTrue(tsumogiri.playedInside(1));
  }

  /**
   * 順位は値の降順で、同点は割る値（s の順位なら p0）の大きい手、それも同じなら番号の小さい手が上（方策の貪欲の argmax と同じ比べ方）。白紙の網
   * （s ≡ 0）の s の順位は p0 の順になる。
   */
  @Test
  public void rankingBreaksTiesByPriorThenIndex() {
    Assert.assertEquals(
        CandidateSet.ranking(new int[] {0, 1, 2, 3}, new float[] {0.2f, 0.5f, 0.5f, -1f}),
        new int[] {1, 2, 0, 3});
    // actions の中の添字を返す
    Assert.assertEquals(
        CandidateSet.ranking(new int[] {4, 1}, new float[] {0f, 0.3f, 0f, 0f, 0.7f}),
        new int[] {0, 1});
    Assert.assertEquals(
        CandidateSet.ranking(
            new int[] {0, 1, 2, 3}, new float[4], new float[] {0.1f, 0.4f, 0.1f, 0.4f}),
        new int[] {1, 3, 0, 2});
    Assert.assertEquals(
        com.epsilon.prism.core.Greedy.argmax(new float[4], new float[] {0.1f, 0.2f, 0.4f, 0.3f}, 4),
        2);
  }
}
