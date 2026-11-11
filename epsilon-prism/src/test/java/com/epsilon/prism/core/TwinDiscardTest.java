package com.epsilon.prism.core;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.TableWalk;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 同値な打牌の切り方（{@link TwinDiscard}）: TSUMOGIRI はツモ牌と同じ種類・同じ赤の選び方の手牌を持つときにツモ牌を、TEDASHI は手牌を切る。
 * 打牌でない手、片方しかない打牌、赤の選び方の違う打牌は替えない。立直の打牌も同じで、種類（打牌か立直の打牌か）は変わらない。
 */
public class TwinDiscardTest {
  private static final int M5 = Tiles.M5, M6 = Tiles.M6, P5 = Tiles.P5, S3 = Tiles.S3;

  private static int hand(int kind, boolean red, boolean riichi) {
    return Action.discard(kind, red, false, riichi);
  }

  private static int drawn(int kind, boolean red, boolean riichi) {
    return Action.discard(kind, red, true, riichi);
  }

  /** ツモ牌が赤でない 5m で、手牌に赤の 5m と赤でない 5m がある手番。赤でない 5m だけが手出しとツモ切りの両方で切れる。 */
  @Test
  public void theDrawnTileOrTheHandTileOfTheSameKindAndRedIsPlayed() {
    int[] legal = {hand(M5, true, false), hand(M5, false, false), drawn(M5, false, false), hand(M6, false, false)};
    Assert.assertEquals(play(TwinDiscard.TSUMOGIRI, legal), new int[] {0, 2, 2, 3});
    Assert.assertEquals(play(TwinDiscard.TEDASHI, legal), new int[] {0, 1, 1, 3});
    Assert.assertEquals(TwinDiscard.TSUMOGIRI.play(legal, new int[] {1, 0, 3}), new int[] {0, 2, 3});
    Assert.assertEquals(TwinDiscard.TEDASHI.play(legal, new int[] {3, 2}), new int[] {1, 3});
  }

  /** ツモ牌が赤の 5p で手牌が赤でない 5p なら、赤の選び方が違うので替えない。鳴いた後の打牌はツモ切りが無いので替えない。 */
  @Test
  public void redChoicesAndDiscardsAfterACallNeverChange() {
    int[] red = {hand(P5, false, false), drawn(P5, true, false)};
    int[] afterCall = {hand(M5, false, false), hand(M6, false, false)};
    for (TwinDiscard twins : TwinDiscard.values()) {
      Assert.assertEquals(play(twins, red), new int[] {0, 1}, twins.name());
      Assert.assertEquals(play(twins, afterCall), new int[] {0, 1}, twins.name());
    }
  }

  /** 立直の打牌も、立直の打牌どうしで替える。ツモ和了・暗槓・応答の手は替えない。 */
  @Test
  public void riichiDiscardsStayRiichiAndOtherActionsNeverChange() {
    int[] legal = {
      Action.code(Action.Type.TSUMO_AGARI),
      Action.code(Action.Type.ANKAN, M6, Action.TileSelection.NOT_APPLICABLE),
      hand(S3, false, true),
      drawn(S3, false, true),
      hand(S3, false, false),
      drawn(S3, false, false)
    };
    Assert.assertEquals(play(TwinDiscard.TSUMOGIRI, legal), new int[] {0, 1, 3, 3, 5, 5});
    Assert.assertEquals(play(TwinDiscard.TEDASHI, legal), new int[] {0, 1, 2, 2, 4, 4});
    int[] response = {
      Action.code(Action.Type.PASS),
      Action.code(Action.Type.PON, S3, Action.TileSelection.NON_AKA),
      Action.code(Action.Type.RON_AGARI)
    };
    for (TwinDiscard twins : TwinDiscard.values()) Assert.assertEquals(play(twins, response), new int[] {0, 1, 2});
  }

  /** 新しいエンジンで打った四人打ちの判断で、手牌にツモ牌と同じ種類がある打牌を確かめる（立直の打牌を含む）。 */
  @Test
  public void fourPlayerTurnsPlayTheChosenTwin() {
    assertWalk(Variant.TENHOU_4P, 41);
  }

  /** 三人打ちでも同じ。 */
  @Test
  public void threePlayerTurnsPlayTheChosenTwin() {
    assertWalk(Variant.TENHOU_3P, 43);
  }

  /** 全ての合法手を、それぞれの切り方で替えた位置。 */
  private static int[] play(TwinDiscard twins, int[] legal) {
    int[] played = new int[legal.length];
    for (int a = 0; a < legal.length; a++) played[a] = twins.play(legal, a);
    return played;
  }

  /**
   * 種の決まった半荘（全ての打ち方。立直しない打ち方の黙聴が、立直の打牌の手出しとツモ切りを作る）の全ての判断の全ての合法手で: 替えた手は
   * 同じ手か同値な打牌で、種類・牌・赤の選び方は変わらない。手出しとツモ切りの両方で切れる打牌は、ツモ牌と同じ種類・同じ赤の選び方の手牌を
   * 持つ手番の打牌で、TSUMOGIRI はツモ切りに、TEDASHI は手出しに替わる。ほかの手は替わらない。打牌と立直の打牌の両方で、両方で切れる判断が
   * 出る。
   */
  private static void assertWalk(Variant variant, long seed) {
    ObservationWriter writer = new ObservationWriter(variant);
    ObservationBatch batch = new ObservationBatch(variant, 1 << 12);
    Observation o = new Observation(variant);
    int[] twins = new int[2];
    for (TableWalk.Policy policy : TableWalk.Policy.values()) {
      new TableWalk(variant, policy)
          .run(
              seed,
              30,
              (engine, slot) -> {
                Decisions decisions = engine.decisions();
                int n = decisions.legalCount(slot);
                int[] legal = new int[n];
                for (int a = 0; a < n; a++) legal[a] = decisions.legal(slot, a);
                batch.clear();
                o.bind(batch, writer.capture(engine.view(decisions.seat(slot)), decisions, slot, batch));
                for (int a = 0; a < n; a++) {
                  Action action = Action.of(legal[a]);
                  int tsumogiri = TwinDiscard.TSUMOGIRI.play(legal, a), tedashi = TwinDiscard.TEDASHI.play(legal, a);
                  for (int played : new int[] {tsumogiri, tedashi}) {
                    Action twin = Action.of(legal[played]);
                    Assert.assertTrue(Twins.same(legal, a, played), action + " -> " + twin);
                    Assert.assertEquals(twin.type(), action.type());
                    Assert.assertEquals(twin.tile(), action.tile());
                    Assert.assertEquals(twin.selection().red(), action.selection().red());
                  }
                  if (!Twins.discard(action)) {
                    Assert.assertEquals(tsumogiri, a, action.toString());
                    Assert.assertEquals(tedashi, a, action.toString());
                    continue;
                  }
                  boolean riichi = action.type() == Action.Type.RIICHI_DAHAI;
                  boolean red = action.selection().red();
                  int fromHand = indexOf(legal, hand(action.tile(), red, riichi));
                  int fromDraw = indexOf(legal, drawn(action.tile(), red, riichi));
                  if (fromHand < 0 || fromDraw < 0) {
                    Assert.assertEquals(tsumogiri, a, action.toString());
                    Assert.assertEquals(tedashi, a, action.toString());
                    continue;
                  }
                  Assert.assertEquals(o.drawnTile(), action.tile(), action.toString());
                  Assert.assertEquals(o.lastEventAka(), red, action.toString());
                  Assert.assertTrue(o.handCount(action.tile()) >= 2, action.toString());
                  Assert.assertEquals(tsumogiri, fromDraw, action.toString());
                  Assert.assertEquals(tedashi, fromHand, action.toString());
                  twins[riichi ? 1 : 0]++;
                }
              });
    }
    Assert.assertTrue(twins[0] > 0 && twins[1] > 0, variant + " discards " + twins[0] + " riichi " + twins[1]);
  }

  private static int indexOf(int[] legal, int code) {
    for (int a = 0; a < legal.length; a++) if (legal[a] == code) return a;
    return -1;
  }
}
