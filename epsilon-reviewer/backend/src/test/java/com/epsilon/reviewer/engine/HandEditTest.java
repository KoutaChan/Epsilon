package com.epsilon.reviewer.engine;

import static com.epsilon.reviewer.engine.TestEdits.hand;

import com.epsilon.core.Action;
import com.epsilon.core.Tile;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.StartGame;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.scenario.HandEdit;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.engine.scenario.ScenarioEdits;
import com.epsilon.reviewer.engine.scenario.ScenarioEngine;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.engine.view.TileCode;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 判断の席の手牌の入れ替えが、席の目の再生を変えた配牌とツモで作り直し、手と合法な打牌・前の窓の振聴・鳴いた後の打牌を変えること、同じ牌は
 * どの位置を指しても同じ 1 枚を替えること、リーチの後と見える牌の上限を断ることを確かめる。
 */
public class HandEditTest {
  private static final int[] SCORES = {25000, 25000, 25000, 25000};

  /**
   * 席 0（親）は 123m456m789m123p と E の単騎。席 1 が 9s をツモ切りし、席 0 が次にツモった 7m の打牌が判断（出来事 11）。E を 9s に替えると、
   * 席 1 の 9s で一気通貫のロンができた窓を見送ったので判断の時点で同巡の振聴になる。
   */
  private static final LoadedRecord FURITEN_ROUND =
      round(
          new int[][] {
            {0, 4, 8, 12, 17, 20, 24, 28, 32, 36, 40, 44, 108},
            {112, 113, 114, 116, 117, 118, 120, 48, 56, 64, 68, 72, 89},
            {128, 129, 130, 132, 133, 134, 121, 60, 69, 90, 96, 37, 45},
            {124, 125, 109, 110, 41, 13, 21, 29, 85, 93, 101, 122, 115}
          },
          Tile.PEI,
          new Tsumo(0, 77),
          new Dahai(0, 77, true),
          new Tsumo(1, 104),
          new Dahai(1, 104, true),
          new Tsumo(2, 81),
          new Dahai(2, 81, true),
          new Tsumo(3, 102),
          new Dahai(3, 102, true),
          new Tsumo(0, 26),
          new Dahai(0, 26, true));

  /** 席 0（親）は W を 3 枚持ち、ドラ表示も W。最初のツモ 2p の打牌が判断（出来事 3）。手牌は 1m 4m 7m 2p 5p 8p 3s 6s 9s E W W W。 */
  private static final LoadedRecord VISIBLE_ROUND =
      round(
          new int[][] {
            {116, 117, 118, 0, 12, 24, 40, 53, 64, 80, 92, 104, 108},
            {4, 17, 28, 44, 56, 68, 72, 84, 96, 112, 120, 124, 128},
            {8, 20, 32, 36, 48, 60, 76, 89, 100, 113, 121, 125, 132},
            {9, 21, 33, 37, 49, 61, 77, 90, 101, 114, 122, 129, 133}
          },
          Tile.SHA,
          new Tsumo(0, 41),
          new Dahai(0, 41, true));

  @Test
  public void editedHandRebuildsEarlierWindowsAndTheLegalDiscards() {
    Answer base = answer(FURITEN_ROUND, 11, TestEdits.NONE);
    Answer edited = answer(FURITEN_ROUND, 11, hand(new HandEdit(12, "9s")));
    Assert.assertFalse(base.input().furiten());
    Assert.assertTrue(edited.input().furiten(), "declining the 9s ron makes the seat furiten");
    Assert.assertTrue(edited.input().legal().contains(discard("9s")));
    Assert.assertFalse(edited.input().legal().contains(discard("E")));
    Assert.assertTrue(base.input().legal().contains(discard("E")));
    ScenarioEditTest.assertUnchanged(
        base.input(),
        edited.input(),
        PolicyInput::events,
        PolicyInput::rivers,
        PolicyInput::scores,
        PolicyInput::indicators,
        PolicyInput::round);
    Assert.assertNotEquals(edited.input().hand(), base.input().hand());
  }

  @Test
  public void equalTilesAreTheSameTileWhicheverPositionIsEdited() {
    PolicyInput first = answer(VISIBLE_ROUND, 3, hand(new HandEdit(10, "P"))).input();
    for (int index : new int[] {11, 12})
      Assert.assertNull(
          PolicyInput.difference(
              first, answer(VISIBLE_ROUND, 3, hand(new HandEdit(index, "P"))).input()),
          "hand index " + index);
    Assert.assertTrue(first.legal().contains(discard("P")));
  }

  @Test
  public void theDrawCanBeReplaced() {
    PolicyInput edited = answer(VISIBLE_ROUND, 3, hand(new HandEdit(13, "C"))).input();
    int drawn = Integer.parseInt(ScenarioEditTest.field(edited.turn(), 2));
    Assert.assertEquals(Tile.typeOf(drawn), Tile.CHUN, edited.turn());
    Assert.assertTrue(edited.legal().contains(tsumogiri("C")));
  }

  @Test
  public void handEditsThatContradictThePositionAreRejected() {
    String noTile = "the hand has no tile at this index at the decision.";
    rejected(VISIBLE_ROUND, 3, hand(new HandEdit(14, "E")), "edits.hand[0]", noTile);
    rejected(VISIBLE_ROUND, 3, hand(new HandEdit(-1, "E")), "edits.hand[0]", noTile);
    rejected(
        VISIBLE_ROUND,
        3,
        hand(new HandEdit(0, "E"), new HandEdit(0, "S")),
        "edits.hand[1]",
        "the same tile is edited twice.");
    rejected(
        VISIBLE_ROUND, 3, hand(new HandEdit(0, "0p")), "edits.hand[0]", "invalid tile code 0p.");
    rejected(
        VISIBLE_ROUND,
        3,
        hand(new HandEdit(0, "W")),
        "edits.hand[0]",
        "more copies of W would be visible than the 4 in the set.");
  }

  @Test
  public void riichiHandsCannotBeEdited() {
    ScenarioEditTest.Decision decision =
        ScenarioEditTest.Decision.find(
            ScenarioEditTest::discards,
            (table, step) -> table.players().get(step.actor()).riichi() ? step.actor() : -1);
    ScenarioEditTest.rejected(
        decision,
        hand(new HandEdit(0, "E")),
        "edits.hand[0]",
        "the hand cannot be edited after a riichi declaration.");
  }

  @Test
  public void discardsAfterTheCallComeFromTheEditedHand() {
    ScenarioEditTest.Decision decision =
        ScenarioEditTest.Decision.find(
            step -> step.offers(Action.Type.PON), (table, step) -> table.activeSeat());
    int seat = decision.step().actor();
    PlayerState own = decision.table().players().get(seat);
    int called = TileCode.type(decision.player().river().getLast().tile());
    int index = 0;
    while (TileCode.type(own.hand().get(index)) == called) index++;
    String tile = ScenarioEditTest.rareTile(decision.table(), seat, own.hand().get(index));
    int pon = -1;
    for (int i = 0; pon < 0; i++)
      if (decision.step().type(i) == Action.Type.PON) pon = decision.step().actionId(i);
    var policy = new RecordingPolicy();
    ScenarioAnswer answer =
        ScenarioEngine.evaluate(
            decision.record(),
            new ScenarioQuery(
                decision.round(),
                decision.step().eventIndex(),
                seat,
                hand(new HandEdit(index, tile)),
                pon),
            policy);
    Assert.assertTrue(policy.inputs.getFirst().legal().contains(discard(tile)));
    Assert.assertTrue(answer.table().players().get(seat).hand().contains(tile));
    Assert.assertEquals(answer.decision().chosen(), -1);
  }

  private static void rejected(
      LoadedRecord record, int eventIndex, ScenarioEdits edits, String name, String reason) {
    ReviewerException failure =
        Assert.expectThrows(ReviewerException.class, () -> answer(record, eventIndex, edits));
    Assert.assertEquals(failure.code(), ErrorCode.INVALID_EDIT, failure.getMessage());
    Assert.assertTrue(failure.getMessage().startsWith(name + ": "), failure.getMessage());
    Assert.assertTrue(failure.getMessage().endsWith(reason), failure.getMessage());
  }

  private static Answer answer(LoadedRecord record, int eventIndex, ScenarioEdits edits) {
    var policy = new RecordingPolicy();
    ScenarioAnswer answer =
        ScenarioEngine.evaluate(record, new ScenarioQuery(0, eventIndex, 0, edits, null), policy);
    Assert.assertEquals(policy.inputs.size(), 1);
    return new Answer(policy.inputs.getFirst(), answer);
  }

  private static int discard(String tile) {
    return Action.dahai(TileCode.type(tile), TileCode.red(tile), false).toIndex();
  }

  private static int tsumogiri(String tile) {
    return Action.dahai(TileCode.type(tile), TileCode.red(tile), true).toIndex();
  }

  /** 東 1 局の親が席 0 の作った局（局の始めの後の出来事を続ける）。 */
  private static LoadedRecord round(int[][] hands, int dora, ReplayEvent... events) {
    List<ReplayEvent> all = new ArrayList<>();
    all.add(new StartGame());
    all.add(new StartKyoku(Tile.TON, 1, 0, 0, 0, dora, hands, SCORES.clone()));
    all.addAll(List.of(events));
    return ScenarioReplayTest.synthetic(all);
  }

  private record Answer(PolicyInput input, ScenarioAnswer answer) {}
}
