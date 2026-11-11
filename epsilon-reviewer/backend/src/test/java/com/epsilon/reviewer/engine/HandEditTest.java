package com.epsilon.reviewer.engine;

import static com.epsilon.reviewer.engine.TestEdits.hand;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.record.RecordEvent.Discard;
import com.epsilon.mahjong.record.RecordEvent.Draw;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.scenario.HandEdit;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.epsilon.reviewer.engine.scenario.ScenarioEdits;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.engine.view.TileCode;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 判断する席の手牌を替えるもしものテスト。その席から見た局を、替えた配牌とツモで再生し直す。そのため、手牌と合法な打牌のほかに、前の応答で
 * ロンを見送ったときの見逃しの振聴と、鳴いた後の打牌も変わる。同じ牌なら、どの位置を指しても同じ 1 枚を替える。立直した後の変更と、
 * 見える枚数が上限を超える変更は断る。
 */
public class HandEditTest {
  /**
   * 席 0（親）は 123m456m789m123p と E の単騎待ち。席 1 は 9s をツモ切りする。判断は 2 つある。1 つは席 3 がツモ切りした 4p に
   * チーするかの応答で、記録のイベント 8 で決まる。もう 1 つはその後に席 0 が 7m をツモったときの打牌で、イベント 9 で決まる。E を 9s に
   * 替えると、席 1 の 9s で一気通貫のロンができたのに見送ったことになり、次のツモまで見逃しの振聴になる。
   */
  private static final LoadedRecord FURITEN_ROUND =
      TestRounds.round(
          new int[][] {
            {0, 4, 8, 12, 17, 20, 24, 28, 32, 36, 40, 44, 108},
            {112, 113, 114, 116, 117, 118, 120, 48, 56, 64, 68, 72, 89},
            {128, 129, 130, 132, 133, 134, 121, 60, 69, 90, 96, 37, 45},
            {124, 125, 109, 110, 41, 13, 21, 29, 85, 93, 101, 122, 115}
          },
          123,
          new Draw(0, 77),
          new Discard(0, 77, true, false),
          new Draw(1, 104),
          new Discard(1, 104, true, false),
          new Draw(2, 81),
          new Discard(2, 81, true, false),
          new Draw(3, 50),
          new Discard(3, 50, true, false),
          new Draw(0, 26),
          new Discard(0, 26, true, false));

  private static final int CHI_RESPONSE = 8;
  private static final int FURITEN_TURN = 9;

  /**
   * 席 0（親）は W を 3 枚持ち、ドラ表示牌も W。判断は最初に 2p をツモったときの打牌で、イベント 1 で決まる。配牌は
   * 1m 4m 7m 2p 5p 8p 3s 6s 9s E W W W。
   */
  private static final LoadedRecord VISIBLE_ROUND =
      TestRounds.round(
          new int[][] {
            {116, 117, 118, 0, 12, 24, 40, 53, 64, 80, 92, 104, 108},
            {4, 17, 28, 44, 56, 68, 72, 84, 96, 112, 120, 124, 128},
            {8, 20, 32, 36, 48, 60, 76, 89, 100, 113, 121, 125, 132},
            {9, 21, 33, 37, 49, 61, 77, 90, 101, 114, 122, 129, 133}
          },
          119,
          new Draw(0, 41),
          new Discard(0, 41, true, false));

  private static final int VISIBLE_TURN = 1;

  /**
   * 席 0（親）は 3m を 2 枚（物理牌 8 と 9）持ち、最初の手番で 2p をツモって、番号の大きい 9 を手出しする。天鳳の牌譜は、このように
   * 打った物理牌をそのまま記録する。判断は次に 2s をツモった後の打牌（イベント 9 で決まる）で、手には 3m の 8 が残っている。その時点の
   * 手牌は 1m 3m 4m 7m 2p 2p 5p 8p 3s 6s 9s E W。
   */
  private static final LoadedRecord PAIR_ROUND =
      TestRounds.round(
          new int[][] {
            {8, 9, 0, 12, 24, 40, 53, 64, 80, 92, 104, 108, 116},
            {4, 17, 28, 44, 56, 68, 72, 84, 96, 112, 120, 124, 128},
            {5, 20, 32, 36, 48, 60, 76, 89, 100, 113, 121, 125, 132},
            {6, 21, 33, 37, 49, 61, 78, 90, 101, 114, 122, 129, 133}
          },
          119,
          new Draw(0, 41),
          new Discard(0, 9, false, false),
          new Draw(1, 105),
          new Discard(1, 105, true, false),
          new Draw(2, 81),
          new Discard(2, 81, true, false),
          new Draw(3, 50),
          new Discard(3, 50, true, false),
          new Draw(0, 77),
          new Discard(0, 77, true, false));

  private static final int PAIR_TURN = 9;

  @Test
  public void editedHandRebuildsEarlierResponsesAndTheLegalDiscards() {
    Answer base = answer(FURITEN_ROUND, CHI_RESPONSE, TestEdits.NONE);
    Answer edited = answer(FURITEN_ROUND, CHI_RESPONSE, hand(new HandEdit(12, "9s")));
    Assert.assertFalse(base.input().furiten());
    Assert.assertTrue(edited.input().furiten(), "declining the 9s ron makes the seat furiten");
    ScenarioEditTest.assertUnchanged(
        base.input(),
        edited.input(),
        ObservedInput::events,
        ObservedInput::scores,
        ObservedInput::indicators);
    Assert.assertNotEquals(edited.input().hand(), base.input().hand());
    ObservedInput turn = answer(FURITEN_ROUND, FURITEN_TURN, hand(new HandEdit(12, "9s"))).input();
    ObservedInput recorded = answer(FURITEN_ROUND, FURITEN_TURN, TestEdits.NONE).input();
    Assert.assertTrue(turn.legal().contains(discard("9s")));
    Assert.assertFalse(turn.legal().contains(discard("E")));
    Assert.assertTrue(recorded.legal().contains(discard("E")));
  }

  @Test
  public void equalTilesAreTheSameTileWhicheverPositionIsEdited() {
    var first = new RecordingPolicy();
    evaluate(VISIBLE_ROUND, VISIBLE_TURN, hand(new HandEdit(10, "P")), first);
    for (int index : new int[] {11, 12}) {
      var other = new RecordingPolicy();
      evaluate(VISIBLE_ROUND, VISIBLE_TURN, hand(new HandEdit(index, "P")), other);
      Assert.assertEquals(other.inputs.getFirst(), first.inputs.getFirst(), "hand index " + index);
    }
    Assert.assertTrue(first.input(0).legal().contains(discard("P")));
  }

  @Test
  public void theCopyLeftOfAPairIsReplacedAfterTheOtherCopyWasDiscarded() {
    ObservedInput recorded = answer(PAIR_ROUND, PAIR_TURN, TestEdits.NONE).input();
    Assert.assertTrue(recorded.legal().contains(discard("3m")));
    ObservedInput edited = answer(PAIR_ROUND, PAIR_TURN, hand(new HandEdit(1, "P"))).input();
    Assert.assertTrue(edited.legal().contains(discard("P")));
    Assert.assertFalse(edited.legal().contains(discard("3m")));
    ScenarioEditTest.assertUnchanged(
        recorded, edited, ObservedInput::events, ObservedInput::scores, ObservedInput::indicators);
  }

  @Test
  public void theDrawCanBeReplaced() {
    ObservedInput edited = answer(VISIBLE_ROUND, VISIBLE_TURN, hand(new HandEdit(13, "C"))).input();
    Assert.assertEquals(edited.drawn(), Tiles.CHUN);
    Assert.assertTrue(edited.legal().contains(tsumogiri("C")));
  }

  @Test
  public void handEditsThatContradictThePositionAreRejected() {
    String noTile = "the hand has no tile at this index at the decision.";
    rejected(VISIBLE_ROUND, VISIBLE_TURN, hand(new HandEdit(14, "E")), "edits.hand[0]", noTile);
    rejected(VISIBLE_ROUND, VISIBLE_TURN, hand(new HandEdit(-1, "E")), "edits.hand[0]", noTile);
    rejected(
        VISIBLE_ROUND,
        VISIBLE_TURN,
        hand(new HandEdit(0, "E"), new HandEdit(0, "S")),
        "edits.hand[1]",
        "the same tile is edited twice.");
    rejected(
        VISIBLE_ROUND,
        VISIBLE_TURN,
        hand(new HandEdit(0, "0p")),
        "edits.hand[0]",
        "invalid tile code 0p.");
    rejected(
        VISIBLE_ROUND,
        VISIBLE_TURN,
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
    int called = TileCode.kind(decision.player().river().getLast().tile());
    int index = 0;
    while (TileCode.kind(own.hand().get(index)) == called) index++;
    String tile = ScenarioEditTest.rareTile(decision.table(), seat, own.hand().get(index));
    int pon = -1;
    for (int i = 0; pon < 0; i++)
      if (decision.step().type(i) == Action.Type.PON) pon = decision.step().actionId(i);
    var policy = new RecordingPolicy();
    ScenarioAnswer answer =
        Scenarios.evaluate(
            decision.record(),
            new ScenarioQuery(
                decision.round(),
                decision.step().eventIndex(),
                seat,
                hand(new HandEdit(index, tile)),
                pon),
            policy);
    Assert.assertTrue(policy.input(0).legal().contains(discard(tile)));
    Assert.assertTrue(answer.table().players().get(seat).hand().contains(tile));
    Assert.assertEquals(answer.decision().chosen(), -1);
  }

  private static void rejected(
      LoadedRecord record, int position, ScenarioEdits edits, String name, String reason) {
    ReviewerException failure =
        Assert.expectThrows(ReviewerException.class, () -> answer(record, position, edits));
    Assert.assertEquals(failure.code(), ErrorCode.INVALID_EDIT, failure.getMessage());
    Assert.assertTrue(failure.getMessage().startsWith(name + ": "), failure.getMessage());
    Assert.assertTrue(failure.getMessage().endsWith(reason), failure.getMessage());
  }

  /** 席 0 の判断で、もしもを評価する。判断は、その判断が決まる記録のイベントの位置（{@code position}）で指す。 */
  private static ScenarioAnswer evaluate(
      LoadedRecord record, int position, ScenarioEdits edits, RecordingPolicy policy) {
    return Scenarios.evaluate(
        record, new ScenarioQuery(0, record.step(0, position), 0, edits, null), policy);
  }

  private static Answer answer(LoadedRecord record, int position, ScenarioEdits edits) {
    var policy = new RecordingPolicy();
    ScenarioAnswer answer = evaluate(record, position, edits, policy);
    Assert.assertEquals(policy.inputs.size(), 1);
    return new Answer(policy.input(0), answer);
  }

  private static int discard(String tile) {
    return Action.discard(TileCode.kind(tile), TileCode.red(tile), false, false);
  }

  private static int tsumogiri(String tile) {
    return Action.discard(TileCode.kind(tile), TileCode.red(tile), true, false);
  }

  private record Answer(ObservedInput input, ScenarioAnswer answer) {}
}
