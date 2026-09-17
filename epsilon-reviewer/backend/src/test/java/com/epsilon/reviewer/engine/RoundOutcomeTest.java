package com.epsilon.reviewer.engine;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.replay.*;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.*;
import com.epsilon.reviewer.dto.ReplayRound;
import com.epsilon.reviewer.dto.ReplayStep;
import com.epsilon.reviewer.dto.RoundOutcome;
import com.epsilon.reviewer.dto.TableState;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParser;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 連続ロン・ツモ・槍槓の局結果と物理牌を保持し、未記録の点棒移動を推測で補わないことを検証する。 */
public class RoundOutcomeTest {
  @Test
  public void consecutiveRonResultsKeepEachWinnerPaymentAndPostEventScore() {
    int[][] hands =
        physicalHands(
            new int[] {3, 3, 3, 4, 5, 12, 13, 14, 21, 22, 23, Tile.NAN, Tile.SHA},
            new int[] {0, 1, 2, 9, 10, 11, 18, 19, 20, Tile.TON, Tile.TON, Tile.TON, Tile.HAKU},
            new int[] {6, 7, 8, 15, 16, 17, 24, 25, 26, Tile.CHUN, Tile.CHUN, Tile.CHUN, Tile.HAKU},
            new int[] {1, 1, 2, 4, 5, 7, 10, 12, 14, 19, 21, Tile.NAN, Tile.PEI});
    List<OutcomeFrame> frames =
        replayFrames(
            List.of(
                new StartGame(),
                new StartKyoku(
                    Tile.TON,
                    1,
                    0,
                    0,
                    0,
                    Tile.HATSU,
                    hands,
                    new int[] {25000, 25000, 25000, 25000}),
                new Tsumo(0, 126),
                new Dahai(0, 126, true),
                new Hora(1, 0, Tile.HAKU, new int[] {-2000, 2000, 0, 0}),
                new Hora(2, 0, Tile.HAKU, new int[] {-3000, 0, 3000, 0}),
                new EndKyoku(),
                new EndGame(new int[] {20000, 27000, 29000, 25000})));
    var wins = frames.stream().filter(frame -> frame.outcome() != null).toList();
    Assert.assertEquals(wins.size(), 2);
    Assert.assertEquals(wins.get(0).outcome().actor().intValue(), 1);
    Assert.assertEquals(wins.get(1).outcome().actor().intValue(), 2);
    Assert.assertEquals(wins.get(0).outcome().target().intValue(), 0);
    Assert.assertEquals(wins.get(1).outcome().winningTile(), "P");
    Assert.assertEquals(wins.get(0).outcome().deltas(), List.of(-2000, 2000, 0, 0));
    Assert.assertEquals(wins.get(1).outcome().deltas(), List.of(-3000, 0, 3000, 0));
    Assert.assertEquals(wins.get(0).state().players().get(0).score(), 23000);
    Assert.assertEquals(wins.get(1).state().players().get(0).score(), 20000);
    Assert.assertEquals(wins.get(1).state().players().get(2).score(), 28000);
    Assert.assertEquals(frames.getLast().state().players().get(2).score(), 29000);
    Assert.assertNull(frames.getLast().outcome());
  }

  @Test
  public void redTsumoRemainsSeparateFromTheWinningConcealedHand() {
    List<OutcomeFrame> frames =
        replayFrames(
            List.of(
                new StartGame(),
                redFiveStart(0),
                new Tsumo(0, 52),
                new Hora(0, 0, Tile.P5, new int[] {6000, -2000, -2000, -2000})));
    var win = frames.getLast();
    Assert.assertEquals(win.outcome().winningTile(), "5pr");
    Assert.assertEquals(win.outcome().actor(), win.outcome().target());
    Assert.assertEquals(win.state().players().get(0).draw(), "5pr");
    Assert.assertEquals(win.state().players().get(0).hand().size(), 13);
    Assert.assertFalse(win.state().players().get(0).hand().contains("5pr"));
    Assert.assertEquals(win.outcome().deltas(), List.of(6000, -2000, -2000, -2000));
  }

  @Test
  public void redRonUsesTheDiscardAndLeavesTheWinnerHandAtThirteenTiles() {
    List<OutcomeFrame> frames =
        replayFrames(
            List.of(
                new StartGame(),
                redFiveStart(1),
                new Tsumo(0, 52),
                new Dahai(0, 52, true),
                new Hora(1, 0, Tile.P5, new int[] {-3900, 3900, 0, 0})));
    var win = frames.getLast();
    Assert.assertEquals(win.outcome().winningTile(), "5pr");
    Assert.assertEquals(win.outcome().actor().intValue(), 1);
    Assert.assertEquals(win.outcome().target().intValue(), 0);
    Assert.assertEquals(win.state().players().get(1).hand().size(), 13);
    Assert.assertNull(win.state().players().get(1).draw());
    Assert.assertEquals(win.state().players().get(0).river().getLast().tile(), "5pr");
  }

  @Test
  public void robbedRedAddedKanIsCapturedBeforeTheBorrowedTurnEventIsCleared() {
    GameState state = new GameState();
    state.recordKanAttempt(3, Tile.P5, true, TurnEvent.KanKind.KAKAN);
    int[] deltas = {0, 8000, 0, -8000};
    var outcome = ReviewSnapshots.snapshotRoundOutcome(new Hora(1, 3, Tile.P5, deltas), state);
    state.clearTurnEvent();
    deltas[1] = 0;
    Assert.assertEquals(outcome.winningTile(), "5pr");
    Assert.assertEquals(outcome.target().intValue(), 3);
    Assert.assertEquals(outcome.deltas(), List.of(0, 8000, 0, -8000));
  }

  @Test
  public void unrecordedDrawPaymentsRemainNullAndDoNotBecomeZeroPayments() {
    List<OutcomeFrame> frames =
        replayFrames(List.of(new StartGame(), redFiveStart(0), new Ryukyoku(null), new EndKyoku()));
    var draw = frames.get(1);
    Assert.assertEquals(draw.outcome().type(), "ryukyoku");
    Assert.assertNull(draw.outcome().actor());
    Assert.assertNull(draw.outcome().deltas());
    var json =
        JsonParser.parseString(new GsonBuilder().serializeNulls().create().toJson(draw))
            .getAsJsonObject();
    Assert.assertTrue(json.getAsJsonObject("outcome").get("deltas").isJsonNull());
    Assert.assertTrue(json.getAsJsonObject("outcome").get("winningTile").isJsonNull());
    Assert.assertNull(frames.getFirst().outcome());
    Assert.assertNull(frames.getLast().outcome());
  }

  @Test
  public void drawPaymentsRetainRecordedTenpaiTransfersWithoutInventingAReason() {
    List<OutcomeFrame> frames =
        replayFrames(
            List.of(
                new StartGame(),
                redFiveStart(0),
                new Ryukyoku(new int[] {1500, -1500, 1500, -1500})));
    var draw = frames.getLast();
    Assert.assertEquals(draw.outcome().deltas(), List.of(1500, -1500, 1500, -1500));
    Assert.assertEquals(draw.state().players().get(0).score(), 26500);
    Assert.assertEquals(draw.state().players().get(1).score(), 23500);
  }

  private record OutcomeFrame(TableState state, RoundOutcome outcome) {}

  private static List<OutcomeFrame> replayFrames(List<ReplayEvent> events) {
    var states = new ReviewStateSequence();
    List<ReplayStep> steps = new ArrayList<>();
    ReplayEngine.replay(
        new ReplayRecord(
            events,
            new RecordMetadata(
                RecordFormat.MJAI, List.of("A", "B", "C", "D"), null, "unavailable", null),
            null,
            RecordCompletion.INCOMPLETE),
        new ReplayObserver() {
          public void onDecision(
              DecisionPoint point, GameState state, int seat, List<Action> legal) {}

          public void onEventApplied(int index, int stateId, ReplayEvent event, GameState state) {
            if (event instanceof StartGame) return;
            states.captureState(stateId, state);
            steps.add(
                new ReplayStep(
                    steps.size(),
                    index,
                    stateId,
                    "event",
                    -1,
                    false,
                    null,
                    List.of(),
                    ReviewSnapshots.snapshotRoundOutcome(event, state)));
          }
        });
    var json = new GsonBuilder().serializeNulls().create();
    var round =
        json.fromJson(
            json.toJson(new ReplayRound("round-0", 0, 0, 0, states.frames(), steps)),
            ReplayRound.class);
    return round.steps().stream()
        .map(
            step ->
                new OutcomeFrame(
                    StateRestoration.at(round.states(), step.stateId()), step.outcome()))
        .toList();
  }

  private static StartKyoku redFiveStart(int winner) {
    int[][] hands = new int[4][13];
    boolean[] used = new boolean[136];
    used[52] = true;
    used[132] = true;
    int[] winningTypes = {0, 1, 2, 6, 7, 8, 18, 19, 20, Tile.TON, Tile.TON, Tile.TON, Tile.P5};
    for (int i = 0; i < winningTypes.length; i++) {
      int id = winningTypes[i] * 4;
      while (used[id]) id++;
      hands[winner][i] = id;
      used[id] = true;
    }
    int next = 0;
    for (int seat = 0; seat < 4; seat++) {
      if (seat == winner) continue;
      for (int i = 0; i < 13; i++) {
        while (used[next]) next++;
        hands[seat][i] = next;
        used[next++] = true;
      }
    }
    return new StartKyoku(
        Tile.TON, 1, 0, 0, 0, Tile.CHUN, hands, new int[] {25000, 25000, 25000, 25000});
  }

  private static int[][] physicalHands(int[]... types) {
    int[] copies = new int[34];
    int[][] hands = new int[types.length][];
    for (int seat = 0; seat < types.length; seat++) {
      hands[seat] = new int[types[seat].length];
      for (int i = 0; i < types[seat].length; i++) {
        int type = types[seat][i];
        hands[seat][i] = type * 4 + copies[type]++;
      }
    }
    return hands;
  }
}
