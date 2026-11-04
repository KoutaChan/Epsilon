package com.epsilon.reviewer.engine;

import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.EndGame;
import com.epsilon.replay.ReplayEvent.EndKyoku;
import com.epsilon.replay.ReplayEvent.Hora;
import com.epsilon.replay.ReplayEvent.Ryukyoku;
import com.epsilon.replay.ReplayEvent.StartGame;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import com.epsilon.reviewer.engine.view.OutcomeProjector;
import com.epsilon.reviewer.engine.view.RoundOutcome;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 連続ロン・ツモ・槍槓の局の結果と物理牌を保ち、点の移動と流局の理由を文書に書かないことを確かめる。 */
public class RoundOutcomeTest {
  @Test
  public void consecutiveRonResultsKeepEachWinnerAndThePostEventScore() {
    int[][] hands =
        physicalHands(
            new int[] {3, 3, 3, 4, 5, 12, 13, 14, 21, 22, 23, Tile.NAN, Tile.SHA},
            new int[] {0, 1, 2, 9, 10, 11, 18, 19, 20, Tile.TON, Tile.TON, Tile.TON, Tile.HAKU},
            new int[] {6, 7, 8, 15, 16, 17, 24, 25, 26, Tile.CHUN, Tile.CHUN, Tile.CHUN, Tile.HAKU},
            new int[] {1, 1, 2, 4, 5, 7, 10, 12, 14, 19, 21, Tile.NAN, Tile.PEI});
    List<Format6.Step> steps =
        eventSteps(
            new StartGame(),
            new StartKyoku(
                Tile.TON, 1, 0, 0, 0, Tile.HATSU, hands, new int[] {25000, 25000, 25000, 25000}),
            new Tsumo(0, 126),
            new Dahai(0, 126, true),
            new Hora(1, 0, Tile.HAKU, new int[] {-2000, 2000, 0, 0}),
            new Hora(2, 0, Tile.HAKU, new int[] {-3000, 0, 3000, 0}),
            new EndKyoku(),
            new EndGame(new int[] {20000, 27000, 29000, 25000}));
    var wins = steps.stream().filter(step -> step.json().has("outcome")).toList();
    Assert.assertEquals(wins.size(), 2);
    Assert.assertEquals(outcome(wins.get(0)), hora(1, 0, "P"));
    Assert.assertEquals(outcome(wins.get(1)), hora(2, 0, "P"));
    Assert.assertEquals(wins.get(0).table().players().get(0).score(), 23000);
    Assert.assertEquals(wins.get(1).table().players().get(0).score(), 20000);
    Assert.assertEquals(wins.get(1).table().players().get(2).score(), 28000);
    Assert.assertEquals(steps.getLast().event(), "end_game");
    Assert.assertEquals(steps.getLast().table().players().get(2).score(), 29000);
    Assert.assertFalse(steps.getLast().json().has("outcome"));
  }

  @Test
  public void redTsumoRemainsSeparateFromTheWinningConcealedHand() {
    List<Format6.Step> steps =
        eventSteps(
            new StartGame(),
            redFiveStart(0),
            new Tsumo(0, 52),
            new Hora(0, 0, Tile.P5, new int[] {6000, -2000, -2000, -2000}));
    Format6.Step win = steps.getLast();
    Assert.assertEquals(outcome(win), hora(0, 0, "5pr"));
    Assert.assertEquals(win.table().players().get(0).draw(), "5pr");
    Assert.assertEquals(win.table().players().get(0).hand().size(), 13);
    Assert.assertFalse(win.table().players().get(0).hand().contains("5pr"));
    Assert.assertEquals(win.table().players().get(0).score(), 31000);
  }

  @Test
  public void redRonUsesTheDiscardAndLeavesTheWinnerHandAtThirteenTiles() {
    List<Format6.Step> steps =
        eventSteps(
            new StartGame(),
            redFiveStart(1),
            new Tsumo(0, 52),
            new Dahai(0, 52, true),
            new Hora(1, 0, Tile.P5, new int[] {-3900, 3900, 0, 0}));
    Format6.Step win = steps.getLast();
    Assert.assertEquals(outcome(win), hora(1, 0, "5pr"));
    Assert.assertEquals(win.table().players().get(1).hand().size(), 13);
    Assert.assertNull(win.table().players().get(1).draw());
    Assert.assertEquals(win.table().players().get(0).river().getLast().tile(), "5pr");
  }

  @Test
  public void robbedRedAddedKanIsCapturedBeforeTheBorrowedTurnEventIsCleared() {
    GameState state = new GameState();
    state.recordKanAttempt(3, Tile.P5, true, TurnEvent.KanKind.KAKAN);
    RoundOutcome outcome =
        OutcomeProjector.of(new Hora(1, 3, Tile.P5, new int[] {0, 8000, 0, -8000}), state);
    state.clearTurnEvent();
    Assert.assertEquals(outcome, new RoundOutcome.Hora(1, 3, "5pr", null));
  }

  @Test
  public void drawOutcomesCarryNeitherPaymentsNorAReason() {
    List<Format6.Step> steps =
        eventSteps(new StartGame(), redFiveStart(0), new Ryukyoku(null), new EndKyoku());
    Assert.assertEquals(
        outcome(steps.get(1)), JsonParser.parseString("{\"type\":\"ryukyoku\"}"));
    Assert.assertFalse(steps.getFirst().json().has("outcome"));
    Assert.assertFalse(steps.getLast().json().has("outcome"));
    Assert.assertEquals(steps.get(1).table().players().get(0).score(), 25000);
  }

  @Test
  public void drawPaymentsMoveTheScoresWithoutAnOutcomePayment() {
    List<Format6.Step> steps =
        eventSteps(
            new StartGame(),
            redFiveStart(0),
            new Ryukyoku(new int[] {1500, -1500, 1500, -1500}));
    Format6.Step draw = steps.getLast();
    Assert.assertEquals(outcome(draw), JsonParser.parseString("{\"type\":\"ryukyoku\"}"));
    Assert.assertEquals(draw.table().players().get(0).score(), 26500);
    Assert.assertEquals(draw.table().players().get(1).score(), 23500);
  }

  private static JsonObject outcome(Format6.Step step) {
    return step.json().getAsJsonObject("outcome");
  }

  private static JsonObject hora(int actor, int target, String tile) {
    JsonObject hora = new JsonObject();
    hora.addProperty("type", "hora");
    hora.addProperty("actor", actor);
    hora.addProperty("target", target);
    hora.addProperty("winningTile", tile);
    return hora;
  }

  /** 解析した文書の出来事の手順。 */
  private static List<Format6.Step> eventSteps(ReplayEvent... events) {
    Format6.Document document =
        Format6.analyze(ScenarioReplayTest.synthetic(List.of(events)), new RecordingPolicy());
    return document.rounds().getFirst().steps().stream()
        .filter(step -> !step.decision())
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
