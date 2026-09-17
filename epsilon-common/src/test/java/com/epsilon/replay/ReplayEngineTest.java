package com.epsilon.replay;

import com.epsilon.core.Action;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.Dora;
import com.epsilon.replay.ReplayEvent.EndGame;
import com.epsilon.replay.ReplayEvent.Hora;
import com.epsilon.replay.ReplayEvent.None;
import com.epsilon.replay.ReplayEvent.Pon;
import com.epsilon.replay.ReplayEvent.Reach;
import com.epsilon.replay.ReplayEvent.ReachAccepted;
import com.epsilon.replay.ReplayEvent.StartGame;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 行動適用前の判断とイベント適用後の観測をモデルなしで確認する。 */
public class ReplayEngineTest {
  @Test
  public void discardCallbackBorrowsTheDrawBeforeHandAndRiverAreChanged() {
    var selected = new ArrayList<Action>();
    int[] scores =
        replaySelected(
            discardedSeven(),
            (state, player, legal, chosenSlot) -> {
              Assert.assertEquals(player, 1);
              Assert.assertTrue(chosenSlot >= 0 && chosenSlot < legal.size());
              Assert.assertEquals(legal.get(chosenSlot), Action.dahai(Tile.M7, false, true));
              Assert.assertEquals(state.hand(player).concealedTileCount(), 14);
              Assert.assertTrue(state.hand(player).containsPhysicalTile(25));
              Assert.assertEquals(state.river(player).size(), 0);
              Assert.assertEquals(((TurnEvent.Draw) state.getTurnEvent()).physicalTileId(), 25);
              selected.add(legal.get(chosenSlot));
            });
    Assert.assertEquals(selected, List.of(Action.dahai(Tile.M7, false, true)));
    Assert.assertEquals(scores, new int[] {25000, 25000, 25000, 25000});
  }

  @Test
  public void truncatedResponseWindowDoesNotInventPassLabels() {
    var choices = new ArrayList<String>();
    replaySelected(
        discardedSeven(),
        (state, player, legal, chosenSlot) ->
            choices.add(player + ":" + legal.get(chosenSlot).type()));
    Assert.assertEquals(choices, List.of("1:DAHAI"));
  }

  @Test
  public void nextDrawConfirmsPassesInSeatOrderAfterTheDoraAnnouncement() {
    var events = discardedSeven();
    events.add(new Dora(Tile.P6));
    events.add(new Tsumo(2, 27));
    var choices = new ArrayList<String>();
    var indicatorCounts = new ArrayList<Integer>();
    replaySelected(
        events,
        (state, player, legal, chosenSlot) -> {
          Action action = legal.get(chosenSlot);
          choices.add(player + ":" + action.type());
          indicatorCounts.add(state.doraState().indicatorCount());
          if (action.type() == Action.Type.PASS) {
            Assert.assertEquals(state.river(1).size(), 1);
            Assert.assertEquals(state.hand(1).concealedTileCount(), 13);
            Assert.assertEquals(state.hand(2).concealedTileCount(), 13);
            Assert.assertEquals(state.doraState().indicatorTileType(1), Tile.P6);
          }
        });
    Assert.assertEquals(choices, List.of("1:DAHAI", "0:PASS", "2:PASS"));
    Assert.assertEquals(indicatorCounts, List.of(1, 2, 2));
  }

  @Test
  public void aPonDoesNotLabelTheBlockedChiAsAPass() {
    var events = discardedSeven();
    events.add(new Pon(0, 1, 25, new int[] {24, 26}));
    var choices = new ArrayList<String>();
    replaySelected(
        events,
        (state, player, legal, chosenSlot) -> {
          Action action = legal.get(chosenSlot);
          choices.add(player + ":" + action.type());
          if (action.type() == Action.Type.PON) {
            Assert.assertEquals(player, 0);
            Assert.assertEquals(action, Action.pon(Tile.M7, false));
            Assert.assertEquals(state.hand(0).meldCount(), 0);
            Assert.assertTrue(state.hand(0).containsPhysicalTile(24));
            Assert.assertTrue(state.hand(0).containsPhysicalTile(26));
            Assert.assertEquals(state.river(1).size(), 1);
          }
        });
    Assert.assertEquals(choices, List.of("1:DAHAI", "0:PON"));
  }

  @Test
  public void anExplicitPassAtTheEndDoesNotResolveAnotherPlayersChoice() {
    var events = discardedSeven();
    events.add(new None(0));
    var choices = new ArrayList<String>();
    replaySelected(
        events,
        (state, player, legal, chosenSlot) ->
            choices.add(player + ":" + legal.get(chosenSlot).type()));
    Assert.assertEquals(choices, List.of("1:DAHAI", "0:PASS"));
  }

  @Test
  public void acceptedRiichiChangesTheObservationAndEndGameOwnsTheFinalScores() {
    int[] startingScores = {25000, 25000, 25000, 25000};
    int[] recordedFinalScores = {31000, 18000, 27000, 24000};
    var choices = new ArrayList<Action.Type>();
    int[] scores =
        replaySelected(
            List.of(
                new StartGame(),
                new StartKyoku(Tile.TON, 1, 0, 0, 1, Tile.HAKU, callHands(), startingScores),
                new Tsumo(1, 25),
                new Reach(1),
                new Dahai(1, 25, true),
                new ReachAccepted(1),
                new EndGame(recordedFinalScores)),
            (state, player, legal, chosenSlot) -> {
              Action.Type type = legal.get(chosenSlot).type();
              boolean declaring = type == Action.Type.RIICHI_DAHAI;
              Assert.assertEquals(state.getScore(1), declaring ? 25000 : 24000);
              Assert.assertEquals(state.getKyotakuCount(), declaring ? 0 : 1);
              choices.add(type);
            });
    Assert.assertEquals(
        choices, List.of(Action.Type.RIICHI_DAHAI, Action.Type.PASS, Action.Type.PASS));
    Assert.assertEquals(startingScores, new int[] {25000, 25000, 25000, 25000});
    recordedFinalScores[0] = -1;
    Assert.assertEquals(scores, new int[] {31000, 18000, 27000, 24000});
  }

  @Test
  public void bothRonCallbacksSeeTheUnsettledScoresBeforeTheirDeltasAreCombined() {
    int[][] hands =
        physicalHands(
            new int[] {3, 3, 3, 4, 5, 12, 13, 14, 21, 22, 23, Tile.NAN, Tile.SHA},
            new int[] {0, 1, 2, 9, 10, 11, 18, 19, 20, Tile.TON, Tile.TON, Tile.TON, Tile.HAKU},
            new int[] {6, 7, 8, 15, 16, 17, 24, 25, 26, Tile.CHUN, Tile.CHUN, Tile.CHUN, Tile.HAKU},
            new int[] {1, 1, 2, 4, 5, 7, 10, 12, 14, 19, 21, Tile.NAN, Tile.PEI});
    var choices = new ArrayList<String>();
    int[] scores =
        replaySelected(
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
                new Hora(2, 0, Tile.HAKU, new int[] {-3000, 0, 3000, 0})),
            (state, player, legal, chosenSlot) -> {
              Action action = legal.get(chosenSlot);
              choices.add(player + ":" + action.type());
              if (action.type() == Action.Type.RON_AGARI) {
                Assert.assertEquals(state.river(0).size(), 1);
                for (int seat = 0; seat < 4; seat++)
                  Assert.assertEquals(state.getScore(seat), 25000);
              }
            });
    Assert.assertEquals(choices, List.of("0:DAHAI", "1:RON_AGARI", "2:RON_AGARI"));
    Assert.assertEquals(scores, new int[] {20000, 27000, 28000, 25000});
  }

  @Test
  public void anIllegalRecordedDiscardFailsBeforeItCanReachTheConsumer() {
    var events = discardedSeven();
    events.set(events.size() - 1, new Dahai(1, 128, false));
    Assert.expectThrows(
        IllegalStateException.class,
        () ->
            replaySelected(
                events,
                (state, player, legal, chosenSlot) ->
                    Assert.fail("An illegal discard must not reach the training callback")));
  }

  @Test
  public void observerSeesRiichiPaymentDoraAndRecordedEndGameScoresInEventOrder() {
    var events =
        new ArrayList<ReplayEvent>(
            List.of(
                new StartGame(),
                new StartKyoku(
                    Tile.TON,
                    1,
                    0,
                    0,
                    1,
                    Tile.HAKU,
                    callHands(),
                    new int[] {25000, 25000, 25000, 25000}),
                new Tsumo(1, 25),
                new Reach(1),
                new Dahai(1, 25, true),
                new ReachAccepted(1),
                new Dora(Tile.P6),
                new EndGame(new int[] {31000, 18000, 27000, 24000})));
    var indices = new ArrayList<Integer>();
    var payments = new ArrayList<Integer>();
    ReplayEngine.replay(
        record(events),
        new ReplayObserver() {
          public void onDecision(
              DecisionPoint point, com.epsilon.core.GameState state, int seat, List<Action> legal) {
            int chosen = point.chosenSlot();
            int eventIndex = point.eventIndex();
            if (chosen >= 0 && legal.get(chosen).type() == Action.Type.RIICHI_DAHAI) {
              Assert.assertEquals(eventIndex, 4);
              Assert.assertEquals(state.getKyotakuCount(), 0);
            }
          }

          public void onEventApplied(
              int eventIndex, int stateId, ReplayEvent event, com.epsilon.core.GameState state) {
            indices.add(eventIndex);
            if (event instanceof ReachAccepted) {
              Assert.assertEquals(state.getKyotakuCount(), 1);
              payments.add(state.score(1));
            }
            if (event instanceof Dora)
              Assert.assertEquals(state.doraState().indicatorTileType(1), Tile.P6);
            if (event instanceof EndGame) Assert.assertEquals(state.score(1), 18000);
          }
        });
    Assert.assertEquals(indices, List.of(0, 1, 2, 3, 4, 5, 6, 7));
    Assert.assertEquals(payments, List.of(24000));
  }

  @Test
  public void responseDecisionsReferenceTheirDiscardAndTheUpdatedDoraState() {
    var events = discardedSeven();
    events.add(new Dora(Tile.P6));
    events.add(new Tsumo(2, 27));
    var revisions = new java.util.HashMap<Integer, Integer>();
    var points = new ArrayList<DecisionPoint>();
    ReplayEngine.replay(
        record(events),
        new ReplayObserver() {
          public void onDecision(
              DecisionPoint point,
              com.epsilon.core.GameState state,
              int player,
              List<Action> legal) {
            points.add(point);
            if (point.choiceKind() == DecisionPoint.ChoiceKind.CONFIRMED_PASS) {
              Assert.assertEquals(point.eventIndex(), 5);
              Assert.assertEquals(point.causeEventIndex(), 3);
              Assert.assertEquals(point.stateId(), revisions.get(4).intValue());
              Assert.assertEquals(state.doraState().indicatorCount(), 2);
            } else {
              Assert.assertEquals(point.causeEventIndex(), 2);
              Assert.assertEquals(point.stateId(), revisions.get(2).intValue());
            }
          }

          public void onEventApplied(
              int index, int stateId, ReplayEvent event, com.epsilon.core.GameState state) {
            revisions.put(index, stateId);
          }
        });
    Assert.assertEquals(points.size(), 3);
    Assert.assertTrue(points.get(0).stateId() < points.get(1).stateId());
  }

  @Test
  public void aTruncatedWindowReportsUnobservedChoicesWithoutInventingSelectedSlots() {
    var events = discardedSeven();
    var unresolved = new ArrayList<Integer>();
    var points = new ArrayList<DecisionPoint>();
    ReplayEngine.replay(
        record(events),
        (point, state, player, legal) -> {
          if (point.choiceKind() == DecisionPoint.ChoiceKind.UNOBSERVED) {
            Assert.assertEquals(point.chosenSlot(), -1);
            Assert.assertEquals(point.eventIndex(), events.size());
            Assert.assertEquals(point.causeEventIndex(), 3);
            Assert.assertTrue(legal.size() > 1);
            unresolved.add(player);
            points.add(point);
          }
        });
    Assert.assertEquals(unresolved, List.of(0, 2));
    Assert.assertEquals(points.get(0).stateId(), points.get(1).stateId());
  }

  @Test
  public void aCallLeavesBlockedChoicesUnobservedAndBecomesTheNextDiscardCause() {
    var events = discardedSeven();
    events.add(new Pon(0, 1, 25, new int[] {24, 26}));
    events.add(new Dahai(0, 3, false));
    var callRevision = new ArrayList<Integer>();
    var blocked = new ArrayList<Integer>();
    ReplayEngine.replay(
        record(events),
        new ReplayObserver() {
          public void onDecision(
              DecisionPoint point,
              com.epsilon.core.GameState state,
              int player,
              List<Action> legal) {
            if (point.choiceKind() == DecisionPoint.ChoiceKind.UNOBSERVED
                && point.eventIndex() == 4) {
              blocked.add(player);
              Assert.assertEquals(point.causeEventIndex(), 3);
            }
            if (point.eventIndex() == 5
                && point.choiceKind() == DecisionPoint.ChoiceKind.RECORDED_ACTION) {
              Assert.assertEquals(point.causeEventIndex(), 4);
              Assert.assertEquals(point.stateId(), callRevision.getFirst().intValue());
              Assert.assertEquals(state.hand(player).meldCount(), 1);
            }
          }

          public void onEventApplied(
              int index, int stateId, ReplayEvent event, com.epsilon.core.GameState state) {
            if (index == 4) callRevision.add(stateId);
          }
        });
    Assert.assertEquals(blocked, List.of(2));
    Assert.assertEquals(callRevision.size(), 1);
  }

  @Test
  public void internalFuritenChangesReceiveANewRevisionBeforeTheNextDecision() {
    int[][] hands =
        physicalHands(
            new int[] {3, 3, 3, 4, 5, 12, 13, 14, 21, 22, 23, Tile.NAN, Tile.SHA},
            new int[] {0, 1, 2, 9, 10, 11, 18, 19, 20, Tile.TON, Tile.TON, Tile.TON, Tile.HAKU},
            new int[] {6, 7, 8, 15, 16, 17, 24, 25, 26, Tile.CHUN, Tile.CHUN, Tile.CHUN, Tile.HAKU},
            new int[] {1, 1, 2, 4, 5, 7, 10, 12, 14, 19, 21, Tile.NAN, Tile.PEI});
    var events =
        List.<ReplayEvent>of(
            new StartGame(),
            new StartKyoku(
                Tile.TON, 1, 0, 0, 0, Tile.HATSU, hands, new int[] {25000, 25000, 25000, 25000}),
            new Tsumo(0, 126),
            new Dahai(0, 126, true),
            new Hora(2, 0, Tile.HAKU, new int[] {-3000, 0, 3000, 0}));
    var declined = new ArrayList<Integer>();
    var wins = new ArrayList<Integer>();
    ReplayEngine.replay(
        record(events),
        (point, state, player, legal) -> {
          if (player == 1 && point.choiceKind() == DecisionPoint.ChoiceKind.UNOBSERVED) {
            declined.add(point.stateId());
            Assert.assertFalse(state.isTemporaryFuriten(1));
          }
          if (player == 2 && point.choiceKind() == DecisionPoint.ChoiceKind.RECORDED_ACTION) {
            wins.add(point.stateId());
            Assert.assertEquals(point.causeEventIndex(), 3);
            Assert.assertTrue(state.isTemporaryFuriten(1));
            Assert.assertTrue(point.stateId() > declined.getFirst());
          }
        });
    Assert.assertEquals(declined.size(), 1);
    Assert.assertEquals(wins.size(), 1);
  }

  @org.testng.annotations.DataProvider
  public Object[][] addedKanEndings() {
    return new Object[][] {{false}, {true}};
  }

  @Test(dataProvider = "addedKanEndings")
  public void addedKanResponsesKeepTheirSourceUntilRonOrRinshanIsResolved(boolean robbed) {
    int[][] hands =
        physicalHands(
            new int[] {6, 6, 0, 1, 2, 3, 4, 5, 7, 8, 9, 14, 15},
            new int[] {0, 0, 1, 2, 3, 10, 11, 12, 19, 20, 21, Tile.HAKU, Tile.HAKU},
            new int[] {0, 1, 2, 9, 10, 11, 18, 19, 20, 7, 8, Tile.TON, Tile.TON},
            new int[] {3, 4, 5, 12, 13, 14, 21, 22, 23, Tile.NAN, Tile.NAN, Tile.SHA, Tile.PEI});
    var events =
        new ArrayList<ReplayEvent>(
            List.of(
                new StartGame(),
                new StartKyoku(
                    Tile.TON,
                    1,
                    0,
                    0,
                    1,
                    Tile.HATSU,
                    hands,
                    new int[] {25000, 25000, 25000, 25000}),
                new Tsumo(1, 26),
                new Dahai(1, 26, true),
                new Pon(0, 1, 26, new int[] {24, 25}),
                new Dahai(0, 0, false),
                new Tsumo(1, 18),
                new Dahai(1, 18, true),
                new Tsumo(2, 53),
                new Dahai(2, 53, true),
                new Tsumo(3, 104),
                new Dahai(3, 104, true),
                new Tsumo(0, 27),
                new ReplayEvent.Kakan(0, 27)));
    if (robbed) events.add(new Hora(2, 0, Tile.M7, new int[] {-8000, 0, 8000, 0}));
    else {
      events.add(new Dora(Tile.P6));
      events.add(new Tsumo(0, 58));
      events.add(new Dahai(0, 58, true));
    }
    var kanRevision = new ArrayList<Integer>();
    var responses = new ArrayList<DecisionPoint>();
    ReplayEngine.replay(
        record(events),
        new ReplayObserver() {
          public void onDecision(
              DecisionPoint point,
              com.epsilon.core.GameState state,
              int player,
              List<Action> legal) {
            if (point.causeEventIndex() == 13) {
              responses.add(point);
              Assert.assertEquals(player, 2);
              Assert.assertEquals(point.eventIndex(), 14);
              Assert.assertEquals(point.stateId(), kanRevision.getFirst().intValue());
              Assert.assertEquals(
                  point.choiceKind(),
                  robbed
                      ? DecisionPoint.ChoiceKind.RECORDED_ACTION
                      : DecisionPoint.ChoiceKind.CONFIRMED_PASS);
              Assert.assertTrue(state.getTurnEvent() instanceof TurnEvent.KanAttempt);
              Assert.assertEquals(state.doraState().indicatorCount(), 1);
            }
            if (point.eventIndex() == 16
                && point.choiceKind() == DecisionPoint.ChoiceKind.RECORDED_ACTION) {
              Assert.assertEquals(point.causeEventIndex(), 15);
              Assert.assertTrue(((TurnEvent.Draw) state.getTurnEvent()).isRinshanDraw());
              Assert.assertEquals(state.doraState().indicatorCount(), 2);
            }
          }

          public void onEventApplied(
              int index, int stateId, ReplayEvent event, com.epsilon.core.GameState state) {
            if (index == 13) kanRevision.add(stateId);
            if (index == 14) {
              Assert.assertTrue(stateId > kanRevision.getFirst());
              if (robbed) {
                Assert.assertEquals(state.score(2), 33000);
                Assert.assertTrue(state.getTurnEvent() instanceof TurnEvent.KanAttempt);
              } else {
                Assert.assertTrue(((TurnEvent.Draw) state.getTurnEvent()).isRinshanDraw());
                Assert.assertEquals(state.doraState().indicatorCount(), 2);
              }
            }
          }
        });
    Assert.assertEquals(responses.size(), 1);
  }

  @Test
  public void nineTerminalsReportsTheChosenActionBeforeTheRoundEnds() {
    var points = new ArrayList<DecisionPoint>();
    int[] drawRevision = new int[1];
    ReplayEngine.replay(
        record(nineTerminalsRound("NINE_TERMINALS")),
        new ReplayObserver() {
          public void onDecision(
              DecisionPoint point,
              com.epsilon.core.GameState state,
              int player,
              List<Action> legal) {
            points.add(point);
            Assert.assertEquals(player, 2);
            Assert.assertTrue(legal.size() > 1);
            Assert.assertEquals(legal.get(point.chosenSlot()), Action.kyushuKyuhai());
            Assert.assertEquals(point.choiceKind(), DecisionPoint.ChoiceKind.RECORDED_ACTION);
            Assert.assertEquals(point.eventIndex(), 3);
            Assert.assertEquals(point.causeEventIndex(), 2);
            Assert.assertEquals(point.stateId(), drawRevision[0]);
            Assert.assertEquals(state.hand(player).concealedTileCount(), 14);
            Assert.assertEquals(((TurnEvent.Draw) state.getTurnEvent()).player(), player);
          }

          public void onEventApplied(
              int index, int stateId, ReplayEvent event, com.epsilon.core.GameState state) {
            if (event instanceof Tsumo) drawRevision[0] = stateId;
            if (event instanceof ReplayEvent.Ryukyoku) {
              Assert.assertEquals(points.size(), 1);
              Assert.assertTrue(stateId > points.getFirst().stateId());
            }
          }
        });
    Assert.assertEquals(points.size(), 1);
  }

  @org.testng.annotations.DataProvider
  public Object[][] nonPlayerSelectedDrawReasons() {
    return new Object[][] {
      {null},
      {"EXHAUSTIVE_DRAW"},
      {"FOUR_RIICHI"},
      {"THREE_RON"},
      {"FOUR_KANS"},
      {"FOUR_WINDS"},
      {"MAHJONG_SOUL_99"}
    };
  }

  @Test(dataProvider = "nonPlayerSelectedDrawReasons")
  public void otherDrawReasonsDoNotInventAChoiceEvenWhenNineTerminalsIsLegal(String reason) {
    ReplayEngine.replay(
        record(nineTerminalsRound(reason)),
        (point, state, player, legal) ->
            Assert.fail("An automatic or unspecified draw must not create a player decision"));
  }

  @Test
  public void anIllegalNineTerminalsDeclarationFailsBeforeReachingTheConsumer() {
    var events = discardedSeven();
    events.set(3, new ReplayEvent.Ryukyoku(new int[4], "NINE_TERMINALS"));
    Assert.expectThrows(
        IllegalStateException.class,
        () ->
            ReplayEngine.replay(
                record(events),
                (point, state, player, legal) ->
                    Assert.fail("An illegal abortive draw must not reach the consumer")));
  }

  private static List<ReplayEvent> nineTerminalsRound(String reason) {
    int[][] hands =
        physicalHands(
            new int[] {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15},
            new int[] {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15},
            new int[] {0, 8, 9, 17, 18, 26, Tile.TON, Tile.NAN, Tile.SHA, 1, 2, 3, 4},
            new int[] {5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 19, 20, 21});
    return List.of(
        new StartGame(),
        new StartKyoku(
            Tile.TON, 3, 0, 0, 2, Tile.HAKU, hands, new int[] {25000, 25000, 25000, 25000}),
        new Tsumo(2, Tile.CHUN * Tile.TILES_PER_TYPE),
        new ReplayEvent.Ryukyoku(new int[4], reason));
  }

  private static ReplayRecord record(List<ReplayEvent> events) {
    return new ReplayRecord(events, null, null, RecordCompletion.INCOMPLETE);
  }

  /** 学習利用側と同じく、選択が確定した判断だけを既存の教師不変条件へ渡す。 */
  private static int[] replaySelected(List<ReplayEvent> events, SelectedDecision assertion) {
    return ReplayEngine.replay(
        record(events),
        (point, state, player, legal) -> {
          if (point.choiceKind() != DecisionPoint.ChoiceKind.UNOBSERVED)
            assertion.accept(state, player, legal, point.chosenSlot());
        });
  }

  @FunctionalInterface
  private interface SelectedDecision {
    void accept(com.epsilon.core.GameState state, int player, List<Action> legal, int chosenSlot);
  }

  private static ArrayList<ReplayEvent> discardedSeven() {
    return new ArrayList<>(
        List.of(
            new StartGame(),
            new StartKyoku(
                Tile.TON,
                1,
                0,
                0,
                1,
                Tile.HAKU,
                callHands(),
                new int[] {25000, 25000, 25000, 25000}),
            new Tsumo(1, 25),
            new Dahai(1, 25, true)));
  }

  /** 席1の七萬を席0がポン、席2がチーできる配牌。 */
  private static int[][] callHands() {
    return new int[][] {
      {24, 26, 3, 5, 9, 13, 17, 21, 28, 32, 36, 56, 60},
      {0, 1, 2, 4, 8, 12, 40, 44, 48, 76, 80, 84, 124},
      {6, 10, 14, 18, 22, 29, 33, 37, 41, 45, 49, 53, 57},
      {7, 11, 15, 19, 23, 30, 34, 38, 42, 46, 50, 54, 58}
    };
  }

  /** 席間で物理牌が重複しないよう、牌種ごとに未使用の牌を割り当てる。 */
  private static int[][] physicalHands(int[]... tileTypesBySeat) {
    int[] copies = new int[Tile.NUM_TILE_TYPES];
    int[][] result = new int[tileTypesBySeat.length][];
    for (int seat = 0; seat < result.length; seat++) {
      result[seat] = new int[tileTypesBySeat[seat].length];
      for (int index = 0; index < result[seat].length; index++) {
        int type = tileTypesBySeat[seat][index];
        result[seat][index] = type * Tile.TILES_PER_TYPE + copies[type]++;
      }
    }
    return result;
  }
}
