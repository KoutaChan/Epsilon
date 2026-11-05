package com.epsilon.replay;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.ActionGenerator;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.engine.GoldenGames;
import com.epsilon.replay.ReplayDecision.Choice;
import com.epsilon.replay.ReplayEvent.Chi;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.Dora;
import com.epsilon.replay.ReplayEvent.EndGame;
import com.epsilon.replay.ReplayEvent.EndKyoku;
import com.epsilon.replay.ReplayEvent.Hora;
import com.epsilon.replay.ReplayEvent.Kakan;
import com.epsilon.replay.ReplayEvent.None;
import com.epsilon.replay.ReplayEvent.Pon;
import com.epsilon.replay.ReplayEvent.Reach;
import com.epsilon.replay.ReplayEvent.ReachAccepted;
import com.epsilon.replay.ReplayEvent.Ryukyoku;
import com.epsilon.replay.ReplayEvent.StartGame;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
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

  @DataProvider
  public Object[][] lowScoreRiichiCases() {
    return new Object[][] {
      {500, true}, {0, true}, {-3500, true},
      {500, false}, {0, false}, {-3500, false}
    };
  }

  @Test(dataProvider = "lowScoreRiichiCases")
  public void onlyRecordedLowScoreRiichiIsAcceptedWithoutChangingScores(
      int startingScore, boolean recordedRiichi) {
    int[] startingScores = {25000, startingScore, 25000, 25000};
    var events =
        new ArrayList<ReplayEvent>(
            List.of(
                new StartGame(),
                new StartKyoku(Tile.TON, 1, 0, 0, 1, Tile.HAKU, callHands(), startingScores),
                new Tsumo(1, 25)));
    if (recordedRiichi) events.add(new Reach(1));
    events.add(new Dahai(1, 25, true));
    if (recordedRiichi) events.add(new ReachAccepted(1));
    var choices = new ArrayList<Action.Type>();
    int[] scores =
        replaySelected(
            events,
            (state, player, legal, chosenSlot) -> {
              if (player != 1) return;
              Assert.assertEquals(state.getScore(player), startingScore);
              var ordinaryActions = new ArrayList<Action>();
              new ActionGenerator()
                  .generateTurnActionsInto(
                      ordinaryActions, state, player, (TurnEvent.Draw) state.getTurnEvent());
              Assert.assertTrue(
                  ordinaryActions.stream().noneMatch(a -> a.type() == Action.Type.RIICHI_DAHAI));
              Assert.assertEquals(
                  legal.stream().anyMatch(a -> a.type() == Action.Type.RIICHI_DAHAI),
                  recordedRiichi);
              choices.add(legal.get(chosenSlot).type());
            });
    Assert.assertEquals(
        choices, List.of(recordedRiichi ? Action.Type.RIICHI_DAHAI : Action.Type.DAHAI));
    Assert.assertEquals(scores[1], startingScore - (recordedRiichi ? 1000 : 0));
    Assert.assertEquals(startingScores[1], startingScore);
  }

  @Test
  public void recordedLowScoreRiichiStillRequiresTenpai() {
    var events =
        List.<ReplayEvent>of(
            new StartGame(),
            new StartKyoku(
                Tile.TON,
                1,
                0,
                0,
                1,
                Tile.HAKU,
                callHands(),
                new int[] {25000, -3500, 25000, 25000}),
            new Tsumo(1, 25),
            new Reach(1),
            new Dahai(1, 0, false));
    Assert.expectThrows(
        IllegalStateException.class,
        () ->
            replaySelected(
                events,
                (state, player, legal, slot) ->
                    Assert.fail("A non-tenpai riichi must not reach the consumer")));
  }

  @Test
  public void bothRonCallbacksSeeTheUnsettledScoresBeforeTheirDeltasAreCombined() {
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
                    ronHands(),
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
          @Override
          public void onDecision(ReplayDecision decision, GameState state, List<Action> legal) {
            int chosen = decision.chosenSlot();
            if (chosen >= 0 && legal.get(chosen).type() == Action.Type.RIICHI_DAHAI) {
              Assert.assertEquals(decision.resolveIndex(), 4);
              Assert.assertEquals(state.getKyotakuCount(), 0);
            }
          }

          @Override
          public void onEventApplied(int index, int revision, ReplayEvent event, GameState state) {
            indices.add(index);
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
    var revisions = new HashMap<Integer, Integer>();
    var decisions = new ArrayList<ReplayDecision>();
    ReplayEngine.replay(
        record(events),
        new ReplayObserver() {
          @Override
          public void onDecision(ReplayDecision decision, GameState state, List<Action> legal) {
            decisions.add(decision);
            if (decision.choice() == Choice.CONFIRMED_PASS) {
              Assert.assertEquals(decision.resolveIndex(), 5);
              Assert.assertEquals(decision.causeIndex(), 3);
              Assert.assertEquals(decision.revision(), revisions.get(4).intValue());
              Assert.assertEquals(state.doraState().indicatorCount(), 2);
            } else {
              Assert.assertEquals(decision.causeIndex(), 2);
              Assert.assertEquals(decision.revision(), revisions.get(2).intValue());
            }
          }

          @Override
          public void onEventApplied(int index, int revision, ReplayEvent event, GameState state) {
            revisions.put(index, revision);
          }
        });
    Assert.assertEquals(decisions.size(), 3);
    Assert.assertTrue(decisions.get(0).revision() < decisions.get(1).revision());
  }

  @Test
  public void aTruncatedWindowReportsUnobservedChoicesWithoutInventingSelectedSlots() {
    var events = discardedSeven();
    var unresolved = new ArrayList<Integer>();
    var decisions = new ArrayList<ReplayDecision>();
    ReplayEngine.replay(
        record(events),
        (decision, state, legal) -> {
          if (decision.choice() == Choice.UNOBSERVED) {
            Assert.assertEquals(decision.chosenSlot(), -1);
            Assert.assertEquals(decision.resolveIndex(), events.size());
            Assert.assertEquals(decision.causeIndex(), 3);
            Assert.assertTrue(legal.size() > 1);
            unresolved.add(decision.seat());
            decisions.add(decision);
          }
        });
    Assert.assertEquals(unresolved, List.of(0, 2));
    Assert.assertEquals(decisions.get(0).revision(), decisions.get(1).revision());
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
          @Override
          public void onDecision(ReplayDecision decision, GameState state, List<Action> legal) {
            if (decision.choice() == Choice.UNOBSERVED && decision.resolveIndex() == 4) {
              blocked.add(decision.seat());
              Assert.assertEquals(decision.causeIndex(), 3);
            }
            if (decision.resolveIndex() == 5 && decision.choice() == Choice.RECORDED_ACTION) {
              Assert.assertEquals(decision.causeIndex(), 4);
              Assert.assertEquals(decision.revision(), callRevision.getFirst().intValue());
              Assert.assertEquals(state.hand(decision.seat()).meldCount(), 1);
              Assert.assertEquals(decision.kind(), EngineDecisionKind.POST_CALL_DAHAI);
            }
          }

          @Override
          public void onEventApplied(int index, int revision, ReplayEvent event, GameState state) {
            if (index == 4) callRevision.add(revision);
          }
        });
    Assert.assertEquals(blocked, List.of(2));
    Assert.assertEquals(callRevision.size(), 1);
  }

  @Test
  public void internalFuritenChangesReceiveANewRevisionBeforeTheNextDecision() {
    var events =
        List.<ReplayEvent>of(
            new StartGame(),
            new StartKyoku(
                Tile.TON,
                1,
                0,
                0,
                0,
                Tile.HATSU,
                ronHands(),
                new int[] {25000, 25000, 25000, 25000}),
            new Tsumo(0, 126),
            new Dahai(0, 126, true),
            new Hora(2, 0, Tile.HAKU, new int[] {-3000, 0, 3000, 0}));
    var declined = new ArrayList<Integer>();
    var wins = new ArrayList<Integer>();
    ReplayEngine.replay(
        record(events),
        (decision, state, legal) -> {
          if (decision.seat() == 1 && decision.choice() == Choice.UNOBSERVED) {
            declined.add(decision.revision());
            Assert.assertFalse(state.isTemporaryFuriten(1));
          }
          if (decision.seat() == 2 && decision.choice() == Choice.RECORDED_ACTION) {
            wins.add(decision.revision());
            Assert.assertEquals(decision.causeIndex(), 3);
            Assert.assertTrue(state.isTemporaryFuriten(1));
            Assert.assertTrue(decision.revision() > declined.getFirst());
          }
        });
    Assert.assertEquals(declined.size(), 1);
    Assert.assertEquals(wins.size(), 1);
  }

  @DataProvider
  public Object[][] addedKanEndings() {
    return new Object[][] {{false}, {true}};
  }

  @Test(dataProvider = "addedKanEndings")
  public void addedKanResponsesKeepTheirSourceUntilRonOrRinshanIsResolved(boolean robbed) {
    var events = addedKan();
    if (robbed) events.add(new Hora(2, 0, Tile.M7, new int[] {-8000, 0, 8000, 0}));
    else {
      events.add(new Dora(Tile.P6));
      events.add(new Tsumo(0, 58));
      events.add(new Dahai(0, 58, true));
    }
    var kanRevision = new ArrayList<Integer>();
    var responses = new ArrayList<ReplayDecision>();
    ReplayEngine.replay(
        record(events),
        new ReplayObserver() {
          @Override
          public void onDecision(ReplayDecision decision, GameState state, List<Action> legal) {
            if (decision.causeIndex() == 13) {
              responses.add(decision);
              Assert.assertEquals(decision.seat(), 2);
              Assert.assertEquals(decision.resolveIndex(), 14);
              Assert.assertEquals(decision.revision(), kanRevision.getFirst().intValue());
              Assert.assertEquals(
                  decision.choice(), robbed ? Choice.RECORDED_ACTION : Choice.CONFIRMED_PASS);
              Assert.assertEquals(decision.kind(), EngineDecisionKind.CHANKAN);
              Assert.assertTrue(state.getTurnEvent() instanceof TurnEvent.KanAttempt);
              Assert.assertEquals(state.doraState().indicatorCount(), 1);
            }
            if (decision.resolveIndex() == 16 && decision.choice() == Choice.RECORDED_ACTION) {
              Assert.assertEquals(decision.causeIndex(), 15);
              Assert.assertTrue(((TurnEvent.Draw) state.getTurnEvent()).isRinshanDraw());
              Assert.assertEquals(state.doraState().indicatorCount(), 2);
            }
          }

          @Override
          public void onEventApplied(int index, int revision, ReplayEvent event, GameState state) {
            if (index == 13) kanRevision.add(revision);
            if (index == 14) {
              Assert.assertTrue(revision > kanRevision.getFirst());
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
  public void anOpenKanDoraBeforeItsDahaiOpensAfterTheDiscard() {
    var events = addedKan();
    events.add(new Tsumo(0, 58));
    events.add(new Dora(Tile.P6));
    events.add(new Dahai(0, 58, true));
    var indicators = new ArrayList<String>();
    ReplayEngine.replay(
        record(events),
        new ReplayObserver() {
          @Override
          public void onDecision(ReplayDecision decision, GameState state, List<Action> legal) {
            if (decision.choice() != Choice.RECORDED_ACTION || decision.resolveIndex() != 16)
              return;
            Assert.assertTrue(((TurnEvent.Draw) state.getTurnEvent()).doraRevealPending());
            indicators.add("decision " + state.doraState().indicatorCount());
          }

          @Override
          public void onEventApplied(int index, int revision, ReplayEvent event, GameState state) {
            if (index >= 15) indicators.add(index + " " + state.doraState().indicatorCount());
          }
        });
    Assert.assertEquals(indicators, List.of("15 1", "decision 1", "16 2"));
  }

  @Test
  public void nineTerminalsReportsTheChosenActionBeforeTheRoundEnds() {
    var decisions = new ArrayList<ReplayDecision>();
    int[] drawRevision = new int[1];
    ReplayEngine.replay(
        record(nineTerminalsRound(Ryukyoku.NINE_TERMINALS)),
        new ReplayObserver() {
          @Override
          public void onDecision(ReplayDecision decision, GameState state, List<Action> legal) {
            decisions.add(decision);
            Assert.assertEquals(decision.seat(), 2);
            Assert.assertTrue(legal.size() > 1);
            Assert.assertEquals(legal.get(decision.chosenSlot()), Action.kyushuKyuhai());
            Assert.assertEquals(decision.choice(), Choice.RECORDED_ACTION);
            Assert.assertEquals(decision.resolveIndex(), 3);
            Assert.assertEquals(decision.causeIndex(), 2);
            Assert.assertEquals(decision.revision(), drawRevision[0]);
            Assert.assertEquals(state.hand(decision.seat()).concealedTileCount(), 14);
            Assert.assertEquals(((TurnEvent.Draw) state.getTurnEvent()).player(), decision.seat());
          }

          @Override
          public void onEventApplied(int index, int revision, ReplayEvent event, GameState state) {
            if (event instanceof Tsumo) drawRevision[0] = revision;
            if (event instanceof Ryukyoku) {
              Assert.assertEquals(decisions.size(), 1);
              Assert.assertTrue(revision > decisions.getFirst().revision());
            }
          }
        });
    Assert.assertEquals(decisions.size(), 1);
  }

  @DataProvider
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
        (decision, state, legal) ->
            Assert.fail("An automatic or unspecified draw must not create a player decision"));
  }

  @Test
  public void anIllegalNineTerminalsDeclarationFailsBeforeReachingTheConsumer() {
    var events = discardedSeven();
    events.set(3, new Ryukyoku(new int[4], Ryukyoku.NINE_TERMINALS));
    Assert.expectThrows(
        IllegalStateException.class,
        () ->
            ReplayEngine.replay(
                record(events),
                (decision, state, legal) ->
                    Assert.fail("An illegal abortive draw must not reach the consumer")));
  }

  @Test
  public void anExplicitPassBeforeACallIsResolvedAtTheRunStartAndNotifiedInOrder() {
    var events = discardedSeven();
    events.add(new None(2));
    events.add(new Pon(0, 1, 25, new int[] {24, 26}));
    events.add(new Dahai(0, 3, false));
    Trace trace = trace(events);
    Assert.assertEquals(
        trace.lines,
        List.of(
            "E 0 StartGame",
            "E 1 StartKyoku",
            "E 2 Tsumo",
            "D 1 RECORDED_ACTION TURN 3<2",
            "E 3 Dahai",
            "D 2 CONFIRMED_PASS RESPONSE 4<3",
            "E 4 None",
            "D 0 RECORDED_ACTION RESPONSE 5<3",
            "E 5 Pon",
            "D 0 RECORDED_ACTION POST_CALL_DAHAI 6<5",
            "E 6 Dahai",
            "D 1 UNOBSERVED RESPONSE 7<6"));
    List<Integer> revisions = trace.revisions;
    for (int line = 5; line <= 7; line++)
      Assert.assertEquals(revisions.get(line), revisions.get(4), "line " + line);
    Assert.assertTrue(revisions.get(8) > revisions.get(4));
    Assert.assertEquals(revisions.get(9), revisions.get(8));
  }

  @DataProvider
  public Object[][] runEndings() {
    return new Object[][] {{false}, {true}};
  }

  @Test(dataProvider = "runEndings")
  public void explicitPassesAreResolvedAtTheStartOfTheirRun(boolean drawFollows) {
    var events = discardedSeven();
    events.add(new None(2));
    events.add(new None(0));
    if (drawFollows) events.add(new Tsumo(2, 27));
    Trace trace = trace(events);
    var expected =
        new ArrayList<>(
            List.of(
                "E 0 StartGame",
                "E 1 StartKyoku",
                "E 2 Tsumo",
                "D 1 RECORDED_ACTION TURN 3<2",
                "E 3 Dahai",
                "D 0 CONFIRMED_PASS RESPONSE 4<3",
                "D 2 CONFIRMED_PASS RESPONSE 4<3",
                "E 4 None",
                "E 5 None"));
    if (drawFollows) expected.add("E 6 Tsumo");
    Assert.assertEquals(trace.lines, expected);
    for (int line = 5; line <= 8; line++)
      Assert.assertEquals(trace.revisions.get(line), trace.revisions.get(4), "line " + line);
  }

  @Test(dataProvider = "runEndings")
  public void aSeatThatDidNotPassIsUnobservedOnlyWhenTheRecordEnds(boolean drawFollows) {
    var events = discardedSeven();
    events.add(new None(0));
    if (drawFollows) events.add(new Tsumo(2, 27));
    Trace trace = trace(events);
    Assert.assertEquals(trace.lines.get(5), "D 0 CONFIRMED_PASS RESPONSE 4<3");
    Assert.assertEquals(
        trace.lines.get(6),
        drawFollows ? "D 2 CONFIRMED_PASS RESPONSE 4<3" : "D 2 UNOBSERVED RESPONSE 4<3");
  }

  @Test
  public void aPassOutsideAWindowIsNotifiedWithoutANewRevision() {
    var events = discardedSeven();
    events.add(3, new None(3));
    Trace trace = trace(events);
    Assert.assertEquals(
        trace.lines.subList(0, 6),
        List.of(
            "E 0 StartGame",
            "E 1 StartKyoku",
            "E 2 Tsumo",
            "E 3 None",
            "D 1 RECORDED_ACTION TURN 4<2",
            "E 4 Dahai"));
    Assert.assertEquals(trace.revisions.get(3), trace.revisions.get(2));
    Assert.assertEquals(trace.revisions.get(4), trace.revisions.get(2));
  }

  @Test
  public void aPassFromASeatWithoutADecisionClosesTheWindowAndKeepsTheRevision() {
    var events = discardedSeven();
    events.add(new None(3));
    Trace trace = trace(events);
    Assert.assertEquals(
        trace.lines.subList(5, trace.lines.size()),
        List.of("D 0 CONFIRMED_PASS RESPONSE 4<3", "D 2 CONFIRMED_PASS RESPONSE 4<3", "E 4 None"));
    Assert.assertEquals(trace.revisions.get(7), trace.revisions.get(4));
  }

  @Test
  public void aPassAfterARonIsNotifiedAfterTheRonIsSettled() {
    Trace trace =
        trace(
            List.of(
                new StartGame(),
                new StartKyoku(
                    Tile.TON,
                    1,
                    0,
                    0,
                    0,
                    Tile.HATSU,
                    ronHands(),
                    new int[] {25000, 25000, 25000, 25000}),
                new Tsumo(0, 126),
                new Dahai(0, 126, true),
                new Hora(2, 0, Tile.HAKU, new int[] {-3000, 0, 3000, 0}),
                new None(1)));
    Assert.assertEquals(
        trace.lines,
        List.of(
            "E 0 StartGame",
            "E 1 StartKyoku",
            "E 2 Tsumo",
            "D 0 RECORDED_ACTION TURN 3<2",
            "E 3 Dahai",
            "D 1 CONFIRMED_PASS RESPONSE 4<3",
            "D 2 RECORDED_ACTION RESPONSE 4<3",
            "E 4 Hora",
            "E 5 None"));
  }

  @Test
  public void aTurnDecisionWithoutADrawFailsAtItsEvent() {
    var events = new ArrayList<>(discardedSeven().subList(0, 2));
    events.add(new Dahai(1, 0, false));
    Trace trace = new Trace();
    var failure =
        Assert.expectThrows(
            IllegalStateException.class, () -> ReplayEngine.replay(record(events), trace));
    Assert.assertEquals(
        failure.getMessage(),
        "Event 2 is a turn decision, but the last turn event is None, not a draw.");
    Assert.assertEquals(trace.lines, List.of("E 0 StartGame", "E 1 StartKyoku"));
  }

  @Test
  public void aSecondCallOnOneDiscardFailsAfterItsDecision() {
    var events = discardedSeven();
    events.add(new Pon(0, 1, 25, new int[] {24, 26}));
    events.add(new Chi(2, 1, 25, new int[] {18, 22}));
    Trace trace = new Trace();
    var failure =
        Assert.expectThrows(
            IllegalStateException.class, () -> ReplayEngine.replay(record(events), trace));
    Assert.assertEquals(
        failure.getMessage(),
        "Event 5 calls a tile, but the last turn event is None, not a discard.");
    Assert.assertEquals(
        trace.lines.subList(5, trace.lines.size()),
        List.of("D 0 RECORDED_ACTION RESPONSE 4<3", "E 4 Pon", "D 2 RECORDED_ACTION RESPONSE 5<3"));
  }

  @DataProvider
  public Object[][] responsesOutsideAWindow() {
    var ponBeforeTheDiscard = new ArrayList<>(discardedSeven().subList(0, 3));
    ponBeforeTheDiscard.add(new Pon(0, 1, 25, new int[] {24, 26}));
    var ponWithoutADecision = discardedSeven();
    ponWithoutADecision.add(new Pon(3, 1, 25, new int[] {27, 31}));
    var ronBeforeTheDiscard = new ArrayList<>(discardedSeven().subList(0, 3));
    ronBeforeTheDiscard.add(new Hora(2, 1, Tile.M7, new int[] {0, -1000, 1000, 0}));
    List<String> untilTheDraw = List.of("E 0 StartGame", "E 1 StartKyoku", "E 2 Tsumo");
    return new Object[][] {
      {ponBeforeTheDiscard, "Response event outside response window: Pon[", untilTheDraw},
      {
        ponWithoutADecision,
        "Response event outside response window: Pon[",
        List.of(
            "E 0 StartGame",
            "E 1 StartKyoku",
            "E 2 Tsumo",
            "D 1 RECORDED_ACTION TURN 3<2",
            "E 3 Dahai",
            "D 0 CONFIRMED_PASS RESPONSE 4<3",
            "D 2 CONFIRMED_PASS RESPONSE 4<3")
      },
      {ronBeforeTheDiscard, "Ron event outside response window: Hora[", untilTheDraw}
    };
  }

  /** 失敗した出来事の通知も、それを resolve とする牌譜の手の判断も届かない（lines は失敗までの全ての写し）。 */
  @Test(dataProvider = "responsesOutsideAWindow")
  public void aResponseOutsideAWindowFailsBeforeItReachesTheObserver(
      List<ReplayEvent> events, String message, List<String> lines) {
    Trace trace = new Trace();
    var failure =
        Assert.expectThrows(
            IllegalStateException.class, () -> ReplayEngine.replay(record(events), trace));
    Assert.assertTrue(failure.getMessage().startsWith(message), failure.getMessage());
    Assert.assertEquals(trace.lines, lines);
  }

  /** 点の無い局の始めは前の局の点を引き継ぐ。2 局目のロンの判断は持ち越した供託を見て、和了の通知は受け取った後の供託 0 を見る。 */
  @Test
  public void scoresCarryAcrossRoundsWhenTheEndGameHasNoScores() {
    var events =
        List.<ReplayEvent>of(
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
            new Ryukyoku(new int[] {-1000, 3000, -1000, -1000}, "EXHAUSTIVE_DRAW"),
            new EndKyoku(),
            new StartKyoku(Tile.TON, 2, 1, 1, 0, Tile.HATSU, ronHands(), null),
            new Tsumo(0, 126),
            new Dahai(0, 126, true),
            new Hora(2, 0, Tile.HAKU, new int[] {-4000, 0, 5000, 0}),
            new EndKyoku(),
            new EndGame(null));
    var ronKyotaku = new ArrayList<Integer>();
    var settledKyotaku = new ArrayList<Integer>();
    ReplayObserver observer =
        new ReplayObserver() {
          @Override
          public void onDecision(ReplayDecision decision, GameState state, List<Action> legal) {
            if (decision.seat() == 2 && decision.resolveIndex() == 11)
              ronKyotaku.add(state.getKyotakuCount());
          }

          @Override
          public void onEventApplied(int index, int revision, ReplayEvent event, GameState state) {
            if (event instanceof Hora) settledKyotaku.add(state.getKyotakuCount());
          }
        };
    int[] scores = ReplayEngine.replay(record(events), observer);
    Assert.assertEquals(ronKyotaku, List.of(1));
    Assert.assertEquals(settledKyotaku, List.of(0));
    Assert.assertEquals(scores, new int[] {20000, 27000, 29000, 24000});
  }

  @Test
  public void aStartGameAfterARoundRestartsTheScoresFromTheStartingPoints() {
    var events =
        List.<ReplayEvent>of(
            new StartGame(),
            new StartKyoku(
                Tile.TON,
                1,
                0,
                0,
                1,
                Tile.HAKU,
                callHands(),
                new int[] {30000, 20000, 25000, 25000}),
            new EndKyoku(),
            new StartGame(),
            new Ryukyoku(new int[] {1000, -1000, 0, 0}));
    var drawScores = new ArrayList<Integer>();
    ReplayObserver observer =
        new ReplayObserver() {
          @Override
          public void onDecision(ReplayDecision decision, GameState state, List<Action> legal) {}

          @Override
          public void onEventApplied(int index, int revision, ReplayEvent event, GameState state) {
            if (event instanceof Ryukyoku)
              for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++)
                drawScores.add(state.getScore(seat));
          }
        };
    int[] scores = ReplayEngine.replay(record(events), observer);
    Assert.assertEquals(drawScores, List.of(26000, 24000, 25000, 25000));
    Assert.assertEquals(scores, new int[] {26000, 24000, 25000, 25000});
  }

  @DataProvider
  public Object[][] observerFailures() {
    return new Object[][] {{true}, {false}};
  }

  @Test(dataProvider = "observerFailures")
  public void anObserverExceptionLeavesTheReplayUnchanged(boolean fromDecision) {
    var marker = new RuntimeException("observer failure");
    ReplayObserver observer =
        new ReplayObserver() {
          @Override
          public void onDecision(ReplayDecision decision, GameState state, List<Action> legal) {
            if (fromDecision) throw marker;
          }

          @Override
          public void onEventApplied(int index, int revision, ReplayEvent event, GameState state) {
            if (!fromDecision) throw marker;
          }
        };
    var thrown =
        Assert.expectThrows(
            RuntimeException.class, () -> ReplayEngine.replay(record(discardedSeven()), observer));
    Assert.assertSame(thrown, marker);
  }

  @Test
  public void everyGoldenDecisionKindMatchesThePublicObservation() {
    for (String name : GoldenGames.records()) {
      ReplayEngine.replay(
          ReplayRecordReader.readRecord(GoldenGames.recordBytes(name)),
          (decision, state, legal) ->
              Assert.assertEquals(
                  decision.kind(),
                  EngineDecisionKind.of(state.publicObservation(decision.seat()), legal),
                  name));
    }
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
        new Ryukyoku(new int[4], reason));
  }

  /** 席0がポンした七萬を加槓するまで（加槓は出来事 13）。 */
  private static ArrayList<ReplayEvent> addedKan() {
    int[][] hands =
        physicalHands(
            new int[] {6, 6, 0, 1, 2, 3, 4, 5, 7, 8, 9, 14, 15},
            new int[] {0, 0, 1, 2, 3, 10, 11, 12, 19, 20, 21, Tile.HAKU, Tile.HAKU},
            new int[] {0, 1, 2, 9, 10, 11, 18, 19, 20, 7, 8, Tile.TON, Tile.TON},
            new int[] {3, 4, 5, 12, 13, 14, 21, 22, 23, Tile.NAN, Tile.NAN, Tile.SHA, Tile.PEI});
    return new ArrayList<>(
        List.of(
            new StartGame(),
            new StartKyoku(
                Tile.TON, 1, 0, 0, 1, Tile.HATSU, hands, new int[] {25000, 25000, 25000, 25000}),
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
            new Kakan(0, 27)));
  }

  private static ReplayRecord record(List<ReplayEvent> events) {
    return new ReplayRecord(events, null, null, RecordCompletion.INCOMPLETE);
  }

  /** 学習利用側と同じく、選択が確定した判断だけを既存の教師不変条件へ渡す。 */
  private static int[] replaySelected(List<ReplayEvent> events, SelectedDecision assertion) {
    return ReplayEngine.replay(
        record(events),
        (decision, state, legal) -> {
          if (decision.choice() != Choice.UNOBSERVED)
            assertion.accept(state, decision.seat(), legal, decision.chosenSlot());
        });
  }

  @FunctionalInterface
  private interface SelectedDecision {
    void accept(GameState state, int player, List<Action> legal, int chosenSlot);
  }

  /** 席 1 が七萬をツモ切りするまで（打牌は出来事 3）。 */
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

  /** 席 0 が白を打てば席 1・2 がロンできる配牌（2 人とも白単騎）。 */
  private static int[][] ronHands() {
    return physicalHands(
        new int[] {3, 3, 3, 4, 5, 12, 13, 14, 21, 22, 23, Tile.NAN, Tile.SHA},
        new int[] {0, 1, 2, 9, 10, 11, 18, 19, 20, Tile.TON, Tile.TON, Tile.TON, Tile.HAKU},
        new int[] {6, 7, 8, 15, 16, 17, 24, 25, 26, Tile.CHUN, Tile.CHUN, Tile.CHUN, Tile.HAKU},
        new int[] {1, 1, 2, 4, 5, 7, 10, 12, 14, 19, 21, Tile.NAN, Tile.PEI});
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

  /**
   * 判断と通知を 1 行ずつ写す。判断は {@code "D 席 選択 場面 resolve<cause"}、通知は {@code "E 位置 型"}。改訂は同じ順で
   * revisions に入る。
   */
  private static final class Trace implements ReplayObserver {
    final List<String> lines = new ArrayList<>();
    final List<Integer> revisions = new ArrayList<>();

    @Override
    public void onDecision(ReplayDecision decision, GameState state, List<Action> legal) {
      lines.add(
          "D "
              + decision.seat()
              + " "
              + decision.choice()
              + " "
              + decision.kind()
              + " "
              + decision.resolveIndex()
              + "<"
              + decision.causeIndex());
      revisions.add(decision.revision());
    }

    @Override
    public void onEventApplied(int index, int revision, ReplayEvent event, GameState state) {
      lines.add("E " + index + " " + event.getClass().getSimpleName());
      revisions.add(revision);
    }
  }

  private static Trace trace(List<ReplayEvent> events) {
    Trace trace = new Trace();
    ReplayEngine.replay(record(events), trace);
    return trace;
  }
}
