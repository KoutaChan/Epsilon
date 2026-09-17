package com.epsilon.reviewer.engine;

import com.epsilon.core.*;
import com.epsilon.replay.*;
import com.epsilon.reviewer.dto.*;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParser;
import java.nio.charset.StandardCharsets;
import java.util.*;
import org.testng.Assert;
import org.testng.annotations.Test;

/** チェックポイントと差分から局面表示を復元し、JSON保存後もイベント順序と赤牌・副露・点数の変化を保つことを検証する。 */
public class ReviewStateSequenceTest {
  private static final Gson JSON = new GsonBuilder().serializeNulls().create();

  @Test
  public void everyEventAndDecisionRestoresExactlyAcrossCheckpointBoundariesAndJson() {
    var record =
        new ReviewEngine()
            .prepareRecord(
                ReviewEngineTest.riverFixture().getBytes(StandardCharsets.UTF_8), "fixture.mjai");
    var states = new ReviewStateSequence();
    Map<Integer, TableState> expected = new LinkedHashMap<>();
    List<ReplayStep> steps = new ArrayList<>();
    ReplayEngine.replay(
        record.replay(),
        new ReplayObserver() {
          public void onDecision(
              DecisionPoint point, GameState state, int seat, List<Action> legal) {
            capture(point.stateId(), state);
            List<ActionCandidate> candidates = new ArrayList<>();
            for (int i = 0; i < legal.size(); i++)
              candidates.add(
                  ReviewSnapshots.candidate(
                      legal.get(i), 1f / legal.size(), i == point.chosenSlot(), state, seat));
            steps.add(
                new ReplayStep(
                    steps.size(),
                    point.eventIndex(),
                    point.stateId(),
                    "decision",
                    seat,
                    true,
                    point.causeEventIndex(),
                    candidates,
                    null));
          }

          public void onEventApplied(int index, int stateId, ReplayEvent event, GameState state) {
            if (event instanceof ReplayEvent.StartGame) return;
            capture(stateId, state);
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

          private void capture(int id, GameState state) {
            states.captureState(id, state);
            expected.putIfAbsent(id, ReviewSnapshots.table(state));
          }
        });
    var round = new ReplayRound("round-0", 0, 0, 0, states.frames(), steps);
    String encoded = JSON.toJson(round);
    var wire = JsonParser.parseString(encoded).getAsJsonObject();
    var firstStep = wire.getAsJsonArray("steps").get(0).getAsJsonObject();
    Assert.assertFalse(
        firstStep.has("players"), "Steps must reference states instead of embedding them");
    Assert.assertFalse(firstStep.has("dora"));
    var firstPlayer =
        wire.getAsJsonArray("states")
            .get(0)
            .getAsJsonObject()
            .getAsJsonObject("checkpoint")
            .getAsJsonArray("players")
            .get(0)
            .getAsJsonObject();
    Assert.assertFalse(firstPlayer.has("name"), "Names belong only to match metadata");
    Assert.assertFalse(firstPlayer.has("seat"), "Player positions identify their seats");
    var restored = JSON.fromJson(encoded, ReplayRound.class);
    Assert.assertEquals(restored, round);
    Assert.assertEquals(restored.states().size(), expected.size());
    Assert.assertTrue(restored.states().size() > 96, "The fixture must span several checkpoints");
    Assert.assertTrue(
        restored.steps().size() > restored.states().size(), "Decisions must share event states");
    for (int index = 0; index < restored.states().size(); index++) {
      StateFrame frame = restored.states().get(index);
      Assert.assertEquals(frame.checkpoint() != null, index % 32 == 0);
      Assert.assertEquals(frame.delta() != null, index % 32 != 0);
    }
    for (ReplayStep step : restored.steps())
      Assert.assertEquals(
          StateRestoration.at(restored.states(), step.stateId()), expected.get(step.stateId()));
    for (int index : new int[] {32, 31, 64, 1, 96, 0, 33}) {
      int id = restored.states().get(index).stateId();
      Assert.assertEquals(
          StateRestoration.at(restored.states(), id),
          expected.get(id),
          "Random access changed the state");
    }
  }

  @Test
  public void deltasPreserveRedDrawClearingRiichiCalledRiverKanUpgradeAndScoreChanges() {
    GameState state = new GameState();
    state.startRoundForReconstruction(0, 0, 0, 0);
    state.initializeWallForReconstruction(new int[0], 52);
    state.revealDoraIndicator(Tile.P5, true);
    state.hand(0).addPhysicalTile(0);
    state.hand(0).addPhysicalTile(32);
    for (int seat = 0; seat < 4; seat++) state.setScore(seat, 25000);
    var sequence = new ReviewStateSequence();
    List<TableState> expected = new ArrayList<>();
    capture(sequence, expected, state);
    state.hand(0).addPhysicalTile(Tile.AKA_M5_ID);
    state.recordDraw(0, Tile.AKA_M5_ID, TurnEvent.DrawSource.WALL, false);
    state.consumeReconstructionLiveWallDraw();
    capture(sequence, expected, state);
    Assert.assertNull(sequence.frames().getLast().delta().players().getFirst().hand());
    state.hand(0).removePhysicalTile(Tile.AKA_M5_ID);
    state.commitDiscard(0, Tile.M5, 1, true, true, true, 0);
    state.setRiichi(0, true);
    state.setScore(0, 24000);
    state.setKyotakuCount(1);
    capture(sequence, expected, state);
    Assert.assertNotNull(sequence.frames().getLast().delta().players().getFirst().draw());
    Assert.assertNull(sequence.frames().getLast().delta().players().getFirst().draw().tile());
    state.markLastDiscardCalled(0);
    Meld pon = Meld.pon(Tile.M5, Meld.RelativeSource.KAMICHA, Meld.AkaSource.CALLED_TILE);
    state.addMeld(1, pon, 0);
    state.setCurrentPlayer(1);
    capture(sequence, expected, state);
    Assert.assertEquals(
        sequence.frames().getLast().delta().players().getFirst().river().removeCount(), 1);
    state.replacePonWithKakan(1, pon, Meld.kakan(pon), 0);
    state.revealDoraIndicator(Tile.P6);
    capture(sequence, expected, state);
    var kan = StateRestoration.at(sequence.frames(), 4).players().get(1).melds().getFirst();
    Assert.assertEquals(kan.tiles(), List.of("5mr", "5m", "5m", "5m"));
    Assert.assertEquals(kan.calledIndex(), 0);
    Assert.assertEquals(kan.addedIndex(), 3);
    state.setScore(0, 16000);
    state.setScore(1, 33000);
    state.setKyotakuCount(0);
    capture(sequence, expected, state);
    state.startRoundForReconstruction(1, 1, 1, 0);
    state.initializeWallForReconstruction(new int[0], 52);
    state.revealDoraIndicator(Tile.TON);
    capture(sequence, expected, state);
    var restored =
        JSON.fromJson(
            JSON.toJson(new ReplayRound("round-0", 0, 0, 0, sequence.frames(), List.of())),
            ReplayRound.class);
    for (int i = 0; i < expected.size(); i++)
      Assert.assertEquals(StateRestoration.at(restored.states(), i), expected.get(i));
    Assert.assertEquals(expected.getFirst().players().get(0).hand(), List.of("1m", "9m"));
    Assert.assertFalse(expected.get(2).players().get(0).river().getFirst().called());
  }

  @Test
  public void handSplicesRetainTheirUnchangedPrefixAndSuffixAndSnapshotsOwnTheirValues() {
    GameState state = new GameState();
    state.hand(0).addPhysicalTile(0);
    state.hand(0).addPhysicalTile(16);
    state.hand(0).addPhysicalTile(32);
    var sequence = new ReviewStateSequence();
    sequence.captureState(10, state);
    state.hand(0).removePhysicalTile(16);
    sequence.captureState(11, state);
    var removed = sequence.frames().getLast().delta().players().getFirst().hand();
    Assert.assertEquals(removed, new ArrayPatch<>(1, 1, List.of()));
    state.hand(0).addPhysicalTile(17);
    sequence.captureState(12, state);
    var added = sequence.frames().getLast().delta().players().getFirst().hand();
    Assert.assertEquals(added, new ArrayPatch<>(1, 0, List.of("5m")));
    sequence.captureState(12, state);
    Assert.assertEquals(
        sequence.frames().size(), 3, "Repeated judgments must reuse the same state");
    state.hand(0).clear();
    Assert.assertEquals(
        StateRestoration.at(sequence.frames(), 10).players().get(0).hand(),
        List.of("1m", "5mr", "9m"));
    Assert.assertEquals(
        StateRestoration.at(sequence.frames(), 12).players().get(0).hand(),
        List.of("1m", "5m", "9m"));
  }

  private static void capture(
      ReviewStateSequence sequence, List<TableState> expected, GameState state) {
    sequence.captureState(expected.size(), state);
    expected.add(ReviewSnapshots.table(state));
  }
}
