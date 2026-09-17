package com.epsilon.reviewer.engine;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.core.*;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.replay.*;
import com.epsilon.replay.ReplayEvent.*;
import com.epsilon.reviewer.dto.*;
import com.epsilon.reviewer.model.ModelDefinition;
import com.google.gson.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.CancellationException;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 各牌譜形式の再生、終局点の根拠、赤牌と副露の表示、保存したモデルを使った解析を検証する。 */
public class ReviewEngineTest {
  @Test
  public void prepareResolvesDuplicateMjaiTilesAndPreservesNamesAndFinalScoreProvenance() {
    var record =
        new ReviewEngine().prepareRecord(mjai(true).getBytes(StandardCharsets.UTF_8), "game.mjai");
    // Escaped Japanese names exercise UTF-8 decoding.
    Assert.assertEquals(
        record.metadata().names(),
        List.of(
            "\u307f\u306a\u3068",
            "\u3053\u306f\u304f",
            "\u3042\u304a\u3044",
            "\u3064\u3070\u3081"));
    Assert.assertEquals(record.metadata().finalScores(), List.of(32600, 28400, 21500, 17500));
    Assert.assertEquals(record.metadata().finalScoresSource(), "mjai.end_game");
    var start =
        record.replay().events().stream()
            .filter(StartKyoku.class::isInstance)
            .map(StartKyoku.class::cast)
            .findFirst()
            .orElseThrow();
    Set<Integer> ids = new HashSet<>();
    for (int[] hand : start.initialHandPhysicalTileIds())
      for (int id : hand)
        Assert.assertTrue(ids.add(id), "Duplicate physical tile in the initial hands");
    Assert.assertEquals(ids.size(), 52);
  }

  @Test
  public void prepareDoesNotInventTerminalPointsForAFragmentAndRejectsHiddenHands() {
    var engine = new ReviewEngine();
    var partial = engine.prepareRecord(mjai(false).getBytes(StandardCharsets.UTF_8), "part.jsonl");
    Assert.assertNull(partial.metadata().finalScores());
    Assert.assertEquals(partial.metadata().finalScoresSource(), "unavailable");
    String unknown = mjai(false).replaceFirst("\\\"7m\\\"", "\\\"?\\\"");
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> engine.prepareRecord(unknown.getBytes(StandardCharsets.UTF_8), "hidden.jsonl"));
  }

  @Test
  public void tenhouKeepsRecordedNamesAndRequiresOwariForFinalScores() {
    String init = xmlInit();
    var engine = new ReviewEngine();
    String prefix =
        "<mjloggm><GO type=\"169\"/><UN n0=\"%E3%81%BF%E3%81%AA%E3%81%A8\" n1=\"B\" n2=\"C\""
            + " n3=\"D\"/>"
            + init;
    var fragment =
        engine.prepareRecord((prefix + "</mjloggm>").getBytes(StandardCharsets.UTF_8), "part.xml");
    Assert.assertNull(fragment.metadata().finalScores());
    var complete =
        engine.prepareRecord(
            (prefix
                    + "<RYUUKYOKU sc=\"250,0,250,0,250,0,250,0\""
                    + " owari=\"326,52.6,284,8.4,215,-18.5,175,-42.5\"/></mjloggm>")
                .getBytes(StandardCharsets.UTF_8),
            "full.xml");
    Assert.assertEquals(complete.metadata().names().getFirst(), "\u307f\u306a\u3068");
    Assert.assertEquals(complete.metadata().finalScores(), List.of(32600, 28400, 21500, 17500));
    Assert.assertEquals(complete.metadata().finalScoresSource(), "tenhou.owari");
  }

  @Test
  public void snapshotsKeepDrawSeparateAndDisplayCalledRiverAndMeldSource() {
    List<ReplayEvent> events =
        List.of(
            new StartGame(),
            new StartKyoku(
                Tile.TON, 1, 0, 0, 1, Tile.HAKU, hands(), new int[] {25000, 25000, 25000, 25000}),
            new Tsumo(1, 25),
            new Dahai(1, 25, true),
            new Pon(0, 1, 25, new int[] {24, 26}),
            new Dora(Tile.P6));
    List<TableState> frames = new ArrayList<>();
    List<String> choices = new ArrayList<>();
    ReplayEngine.replay(
        replayRecord(events),
        new ReplayObserver() {
          public void onDecision(
              DecisionPoint point, GameState state, int seat, List<Action> legal) {
            int chosen = point.chosenSlot();
            choices.add(seat + ":" + (chosen < 0 ? "UNOBSERVED" : legal.get(chosen).type()));
          }

          public void onEventApplied(int index, int stateId, ReplayEvent event, GameState state) {
            if (!(event instanceof StartGame)) frames.add(ReviewSnapshots.table(state));
          }
        });
    Assert.assertEquals(choices, List.of("1:DAHAI", "2:UNOBSERVED", "0:PON"));
    Assert.assertEquals(frames.get(1).players().get(1).draw(), "7m");
    Assert.assertEquals(frames.get(1).players().get(1).hand().size(), 13);
    var called = frames.get(3);
    Assert.assertTrue(called.players().get(1).river().getFirst().called());
    Assert.assertEquals(called.players().get(0).melds().getFirst().from(), 1);
    Assert.assertEquals(
        called.players().get(0).melds().getFirst().tiles(), List.of("7m", "7m", "7m"));
    Assert.assertEquals(frames.getLast().dora(), List.of("P", "6p"));
    // Later events must not mutate earlier snapshots.
    Assert.assertFalse(frames.get(2).players().get(1).river().getFirst().called());
  }

  @Test
  public void redDoraRemainsRedThroughMjaiParsingReplayAndRendering() {
    JsonArray events = JsonParser.parseString(mjai(false)).getAsJsonArray();
    events.get(1).getAsJsonObject().addProperty("dora_marker", "5pr");
    events.add(JsonParser.parseString("{\"type\":\"dora\",\"dora_marker\":\"5p\"}"));
    var record =
        new ReviewEngine()
            .prepareRecord(events.toString().getBytes(StandardCharsets.UTF_8), "red.mjai");
    List<List<String>> indicators = new ArrayList<>();
    ReplayEngine.replay(
        record.replay(),
        new ReplayObserver() {
          public void onDecision(
              DecisionPoint point, GameState state, int seat, List<Action> legal) {}

          public void onEventApplied(int index, int stateId, ReplayEvent event, GameState state) {
            if (event instanceof StartGame) return;
            indicators.add(ReviewSnapshots.table(state).dora());
            Assert.assertTrue(
                AkaTileMask.containsTile(state.doraState().visibleIndicatorAkaMask(), Tile.P5));
          }
        });
    Assert.assertEquals(indicators.getFirst(), List.of("5pr"));
    Assert.assertEquals(indicators.getLast(), List.of("5pr", "5p"));
  }

  @Test
  public void decodedMahjongSoulArrayUsesTheSameValidatedReplayAndKeepsItsSource() {
    JsonObject start =
        JsonParser.parseString(
                "{\"chang\":0,\"ju\":0,\"ben\":0,\"liqibang\":0,\"dora\":\"7z\",\"scores\":[25000,25000,25000,25000]}")
            .getAsJsonObject();
    for (int seat = 0; seat < 4; seat++) {
      JsonArray tiles = new JsonArray();
      for (int id : hands()[seat]) {
        int type = Tile.typeOf(id);
        tiles.add(
            type >= 27
                ? (type - 26) + "z"
                : (Tile.isAka(id)
                    ? "0" + "mps".charAt(type / 9)
                    : ReviewSnapshots.tile(type, false)));
      }
      start.add("tiles" + seat, tiles);
    }
    start.add(
        "tiles0",
        new Gson()
            .toJsonTree(
                List.of(
                    "1m", "9m", "1p", "9p", "1s", "9s", "1z", "2z", "3z", "2m", "3m", "4m", "5m",
                    "7z")));
    JsonObject first = new JsonObject();
    first.addProperty("name", ".lq.RecordNewRound");
    first.add("data", start);
    JsonObject last =
        JsonParser.parseString(
                "{\"name\":\".lq.RecordLiuJu\",\"data\":{\"type\":1,\"gameend\":{\"scores\":[25000,25000,25000,25000]}}}")
            .getAsJsonObject();
    JsonArray records = new JsonArray();
    records.add(first);
    records.add(last);
    var prepared =
        new ReviewEngine()
            .prepareRecord(records.toString().getBytes(StandardCharsets.UTF_8), "majsoul.json");
    Assert.assertEquals(prepared.metadata().source(), "majsoul");
    Assert.assertEquals(prepared.metadata().finalScores(), List.of(25000, 25000, 25000, 25000));
    Assert.assertEquals(prepared.metadata().finalScoresSource(), "majsoul.gameend.scores");
    Assert.assertEquals(prepared.metadata().roundCount(), 1);
  }

  @Test(groups = "native")
  public void restoredCheckpointReviewsEverySeatWithTheActualNormalizedPolicy() throws Exception {
    Path directory = Files.createTempDirectory("reviewer-native-");
    Path checkpoint = directory.resolve("checkpoint");
    var engine = new ReviewEngine();
    var record = engine.prepareRecord(mjai(true).getBytes(StandardCharsets.UTF_8), "game.mjai");
    try (Model source =
        NetworkFactory.createDecisionModel(Device.cpu(), false, 32, EpsilonUtilityProfile.TOP)) {
      EpsilonDecisionCheckpointManager.saveInitial(source, checkpoint);
      var model =
          new ModelDefinition(
              "fixture",
              "fixed-revision",
              "Fixture",
              "1",
              "epsilon-nano",
              checkpoint,
              Map.of("device", "cpu"));
      var result = engine.analyze(record, model, () -> false, ignored -> {});
      Gson json = new GsonBuilder().serializeNulls().create();
      Files.writeString(Path.of("target/reviewer-fixture.json"), json.toJson(result));
      String fixtureDirectory = System.getProperty("epsilon.reviewer.fixtureDir");
      if (fixtureDirectory != null) {
        Path fixture = Path.of(fixtureDirectory);
        Files.createDirectories(fixture);
        EpsilonDecisionCheckpointManager.saveInitial(source, fixture.resolve("model"));
        Files.writeString(fixture.resolve("record.mjai"), mjai(true));
        Files.writeString(fixture.resolve("result.json"), json.toJson(result));
        String riverRecord = riverFixture();
        var riverPrepared =
            engine.prepareRecord(riverRecord.getBytes(StandardCharsets.UTF_8), "river.mjai");
        var riverResult = engine.analyze(riverPrepared, model, () -> false, ignored -> {});
        for (var player :
            StateRestoration.at(
                    riverResult.rounds().getLast().states(),
                    riverResult.rounds().getLast().steps().getLast().stateId())
                .players()) Assert.assertEquals(player.river().size(), 14);
        Files.writeString(fixture.resolve("river.mjai"), riverRecord);
        Files.writeString(fixture.resolve("river-result.json"), json.toJson(riverResult));
      }
      Set<Integer> seats = new HashSet<>();
      var decisions =
          result.rounds().stream()
              .flatMap(round -> round.steps().stream())
              .filter(ReplayStep::decision)
              .toList();
      for (var step : decisions) {
        seats.add(step.actor());
        Assert.assertEquals(step.candidates().stream().filter(ActionCandidate::chosen).count(), 1L);
        Assert.assertEquals(
            step.candidates().stream().mapToDouble(ActionCandidate::probability).sum(),
            1.0,
            0.0001);
      }
      Assert.assertEquals(seats, Set.of(0, 1, 2, 3));
      List<float[]> direct = new ArrayList<>();
      try (var server = EpsilonDecisionInferenceServer.forFrozenModel(source, 1)) {
        ReplayEngine.replay(
            record.replay(),
            (point, state, seat, legal) -> {
              var builder =
                  DecisionBatchBuilder.inference(
                      1, DecisionBatchBuilder.selectInferenceBucket(legal));
              builder.encodeObservedInferenceRow(0, state.publicObservation(seat), legal);
              float[] probabilities =
                  server
                      .evaluateBatch(builder.buildEncodedInferenceRows())
                      .getFirst()
                      .policyProbabilities();
              float[] byAction = new float[Action.ACTION_SPACE_SIZE];
              for (int i = 0; i < legal.size(); i++)
                byAction[legal.get(i).toIndex()] = probabilities[i];
              direct.add(byAction);
            });
      }
      Assert.assertEquals(decisions.size(), direct.size());
      for (int i = 0; i < decisions.size(); i++)
        for (var candidate : decisions.get(i).candidates())
          Assert.assertEquals(
              candidate.probability(), direct.get(i)[candidate.actionId()], 0.00001f);
      Assert.assertEquals(result.model().revision(), "fixed-revision");
      Assert.expectThrows(
          CancellationException.class,
          () -> engine.analyze(record, model, () -> true, ignored -> {}));
    } finally {
      try (var paths = Files.walk(directory)) {
        for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }

  @Test
  public void completedGameCannotBeFollowedByAnotherRoundOrAnotherGame() {
    var engine = new ReviewEngine();
    JsonArray games = JsonParser.parseString(mjai(true)).getAsJsonArray();
    games.addAll(JsonParser.parseString(mjai(false)).getAsJsonArray());
    String combined = games.toString();
    try {
      engine.prepareRecord(combined.getBytes(StandardCharsets.UTF_8), "combined.mjai");
      org.testng.Assert.fail("Accepted another game after the completed game");
    } catch (IllegalArgumentException expected) {
      org.testng.Assert.assertTrue(expected.getMessage().contains("end_game"));
    }
  }

  private static ReplayRecord replayRecord(List<ReplayEvent> events) {
    return new ReplayRecord(
        events,
        new com.epsilon.replay.RecordMetadata(
            RecordFormat.MJAI, List.of("A", "B", "C", "D"), null, "unavailable", null),
        null,
        RecordCompletion.INCOMPLETE);
  }

  private static String mjai(boolean complete) {
    JsonArray events = new JsonArray();
    var header =
        JsonParser.parseString(
                "{\"type\":\"start_game\",\"names\":[\"\u307f\u306a\u3068\",\"\u3053\u306f\u304f\",\"\u3042\u304a\u3044\",\"\u3064\u3070\u3081\"]}")
            .getAsJsonObject();
    events.add(header);
    JsonObject start =
        JsonParser.parseString(
                "{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"kyoku\":1,\"honba\":0,\"kyotaku\":0,\"oya\":0,\"dora_marker\":\"C\",\"scores\":[25000,25000,25000,25000]}")
            .getAsJsonObject();
    JsonArray tehais = new JsonArray();
    for (int[] hand : hands()) {
      JsonArray tiles = new JsonArray();
      for (int id : hand) tiles.add(ReviewSnapshots.tile(Tile.typeOf(id), Tile.isAka(id)));
      tehais.add(tiles);
    }
    start.add("tehais", tehais);
    events.add(start);
    for (int seat = 0; seat < 4; seat++) {
      events.add(
          JsonParser.parseString("{\"type\":\"tsumo\",\"actor\":" + seat + ",\"pai\":\"F\"}"));
      events.add(
          JsonParser.parseString(
              "{\"type\":\"dahai\",\"actor\":" + seat + ",\"pai\":\"F\",\"tsumogiri\":true}"));
    }
    if (complete) {
      events.add(
          JsonParser.parseString("{\"type\":\"ryukyoku\",\"deltas\":[7600,3400,-3500,-7500]}"));
      events.add(JsonParser.parseString("{\"type\":\"end_kyoku\"}"));
      events.add(
          JsonParser.parseString("{\"type\":\"end_game\",\"scores\":[32600,28400,21500,17500]}"));
    }
    return events.toString();
  }

  static String riverFixture() {
    JsonArray original = JsonParser.parseString(mjai(false)).getAsJsonArray();
    JsonArray events = new JsonArray();
    events.add(original.get(0));
    events.add(original.get(1));
    Set<Integer> used = new HashSet<>();
    used.add(132);
    for (int[] hand : hands()) for (int id : hand) used.add(id);
    List<Integer> wall = new ArrayList<>();
    for (int id = 0; id < 136; id++) if (!used.contains(id)) wall.add(id);
    Collections.shuffle(wall, new Random(419));
    for (int i = 0; i < 56; i++) {
      int id = wall.get(i), seat = i % 4;
      String tile = ReviewSnapshots.tile(Tile.typeOf(id), Tile.isAka(id));
      JsonObject draw = new JsonObject();
      draw.addProperty("type", "tsumo");
      draw.addProperty("actor", seat);
      draw.addProperty("pai", tile);
      events.add(draw);
      JsonObject discard = new JsonObject();
      discard.addProperty("type", "dahai");
      discard.addProperty("actor", seat);
      discard.addProperty("pai", tile);
      discard.addProperty("tsumogiri", true);
      events.add(discard);
    }
    return events.toString();
  }

  private static String xmlInit() {
    StringBuilder xml =
        new StringBuilder("<INIT seed=\"0,0,0,0,0,132\" ten=\"250,250,250,250\" oya=\"0\"");
    int seat = 0;
    for (int[] hand : hands())
      xml.append(" hai")
          .append(seat++)
          .append("=\"")
          .append(String.join(",", Arrays.stream(hand).mapToObj(Integer::toString).toList()))
          .append("\"");
    return xml.append("/>").toString();
  }

  private static int[][] hands() {
    return new int[][] {
      {24, 26, 3, 5, 9, 13, 17, 21, 28, 32, 36, 56, 60},
      {0, 1, 2, 4, 8, 12, 40, 44, 48, 76, 80, 84, 124},
      {6, 10, 14, 18, 22, 29, 33, 37, 41, 45, 49, 53, 57},
      {7, 11, 15, 19, 23, 30, 34, 38, 42, 46, 50, 54, 58}
    };
  }
}
