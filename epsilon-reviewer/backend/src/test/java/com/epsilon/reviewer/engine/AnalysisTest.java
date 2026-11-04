package com.epsilon.reviewer.engine;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.core.Action;
import com.epsilon.core.AkaTileMask;
import com.epsilon.core.GameState;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.Tile;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.replay.DecisionPoint;
import com.epsilon.replay.ReplayEngine;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.Dora;
import com.epsilon.replay.ReplayEvent.Pon;
import com.epsilon.replay.ReplayEvent.StartGame;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import com.epsilon.replay.ReplayObserver;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.engine.format.ResultWriter;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.load.RecordLoader;
import com.epsilon.reviewer.engine.review.Analysis;
import com.epsilon.reviewer.engine.review.JobControl;
import com.epsilon.reviewer.engine.review.ReviewModel;
import com.epsilon.reviewer.engine.view.TableProjector;
import com.epsilon.reviewer.engine.view.TableState;
import com.epsilon.reviewer.engine.view.TileCode;
import com.epsilon.reviewer.model.PolicyProviders;
import com.epsilon.spi.ReviewEvaluation;
import com.epsilon.spi.ReviewPolicy;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.function.Function;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 各牌譜形式の読み、最終点の根拠、赤牌と副露の表示、候補の並べ替え、取り消しと進み、保存したモデルを使った解析を確かめる。 */
public class AnalysisTest {
  @Test
  public void loadResolvesDuplicateMjaiTilesAndKeepsNamesAndFinalScores() {
    LoadedRecord record = RecordLoader.load(mjai(true).getBytes(StandardCharsets.UTF_8));
    // 日本語の名前は UTF-8 の読みを通す。
    Assert.assertEquals(
        record.summary().names(),
        List.of(
            "みなと",
            "こはく",
            "あおい",
            "つばめ"));
    Assert.assertEquals(record.summary().finalScores(), List.of(32600, 28400, 21500, 17500));
    Assert.assertEquals(record.summary().source(), "mjai");
    Assert.assertEquals(record.summary().roundCount(), 1);
    var start = (StartKyoku) record.replay().events().get(record.roundStarts()[0]);
    Set<Integer> ids = new HashSet<>();
    for (int[] hand : start.initialHandPhysicalTileIds())
      for (int id : hand)
        Assert.assertTrue(ids.add(id), "Duplicate physical tile in the initial hands");
    Assert.assertEquals(ids.size(), 52);
  }

  @Test
  public void loadDoesNotInventTerminalPointsForAFragmentAndRejectsHiddenHands() {
    var partial = RecordLoader.load(mjai(false).getBytes(StandardCharsets.UTF_8));
    Assert.assertNull(partial.summary().finalScores());
    String unknown = mjai(false).replaceFirst("\\\"7m\\\"", "\\\"?\\\"");
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> RecordLoader.load(unknown.getBytes(StandardCharsets.UTF_8)));
  }

  @Test
  public void tenhouKeepsRecordedNamesAndRequiresOwariForFinalScores() {
    String prefix =
        "<mjloggm><GO type=\"169\"/><UN n0=\"%E3%81%BF%E3%81%AA%E3%81%A8\" n1=\"B\" n2=\"C\""
            + " n3=\"D\"/>"
            + xmlInit();
    var fragment = RecordLoader.load((prefix + "</mjloggm>").getBytes(StandardCharsets.UTF_8));
    Assert.assertNull(fragment.summary().finalScores());
    var complete =
        RecordLoader.load(
            (prefix
                    + "<RYUUKYOKU sc=\"250,0,250,0,250,0,250,0\""
                    + " owari=\"326,52.6,284,8.4,215,-18.5,175,-42.5\"/></mjloggm>")
                .getBytes(StandardCharsets.UTF_8));
    Assert.assertEquals(complete.summary().names().getFirst(), "みなと");
    Assert.assertEquals(complete.summary().finalScores(), List.of(32600, 28400, 21500, 17500));
    Assert.assertEquals(complete.summary().source(), "tenhou");
  }

  @Test
  public void tablesKeepTheDrawSeparateAndShowTheCalledRiverAndMeldSource() {
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
        ScenarioReplayTest.synthetic(events).replay(),
        new ReplayObserver() {
          @Override
          public void onDecision(
              DecisionPoint point, GameState state, int seat, List<Action> legal) {
            int chosen = point.chosenSlot();
            choices.add(seat + ":" + (chosen < 0 ? "UNOBSERVED" : legal.get(chosen).type()));
          }

          @Override
          public void onEventApplied(int index, int stateId, ReplayEvent event, GameState state) {
            if (!(event instanceof StartGame)) frames.add(TableProjector.table(state));
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
    // 後の出来事が前の局面を変えない。
    Assert.assertFalse(frames.get(2).players().get(1).river().getFirst().called());
  }

  @Test
  public void redDoraRemainsRedThroughMjaiParsingReplayAndProjection() {
    JsonArray events = JsonParser.parseString(mjai(false)).getAsJsonArray();
    events.get(1).getAsJsonObject().addProperty("dora_marker", "5pr");
    events.add(JsonParser.parseString("{\"type\":\"dora\",\"dora_marker\":\"5p\"}"));
    var record = RecordLoader.load(events.toString().getBytes(StandardCharsets.UTF_8));
    List<List<String>> indicators = new ArrayList<>();
    ReplayEngine.replay(
        record.replay(),
        new ReplayObserver() {
          @Override
          public void onDecision(
              DecisionPoint point, GameState state, int seat, List<Action> legal) {}

          @Override
          public void onEventApplied(int index, int stateId, ReplayEvent event, GameState state) {
            if (event instanceof StartGame) return;
            indicators.add(TableProjector.table(state).dora());
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
                "{\"chang\":0,\"ju\":0,\"ben\":0,\"liqibang\":0,\"dora\":\"7z\","
                    + "\"scores\":[25000,25000,25000,25000]}")
            .getAsJsonObject();
    for (int seat = 0; seat < 4; seat++) {
      JsonArray tiles = new JsonArray();
      for (int id : hands()[seat]) {
        int type = Tile.typeOf(id);
        tiles.add(
            type >= 27
                ? (type - 26) + "z"
                : (Tile.isAka(id) ? "0" + "mps".charAt(type / 9) : TileCode.of(type, false)));
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
                "{\"name\":\".lq.RecordLiuJu\",\"data\":{\"type\":1,"
                    + "\"gameend\":{\"scores\":[25000,25000,25000,25000]}}}")
            .getAsJsonObject();
    JsonArray records = new JsonArray();
    records.add(first);
    records.add(last);
    var loaded = RecordLoader.load(records.toString().getBytes(StandardCharsets.UTF_8));
    Assert.assertEquals(loaded.summary().source(), "majsoul");
    Assert.assertEquals(loaded.summary().finalScores(), List.of(25000, 25000, 25000, 25000));
    Assert.assertEquals(loaded.summary().roundCount(), 1);
  }

  @Test(groups = "native")
  public void restoredNanoCheckpointReviewsEverySeatWithTheActualNormalizedPolicy()
      throws Exception {
    Path directory = TestFiles.directory("reviewer-native-");
    Path checkpoint = directory.resolve("checkpoint");
    LoadedRecord record = RecordLoader.load(mjai(true).getBytes(StandardCharsets.UTF_8));
    try (Model source =
        NetworkFactory.createDecisionModel(Device.cpu(), false, 32, EpsilonUtilityProfile.TOP)) {
      EpsilonDecisionCheckpointManager.saveInitial(source, checkpoint);
      try (ReviewPolicy policy =
          PolicyProviders.load()
              .find("epsilon-nano")
              .orElseThrow()
              .openReview(checkpoint, Map.of("device", "cpu"))) {
        ReviewModel model = model(policy);
        Format6.Document document = Format6.analyze(record, model);
        Assert.assertFalse(document.json().has("temperature"), "Nano has no value head");
        Set<Integer> seats = new HashSet<>();
        List<Format6.Step> decisions = steps(document);
        for (Format6.Step step : decisions) {
          seats.add(step.actor());
          Assert.assertTrue(step.chosen() >= 0);
          double sum = 0;
          for (int i = 0; i < step.candidates().size(); i++) sum += value(step, i);
          Assert.assertEquals(sum, 1.0, 0.0001);
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
        for (int d = 0; d < decisions.size(); d++)
          for (int i = 0; i < decisions.get(d).candidates().size(); i++)
            Assert.assertEquals(
                value(decisions.get(d), i), direct.get(d)[decisions.get(d).actionId(i)], 0.00001f);
        Assert.expectThrows(
            CancellationException.class,
            () -> run(record, model, cancelledAfter(0, new ArrayList<>())));
      }
    } finally {
      TestFiles.delete(directory);
    }
  }

  @Test
  public void valuedPolicySortsByValueLossThenProbabilityAndKeepsTheChosenAction() {
    LoadedRecord record = record();
    List<Integer> chosen = new ArrayList<>();
    ReplayEngine.replay(
        record.replay(),
        (point, state, seat, legal) ->
            chosen.add(point.chosenSlot() < 0 ? -1 : legal.get(point.chosenSlot()).toIndex()));
    // 価値は actionId の 2 つずつで同じにし、同じ価値の中は確率（actionId が大きいほど高い）の順になることも確かめる。
    Format6.Document document =
        Format6.analyze(
            record,
            fake(
                legal -> {
                  float[] values = new float[legal.size()];
                  float[] logits = new float[legal.size()];
                  for (int i = 0; i < legal.size(); i++) {
                    values[i] = -0.001f * (legal.get(i).toIndex() / 2);
                    logits[i] = 0.001f * legal.get(i).toIndex();
                  }
                  return ReviewEvaluation.valued(values, ReviewEvaluation.softmax(logits, 0.02));
                }));
    Assert.assertEquals(document.json().get("temperature").getAsDouble(), 0.002);
    List<Format6.Step> decisions = steps(document);
    Assert.assertEquals(decisions.size(), chosen.size());
    Assert.assertTrue(decisions.stream().anyMatch(step -> step.candidates().size() > 2));
    for (int s = 0; s < decisions.size(); s++) {
      Format6.Step step = decisions.get(s);
      List<Integer> ids = new ArrayList<>();
      for (int i = 0; i < step.candidates().size(); i++) ids.add(step.actionId(i));
      int best = ids.stream().mapToInt(id -> id / 2).min().orElseThrow();
      List<Integer> expected =
          ids.stream()
              .sorted(
                  Comparator.<Integer>comparingInt(id -> id / 2)
                      .thenComparing(Comparator.reverseOrder()))
              .toList();
      Assert.assertEquals(ids, expected);
      Assert.assertEquals(value(step, 0), 0f);
      for (int i = 0; i < ids.size(); i++)
        Assert.assertEquals(value(step, i), 0.001f * (ids.get(i) / 2 - best), 1e-5f);
      Assert.assertEquals(ids.get(step.chosen()).intValue(), chosen.get(s).intValue());
      // 確率は価値から導けない（別の温度の softmax）ので、候補が 2 つ以上なら確率を書く。
      Assert.assertEquals(step.json().has("probabilities"), ids.size() > 1);
    }
  }

  @Test
  public void probabilityOnlyPolicyHasNoTemperatureAndSortsByProbability() {
    Format6.Document document =
        Format6.analyze(
            record(),
            fake(
                legal -> {
                  float[] logits = new float[legal.size()];
                  for (int i = 0; i < legal.size(); i++)
                    logits[i] = 0.001f * legal.get(i).toIndex();
                  return ReviewEvaluation.policy(ReviewEvaluation.softmax(logits, 0.02));
                }));
    Assert.assertFalse(document.json().has("temperature"));
    List<Format6.Step> decisions = steps(document);
    Assert.assertTrue(decisions.stream().anyMatch(step -> step.candidates().size() > 2));
    for (Format6.Step step : decisions) {
      for (int i = 1; i < step.candidates().size(); i++)
        Assert.assertTrue(value(step, i - 1) >= value(step, i));
      Assert.assertTrue(step.chosen() >= 0);
      Assert.assertFalse(step.json().has("probabilities"));
    }
  }

  @Test
  public void policyOutputForAnotherNumberOfActionsIsRejected() {
    Assert.expectThrows(
        IllegalStateException.class,
        () ->
            Format6.analyze(
                record(),
                fake(
                    legal -> {
                      float[] probabilities = new float[legal.size() + 1];
                      Arrays.fill(probabilities, 1f / probabilities.length);
                      return ReviewEvaluation.policy(probabilities);
                    })));
  }

  @Test
  public void cancellationStopsAtTheNextEventAndProgressStaysBelowTheEnd() throws Exception {
    LoadedRecord record = record();
    List<Integer> progress = new ArrayList<>();
    Assert.expectThrows(
        CancellationException.class,
        () -> run(record, new RecordingPolicy(), cancelledAfter(5, progress)));
    Assert.assertEquals(progress.size(), 5);
    List<Integer> complete = new ArrayList<>();
    run(record, new RecordingPolicy(), cancelledAfter(Integer.MAX_VALUE, complete));
    Assert.assertEquals(complete.getLast().intValue(), 99);
    for (int i = 1; i < complete.size(); i++)
      Assert.assertTrue(complete.get(i - 1) <= complete.get(i));
  }

  @Test
  public void completedGameCannotBeFollowedByAnotherRoundOrAnotherGame() {
    JsonArray games = JsonParser.parseString(mjai(true)).getAsJsonArray();
    games.addAll(JsonParser.parseString(mjai(false)).getAsJsonArray());
    IllegalArgumentException refused =
        Assert.expectThrows(
            IllegalArgumentException.class,
            () -> RecordLoader.load(games.toString().getBytes(StandardCharsets.UTF_8)));
    Assert.assertTrue(refused.getMessage().contains("end_game"), refused.getMessage());
  }

  private static void run(LoadedRecord record, ReviewModel model, JobControl control)
      throws Exception {
    try (var writer =
        new ResultWriter(
            OutputStream.nullOutputStream(),
            "00000000-0000-4000-8000-000000000000",
            "2026-11-04T00:00:00Z",
            Format6.MODEL,
            record.summary())) {
      Analysis.run(record, model, control, writer);
    }
  }

  /** 進みを count 回知らされた後に取り消す。 */
  private static JobControl cancelledAfter(int count, List<Integer> progress) {
    return new JobControl() {
      @Override
      public boolean cancelled() {
        return progress.size() >= count;
      }

      @Override
      public void progress(int percent) {
        progress.add(percent);
      }
    };
  }

  private static List<Format6.Step> steps(Format6.Document document) {
    return document.decisions().stream().map(Format6.Located::step).toList();
  }

  private static float value(Format6.Step step, int candidate) {
    return step.candidates().get(candidate).getAsJsonArray().get(1).getAsFloat();
  }

  private static LoadedRecord record() {
    return RecordLoader.load(mjai(true).getBytes(StandardCharsets.UTF_8));
  }

  /** 合法手だけから評価する偽のモデル。 */
  private static ReviewModel fake(Function<List<Action>, ReviewEvaluation> evaluator) {
    return new ReviewModel() {
      @Override
      public ReviewEvaluation evaluate(PublicObservation observation, List<Action> legal) {
        return evaluator.apply(legal);
      }

      @Override
      public double temperature() {
        return ReviewEvaluation.DISPLAY_TEMPERATURE;
      }
    };
  }

  private static ReviewModel model(ReviewPolicy policy) {
    return new ReviewModel() {
      @Override
      public ReviewEvaluation evaluate(PublicObservation observation, List<Action> legal) {
        return policy.evaluate(observation, legal);
      }

      @Override
      public double temperature() {
        return ReviewEvaluation.DISPLAY_TEMPERATURE;
      }
    };
  }

  static String mjai(boolean complete) {
    JsonArray events = new JsonArray();
    events.add(
        JsonParser.parseString(
            "{\"type\":\"start_game\",\"names\":[\"みなと\",\"こはく\",\"あおい\",\"つばめ\"]}"));
    JsonObject start =
        JsonParser.parseString(
                "{\"type\":\"start_kyoku\",\"bakaze\":\"E\",\"kyoku\":1,\"honba\":0,"
                    + "\"kyotaku\":0,\"oya\":0,\"dora_marker\":\"C\","
                    + "\"scores\":[25000,25000,25000,25000]}")
            .getAsJsonObject();
    JsonArray tehais = new JsonArray();
    for (int[] hand : hands()) {
      JsonArray tiles = new JsonArray();
      for (int id : hand) tiles.add(TileCode.of(Tile.typeOf(id), Tile.isAka(id)));
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

  /** 14 巡の河を持つ mjai の 1 局（全員がツモ切り）。 */
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
      int id = wall.get(i);
      int seat = i % 4;
      String tile = TileCode.of(Tile.typeOf(id), Tile.isAka(id));
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
