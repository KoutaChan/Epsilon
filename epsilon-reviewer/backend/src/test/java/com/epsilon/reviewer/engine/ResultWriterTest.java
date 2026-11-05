package com.epsilon.reviewer.engine;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.replay.ReplayDecision;
import com.epsilon.replay.ReplayEngine;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.Hora;
import com.epsilon.replay.ReplayEvent.None;
import com.epsilon.replay.ReplayEvent.StartGame;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import com.epsilon.replay.ReplayObserver;
import com.epsilon.reviewer.engine.format.ResultWriter;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.load.RecordLoader;
import com.epsilon.reviewer.engine.load.RecordSummary;
import com.epsilon.reviewer.engine.review.EvaluatedDecision;
import com.epsilon.reviewer.engine.view.EventKind;
import com.epsilon.reviewer.engine.view.TableProjector;
import com.epsilon.reviewer.engine.view.TableState;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.zip.Deflater;
import java.util.zip.GZIPOutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * ResultWriter が書いた文書を、書き手の部品を使わない試験の読み手（Format6）で戻して確かめる。全ての手順の局面が TableProjector の局面と一致し、出来事の
 * 位置・判断の cause と resolve が再生と一致し、候補の行・chosen・probabilities・null の欄が決まりどおり。double-ron-a の文書を
 * target/format6/double-ron-a.json に書き残す（組み上げで画面の読み手に通す）。
 */
public class ResultWriterTest {
  private static final Logger LOG = LoggerFactory.getLogger(ResultWriterTest.class);
  private static final Set<String> CALLS = Set.of("CHI", "PON", "DAIMINKAN", "ANKAN", "KAKAN");
  private static final Set<EventKind> ACTORLESS =
      EnumSet.of(
          EventKind.START_KYOKU,
          EventKind.DORA,
          EventKind.RYUKYOKU,
          EventKind.END_KYOKU,
          EventKind.END_GAME);

  @Test
  public void everyStepRestoresTheProjectedTableAndTheReplayPositions() throws IOException {
    Map<String, LoadedRecord> records = new LinkedHashMap<>();
    for (String name : GoldenRecords.NAMES) records.put(name, GoldenRecords.read(name));
    records.put(
        "river.mjai",
        RecordLoader.load(AnalysisTest.riverFixture().getBytes(StandardCharsets.UTF_8)));
    records.put("chankan round", ScenarioReplayTest.chankanRound());
    for (Map.Entry<String, LoadedRecord> entry : records.entrySet()) {
      String name = entry.getKey();
      List<Expected> expected = expected(entry.getValue());
      String text = Format6.text(entry.getValue(), new RecordingPolicy());
      Format6.Document document = Format6.decode(JsonParser.parseString(text).getAsJsonObject());
      List<Format6.Step> steps = new ArrayList<>();
      for (Format6.Round round : document.rounds()) steps.addAll(round.steps());
      Assert.assertEquals(steps.size(), expected.size(), name);
      for (int i = 0; i < steps.size(); i++) {
        Format6.Step step = steps.get(i);
        Expected want = expected.get(i);
        String where = name + " step " + i;
        Assert.assertEquals(step.decision(), want.decision(), where);
        Assert.assertEquals(step.eventIndex(), want.eventIndex(), where);
        Assert.assertEquals(step.actor(), want.actor(), where);
        Assert.assertEquals(step.table(), want.table(), where);
        if (step.decision()) Assert.assertEquals(step.cause(), want.cause(), where);
        else Assert.assertEquals(step.event(), want.event(), where);
      }
      assertOmissions(name, document);
      byte[] raw = text.getBytes(StandardCharsets.UTF_8);
      LOG.info(
          "Format 6 for {}: {} bytes raw, {} bytes gzip 9, {} decisions.",
          name,
          raw.length,
          gzip(raw).length,
          document.decisions().size());
      if (name.equals("double-ron-a")) {
        Path target = Files.createDirectories(Path.of("target", "format6"));
        Files.write(target.resolve("double-ron-a.json"), raw);
      }
    }
  }

  @Test
  public void eventStepsMustBeContiguousAndEvaluationsMustNotMix() throws IOException {
    LoadedRecord record = GoldenRecords.read("kakan-a");
    TableState table = expected(record).getFirst().table();
    try (var writer = writer()) {
      writer.startRound(0, 0, 0, 1, table);
      writer.event(1, EventKind.START_KYOKU, -1, null, null);
      Assert.expectThrows(
          IllegalStateException.class, () -> writer.event(3, EventKind.TSUMO, 0, null, null));
      writer.event(2, EventKind.TSUMO, 0, null, null);
      writer.decision(0, 2, 3, decision(true), null);
      Assert.expectThrows(
          IllegalStateException.class, () -> writer.decision(0, 2, 3, decision(false), null));
      Assert.expectThrows(IllegalStateException.class, () -> writer.finish(null));
    }
    try (var writer = writer()) {
      Assert.expectThrows(
          IllegalStateException.class, () -> writer.event(1, EventKind.TSUMO, 0, null, null));
      writer.startRound(0, 0, 0, 1, table);
      Assert.expectThrows(
          IllegalStateException.class, () -> writer.decision(0, 1, 2, decision(true), null));
    }
  }

  /** 席 1・2 が席 0 の白でロンでき、席 2 のロンの後に席 1 が見送りを明示する（mjai の none）。 */
  @Test
  public void aPassAfterARonIsWrittenAfterTheRonInRecordOrder() {
    int[][] hands = {
      {12, 13, 14, 16, 20, 48, 52, 56, 84, 88, 92, 112, 116},
      {0, 4, 8, 36, 40, 44, 72, 76, 80, 108, 109, 110, 124},
      {24, 28, 32, 60, 64, 68, 96, 100, 104, 132, 133, 134, 125},
      {5, 6, 9, 17, 21, 29, 41, 49, 57, 77, 85, 113, 120}
    };
    List<ReplayEvent> events =
        List.of(
            new StartGame(),
            new StartKyoku(
                Tile.TON, 1, 0, 0, 0, Tile.HATSU, hands, new int[] {25000, 25000, 25000, 25000}),
            new Tsumo(0, 126),
            new Dahai(0, 126, true),
            new Hora(2, 0, Tile.HAKU, new int[] {-3000, 0, 3000, 0}),
            new None(1));
    Format6.Document document =
        Format6.analyze(ScenarioReplayTest.synthetic(events), new RecordingPolicy());
    List<String> written = new ArrayList<>();
    for (Format6.Step step : document.rounds().getFirst().steps())
      if (!step.decision()) written.add(step.event());
    Assert.assertEquals(written, List.of("start_kyoku", "tsumo", "dahai", "hora", "none"));
  }

  @Test
  public void policyOnlyDocumentsHaveNoTemperature() throws IOException {
    LoadedRecord record = GoldenRecords.read("kakan-a");
    var out = new ByteArrayOutputStream();
    try (var writer =
        new ResultWriter(out, "id", "2026-11-04T00:00:00Z", Format6.MODEL, record.summary())) {
      writer.startRound(0, 0, 0, 1, expected(record).getFirst().table());
      writer.event(1, EventKind.START_KYOKU, -1, null, null);
      writer.decision(0, 1, 2, decision(false), null);
      writer.finish(null);
    }
    JsonObject json =
        JsonParser.parseString(out.toString(StandardCharsets.UTF_8)).getAsJsonObject();
    Assert.assertFalse(json.has("temperature"));
    JsonObject step =
        json.getAsJsonArray("rounds").get(0).getAsJsonObject().getAsJsonArray("steps").get(1)
            .getAsJsonObject();
    Assert.assertEquals(
        step.toString(), "{\"seat\":0,\"candidates\":[[5,0.75],[371,0.25]],\"chosen\":1}");
  }

  /** 候補の行・chosen・probabilities・actor・null の書き分け。 */
  private static void assertOmissions(String name, Format6.Document document) {
    JsonObject json = document.json();
    Assert.assertEquals(json.get("format").getAsInt(), 6, name);
    Assert.assertEquals(json.get("temperature").getAsDouble(), 0.002, name);
    assertNullsOnlyForDraws(json, null);
    for (Format6.Round round : document.rounds())
      for (Format6.Step step : round.steps()) {
        JsonObject object = step.json();
        if (!step.decision()) {
          EventKind kind = EventKind.valueOf(step.event().toUpperCase(Locale.ROOT));
          Assert.assertEquals(object.has("actor"), !ACTORLESS.contains(kind), name + " " + object);
          Assert.assertEquals(
              object.has("outcome"),
              kind == EventKind.HORA || kind == EventKind.RYUKYOKU,
              name + " " + object);
          continue;
        }
        Assert.assertFalse(object.has("probabilities"), "RecordingPolicy probabilities derive");
        if (object.has("chosen"))
          Assert.assertTrue(step.chosen() < step.candidates().size(), name + " " + object);
        for (JsonElement row : step.candidates()) {
          JsonArray candidate = row.getAsJsonArray();
          String type = Action.fromIndex(candidate.get(0).getAsInt()).type().name();
          Assert.assertEquals(candidate.size(), CALLS.contains(type) ? 3 : 2, name + " " + row);
        }
      }
  }

  /** null は局面のツモ牌（draw）だけ。 */
  private static void assertNullsOnlyForDraws(JsonElement element, String key) {
    if (element.isJsonNull()) Assert.assertEquals(key, "draw");
    else if (element.isJsonArray())
      for (JsonElement item : element.getAsJsonArray()) assertNullsOnlyForDraws(item, key);
    else if (element.isJsonObject())
      for (Map.Entry<String, JsonElement> entry : element.getAsJsonObject().entrySet())
        assertNullsOnlyForDraws(entry.getValue(), entry.getKey());
  }

  /** 再生の通知の順の手順（start_game と最初の局より前は除く）と、その時の表示の局面。 */
  private static List<Expected> expected(LoadedRecord record) {
    List<Expected> steps = new ArrayList<>();
    ReplayEngine.replay(
        record.replay(),
        new ReplayObserver() {
          private boolean started;

          @Override
          public void onDecision(ReplayDecision decision, GameState state, List<Action> legal) {
            steps.add(
                new Expected(
                    true,
                    null,
                    decision.seat(),
                    decision.resolveIndex(),
                    decision.causeIndex(),
                    TableProjector.table(state)));
          }

          @Override
          public void onEventApplied(int index, int revision, ReplayEvent event, GameState state) {
            if (event instanceof StartGame) return;
            started |= event instanceof StartKyoku;
            if (!started) return;
            steps.add(
                new Expected(
                    false,
                    EventKind.of(event).wireName(),
                    EventKind.actor(event),
                    index,
                    -1,
                    TableProjector.table(state)));
          }
        });
    return steps;
  }

  private static ResultWriter writer() throws IOException {
    return new ResultWriter(
        OutputStream.nullOutputStream(),
        "id",
        "2026-11-04T00:00:00Z",
        Format6.MODEL,
        new RecordSummary("mjai", List.of("A", "B", "C", "D"), null, 1));
  }

  /** 打牌 1m と九種九牌の 2 候補の判断。 */
  private static EvaluatedDecision decision(boolean valued) {
    return new EvaluatedDecision(
        new int[] {5, 371},
        valued ? new float[] {0f, 0.001f} : new float[] {0.75f, 0.25f},
        new String[2][],
        1,
        null,
        valued);
  }

  private static byte[] gzip(byte[] raw) throws IOException {
    var out = new ByteArrayOutputStream();
    try (var gzip =
        new GZIPOutputStream(out) {
          {
            def.setLevel(Deflater.BEST_COMPRESSION);
          }
        }) {
      gzip.write(raw);
    }
    return out.toByteArray();
  }

  /** 再生が知らせた 1 手順。cause は判断だけ。 */
  private record Expected(
      boolean decision, String event, int actor, int eventIndex, int cause, TableState table) {}
}
