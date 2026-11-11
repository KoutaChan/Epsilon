package com.epsilon.reviewer.engine;

import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.ReplayDriver;
import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.FixtureFamily;
import com.epsilon.reviewer.engine.format.ResultWriter;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.load.RecordSummary;
import com.epsilon.reviewer.engine.review.EvaluatedDecision;
import com.epsilon.reviewer.engine.view.EventKind;
import com.epsilon.reviewer.engine.view.RecordTable;
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
 * ResultWriter が書いた文書を、ライターのコードを使わないテスト用のリーダー（Format6）で読み戻して確かめる。どの手順の局面も、
 * 記録の手でエンジンを進めながら、同時に表示の卓も進めたときの局面と一致する。イベントの位置と、判断の cause・resolve は、
 * 読み込みのときに作ったインデックスと一致する。候補の行・chosen・null のフィールドは、決まりどおりに書くか省く。
 * 最初の牌譜の文書は target/format6/&lt;牌譜の名前&gt;.json に残す。
 */
public class ResultWriterTest {
  private static final Logger LOG = LoggerFactory.getLogger(ResultWriterTest.class);
  private static final Set<Action.Type> CALLS =
      Set.of(
          Action.Type.CHI,
          Action.Type.PON,
          Action.Type.DAIMINKAN,
          Action.Type.ANKAN,
          Action.Type.KAKAN);

  @Test
  public void everyStepRestoresTheTableAndTheIndexedPositions() throws IOException {
    Map<String, LoadedRecord> records = new LinkedHashMap<>(PlayedRecords.fourPlayer());
    records.put("chankan round", TestRounds.chankanRound());
    String sample = records.keySet().iterator().next();
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
      if (name.equals(sample)) {
        Path target = Files.createDirectories(Path.of("target", "format6"));
        Files.write(target.resolve(name + ".json"), raw);
      }
    }
  }

  @Test
  public void eventStepsMustBeContiguousAndEvaluationsMustNotMix() throws IOException {
    LoadedRecord record = PlayedRecords.read(Variant.TENHOU_4P, 0);
    TableState table = expected(record).getFirst().table();
    try (var writer = writer()) {
      writer.startRound(0, 0, 0, 0, table);
      writer.event(0, EventKind.START_KYOKU, -1, null, null);
      Assert.expectThrows(
          IllegalStateException.class, () -> writer.event(2, EventKind.TSUMO, 0, null, null));
      writer.event(1, EventKind.TSUMO, 0, null, null);
      writer.decision(0, 1, 2, decision(true));
      Assert.expectThrows(
          IllegalStateException.class, () -> writer.decision(0, 1, 2, decision(false)));
      Assert.expectThrows(IllegalStateException.class, () -> writer.finish(null));
    }
    try (var writer = writer()) {
      Assert.expectThrows(
          IllegalStateException.class, () -> writer.event(0, EventKind.TSUMO, 0, null, null));
      writer.startRound(0, 0, 0, 0, table);
      Assert.expectThrows(
          IllegalStateException.class, () -> writer.decision(0, 0, 1, decision(true)));
    }
  }

  @Test
  public void policyOnlyDocumentsHaveNoTemperature() throws IOException {
    LoadedRecord record = PlayedRecords.read(Variant.TENHOU_4P, 0);
    var out = new ByteArrayOutputStream();
    try (var writer =
        new ResultWriter(out, "id", "2026-11-04T00:00:00Z", Format6.MODEL, record.summary())) {
      writer.startRound(0, 0, 0, 0, expected(record).getFirst().table());
      writer.event(0, EventKind.START_KYOKU, -1, null, null);
      writer.decision(0, 0, 1, decision(false));
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

  /** 候補の行・chosen・actor・null を、どの手順で書き、どの手順で省くかを確かめる。 */
  private static void assertOmissions(String name, Format6.Document document) {
    JsonObject json = document.json();
    Assert.assertEquals(json.get("format").getAsInt(), 6, name);
    Assert.assertEquals(
        json.get("temperature").getAsDouble(), FixtureFamily.TEMPERATURE, name);
    assertNullsOnlyForDraws(json, null);
    for (Format6.Round round : document.rounds())
      for (Format6.Step step : round.steps()) {
        JsonObject object = step.json();
        if (!step.decision()) {
          EventKind kind = EventKind.valueOf(step.event().toUpperCase(Locale.ROOT));
          Assert.assertEquals(object.has("actor"), kind.hasActor(), name + " " + object);
          Assert.assertEquals(
              object.has("outcome"),
              kind == EventKind.HORA || kind == EventKind.RYUKYOKU,
              name + " " + object);
          continue;
        }
        if (object.has("chosen"))
          Assert.assertTrue(step.chosen() < step.candidates().size(), name + " " + object);
        for (JsonElement row : step.candidates()) {
          JsonArray candidate = row.getAsJsonArray();
          Action.Type type = Action.of(candidate.get(0).getAsInt()).type();
          Assert.assertEquals(candidate.size(), CALLS.contains(type) ? 3 : 2, name + " " + row);
        }
      }
  }

  /** null が現れるのは局面のツモ牌（draw）だけであることを確かめる。 */
  private static void assertNullsOnlyForDraws(JsonElement element, String key) {
    if (element.isJsonNull()) Assert.assertEquals(key, "draw");
    else if (element.isJsonArray())
      for (JsonElement item : element.getAsJsonArray()) assertNullsOnlyForDraws(item, key);
    else if (element.isJsonObject())
      for (Map.Entry<String, JsonElement> entry : element.getAsJsonObject().entrySet())
        assertNullsOnlyForDraws(entry.getValue(), entry.getKey());
  }

  /**
   * 期待する手順と局面を作る。記録の手でエンジンを進めて判断の時点を求め、同時に記録のイベントを表示の卓に当てて局面を作る。
   * 手順の位置は 0 から数え、読み込みのときのインデックス（局の最初の手順、イベントの最初の手順、手順の数）と照合する。
   */
  private static List<Expected> expected(LoadedRecord record) {
    var steps = new Steps();
    var variant = record.game().variant();
    var driver = new ReplayDriver(variant);
    var ws = new RuleWorkspace(variant);
    for (int index = 0; index < record.rounds().size(); index++) {
      RecordedRound round = record.rounds().get(index);
      var table = new RecordTable(round);
      Assert.assertEquals(record.firstStep(index), steps.next);
      steps.event("start_kyoku", -1, table.state());
      List<RecordEvent> events = round.events();
      int applied = 0;
      Boundary boundary = driver.start(round, ws);
      while (boundary == Boundary.DECISIONS) {
        int cursor = driver.cursor();
        for (; applied < cursor; applied++) apply(record, index, applied, table, steps);
        Decisions decisions = driver.engine().decisions();
        for (int slot = 0; slot < decisions.size(); slot++)
          steps.list.add(
              new Expected(
                  true,
                  null,
                  decisions.seat(slot),
                  record.step(index, cursor),
                  steps.next - 1,
                  table.state()));
        boundary = driver.step(ws);
      }
      for (; applied < events.size(); applied++) apply(record, index, applied, table, steps);
      Assert.assertEquals(record.step(index, events.size()), steps.next);
      RoundResult result = round.result();
      RoundEnd end = driver.engine().roundEnd();
      if (result.win())
        for (int win = 0; win < result.wins().size(); win++) {
          table.settleWin(end, win);
          steps.event("hora", result.wins().get(win).seat(), table.state());
        }
      else {
        table.settleDraw(end);
        steps.event("ryukyoku", -1, table.state());
      }
      steps.event("end_kyoku", -1, table.state());
    }
    steps.event("end_game", -1, steps.list.getLast().table());
    Assert.assertEquals(record.stepCount(), steps.next);
    return steps.list;
  }

  /**
   * 記録のイベントを表示の卓に当てる。立直の宣言牌の打牌では、その前に立直の宣言の手順を置く。
   *
   * @param position 局の中のイベントの位置
   */
  private static void apply(
      LoadedRecord record, int round, int position, RecordTable table, Steps steps) {
    Assert.assertEquals(record.step(round, position), steps.next);
    RecordEvent event = record.rounds().get(round).events().get(position);
    int actor = EventKind.actor(event);
    if (EventKind.declaresRiichi(event))
      steps.event(EventKind.REACH.wireName(), actor, steps.list.getLast().table());
    table.apply(event);
    EventKind kind = EventKind.of(event);
    steps.event(kind.wireName(), kind.hasActor() ? actor : -1, table.state());
  }

  /** 期待する手順の並びと、次のイベントの手順の位置。判断の手順は位置を持たず、位置を進めない。 */
  private static final class Steps {
    final List<Expected> list = new ArrayList<>();
    int next;

    void event(String name, int actor, TableState table) {
      list.add(new Expected(false, name, actor, next++, -1, table));
    }
  }

  private static ResultWriter writer() throws IOException {
    return new ResultWriter(
        OutputStream.nullOutputStream(),
        "id",
        "2026-11-04T00:00:00Z",
        Format6.MODEL,
        new RecordSummary("mjai", List.of("A", "B", "C", "D"), List.of(25000, 25000, 25000, 25000), 1));
  }

  /** 候補が 2 つの判断。手の番号に意味は無く、書き分けを見るためだけに使う。 */
  private static EvaluatedDecision decision(boolean valued) {
    return new EvaluatedDecision(
        new int[] {5, 371},
        valued ? new float[] {0f, 0.001f} : new float[] {0.75f, 0.25f},
        new String[2][],
        1,
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

  /** 期待する手順 1 つ。cause を持つのは判断だけ。 */
  private record Expected(
      boolean decision, String event, int actor, int eventIndex, int cause, TableState table) {}
}
