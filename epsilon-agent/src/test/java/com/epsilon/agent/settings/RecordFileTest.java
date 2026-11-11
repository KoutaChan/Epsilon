package com.epsilon.agent.settings;

import static com.epsilon.agent.settings.Fixtures.number;
import static com.epsilon.agent.settings.Fixtures.table;
import static com.epsilon.agent.settings.Fixtures.text;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.settings.Fixtures.Kind;
import com.epsilon.agent.settings.Fixtures.Variant;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.testng.annotations.Test;

/** プログラムが書くファイル。注釈の無いフィールドは全て必須で、知らないフィールドはエラーになる。既定値の注釈があるフィールドは、無ければ読むときに既定値で補い、書き直した後はもう補わない。 */
public class RecordFileTest {

  record Part(String name, @Key(doc = "重み", value = "1.0") double weight) {}

  enum Phase {
    COLLECT,
    FIT
  }

  record State(
      int next,
      @Key(doc = "行の総数", value = "0") long rows,
      @Key(doc = "止めた理由", presence = Presence.ABSENT) String reason,
      Phase phase,
      Variant variant,
      Node.Table settings,
      @Key(doc = "前の値", presence = Presence.ABSENT) Node before,
      Instant since,
      Path directory,
      boolean paused,
      float share,
      List<Part> parts,
      Map<Kind, Integer> counts,
      Map<String, Part> named,
      Set<String> tags) {}

  private static State state() {
    return new State(
        3,
        10,
        null,
        Phase.FIT,
        Variant.THREE,
        table("fit", table("lrPeak", number("0.002")), "run", table("seed", number("7"))),
        number("4.0"),
        Instant.parse("2026-11-09T00:00:00Z"),
        Path.of("/runs/x"),
        true,
        0.1f,
        List.of(new Part("a", 0.5), new Part("b", 2.0)),
        Map.of(Kind.CALL, 2, Kind.KAN, 1),
        Map.of("x", new Part("x", 1.0)),
        Set.of("t1", "t2"));
  }

  @Test
  public void writeReadsBackTheSameRecord() throws Exception {
    State state = state();
    String json = RecordFile.json(state);
    assertFalse(json.contains("\n"), json);
    assertFalse(json.contains("reason"), json);
    RecordFile.Loaded<State> loaded = RecordFile.read(State.class, json, "state.json");
    assertEquals(loaded.value(), state);
    assertEquals(loaded.added(), List.of());
    assertEquals(RecordFile.read(State.class, RecordFile.tree(state)).value(), state);
    Path file = Files.createTempFile("state", ".json");
    try {
      Files.writeString(file, json, StandardCharsets.UTF_8);
      assertEquals(RecordFile.read(State.class, file).value(), state);
    } finally {
      Files.delete(file);
    }
  }

  @Test
  public void initialValuesComeInOnce() {
    String older =
        RecordFile.json(state())
            .replace("\"rows\":10,", "")
            .replace(",\"weight\":2.0", "")
            .replace("\"before\":4.0,", "");
    RecordFile.Loaded<State> loaded = RecordFile.read(State.class, older, "state.json");
    assertEquals(loaded.added(), List.of("rows", "parts[1].weight"));
    assertEquals(loaded.value().rows(), 0L);
    assertEquals(loaded.value().parts().get(1), new Part("b", 1.0));
    assertNull(loaded.value().before());
    RecordFile.Loaded<State> rewritten = RecordFile.read(State.class, RecordFile.json(loaded.value()), "state.json");
    assertEquals(rewritten.added(), List.of());
    assertEquals(rewritten.value(), loaded.value());
  }

  @Test
  public void missingAndUnknownFieldsAreErrors() {
    String json = RecordFile.json(state());
    SettingsException thrown =
        expectThrows(
            SettingsException.class,
            () ->
                RecordFile.read(
                    State.class,
                    json.replace("\"next\":3,", "").replace("\"paused\":true", "\"paused\":null,\"extra\":{\"deep\":1}"),
                    "state.json"));
    assertEquals(
        thrown.problems(),
        List.of(
            "Missing field next in state.json",
            "Missing field paused in state.json",
            "Unknown field extra.deep at state.json"));
    SettingsException part =
        expectThrows(
            SettingsException.class,
            () -> RecordFile.read(State.class, json.replace("{\"name\":\"a\",", "{"), "state.json"));
    assertEquals(part.problems(), List.of("Missing field parts[0].name in state.json"));
  }

  @Test
  public void wrongTypesAreErrors() {
    String json =
        RecordFile.json(state())
            .replace("\"next\":3", "\"next\":\"x\"")
            .replace("\"since\":\"2026-11-09T00:00:00Z\"", "\"since\":\"later\"")
            .replace("\"phase\":\"FIT\"", "\"phase\":\"PLAY\"")
            .replace("\"settings\":{", "\"settings\":[1],\"was\":{");
    SettingsException thrown = expectThrows(SettingsException.class, () -> RecordFile.read(State.class, json, "s.json"));
    assertEquals(
        thrown.problems(),
        List.of(
            "Invalid field next=x at s.json: expected an integer",
            "Invalid field phase=PLAY at s.json: expected one of [COLLECT, FIT]",
            "Invalid field settings=[1] at s.json: expected a table",
            "Invalid field since=later at s.json: expected an instant such as 2026-11-12T03:00:00Z",
            "Unknown field was.fit.lrPeak at s.json",
            "Unknown field was.run.seed at s.json"));
    expectThrows(SettingsException.class, () -> RecordFile.read(State.class, "{", "s.json"));
    expectThrows(SettingsException.class, () -> RecordFile.read(State.class, "[]", "s.json"));
  }

  @Test
  public void aNodeFieldHoldsAnyValue() {
    for (Node before : List.of(number("1"), text("x"), Fixtures.array(), table("a", Fixtures.bool(true)))) {
      State state = state();
      State changed =
          new State(
              state.next(),
              state.rows(),
              "stopped",
              state.phase(),
              state.variant(),
              state.settings(),
              before,
              state.since(),
              state.directory(),
              state.paused(),
              state.share(),
              state.parts(),
              state.counts(),
              state.named(),
              state.tags());
      assertEquals(RecordFile.read(State.class, RecordFile.json(changed), "state.json").value(), changed);
    }
  }
}
