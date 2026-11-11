package com.epsilon.agent.settings;

import static com.epsilon.agent.settings.Fixtures.array;
import static com.epsilon.agent.settings.Fixtures.bool;
import static com.epsilon.agent.settings.Fixtures.number;
import static com.epsilon.agent.settings.Fixtures.table;
import static com.epsilon.agent.settings.Fixtures.text;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.google.gson.JsonParser;
import java.util.List;
import org.testng.annotations.Test;

/** JSON の読み書き。TOML と同じ木になること、同じキーが 2 回出たときと null の扱い、書いて読み直すと同じ木に戻ることを確かめる。 */
public class JsonTest {

  private static final String JSON =
      """
      {"a": 1, "b": 1.50, "c": "x\\n\\u00e9", "d": true, "e": [1, 2.5e3, "s", [], {}],
       "f": {"g": {"h": "i"}, "skip": null}, "t": [{"k": true}, {"k": false}], "big": 123456789012345678901234567890}
      """;

  private static final String TOML =
      """
      a = 1
      b = 1.5
      c = "x\\n\\u00e9"
      d = true
      e = [1, 2500, "s", [], {}]
      big = 123456789012345678901234567890
      [f.g]
      h = "i"
      [[t]]
      k = true
      [[t]]
      k = false
      """;

  @Test
  public void jsonAndTomlGiveTheSameTree() {
    Node.Table json = Json.read(JSON, "x.json");
    assertEquals(json, Toml.read(TOML, "x.toml"));
    assertEquals(json.get("a").origin(), "x.json");
    assertEquals(((Node.Numeral) json.get("b")).literal(), "1.50");
    assertTrue(((Node.Numeral) json.get("a")).integer());
    assertFalse(((Node.Numeral) json.get("b")).integer());
    assertEquals(Json.tree(JsonParser.parseString(JSON), "x.json"), json);
  }

  @Test
  public void nullFieldsAreAbsent() {
    assertEquals(Json.read("{\"a\": null, \"b\": {\"c\": null}}", "n.json"), table("b", table()));
    assertEquals(Json.tree(JsonParser.parseString("{\"a\": null, \"b\": 1}"), "n.json"), table("b", number("1")));
  }

  @Test
  public void duplicateKeysAreErrors() {
    SettingsException top = expectThrows(SettingsException.class, () -> Json.read("{\"a\":1,\"a\":2}", "dup.json"));
    assertEquals(top.problems(), List.of("Cannot read dup.json: duplicate key a at $.a"));
    SettingsException nested =
        expectThrows(SettingsException.class, () -> Json.read("{\"x\":{\"a\":1,\"b\":2,\"a\":null}}", "dup.json"));
    assertEquals(nested.problems(), List.of("Cannot read dup.json: duplicate key a at $.x.a"));
  }

  @Test
  public void malformedJsonIsAnError() {
    for (String text : List.of("[1]", "\"x\"", "{} {}", "{a: 1}", "{\"a\": 1,}", "{\"a\": [1, null]}", "", "{\"a\": NaN}")) {
      SettingsException thrown = expectThrows(SettingsException.class, () -> Json.read(text, "bad.json"));
      assertEquals(thrown.problems().size(), 1, text);
      assertTrue(thrown.getMessage().startsWith("Cannot read bad.json: "), thrown.getMessage());
    }
    SettingsException list =
        expectThrows(SettingsException.class, () -> Json.tree(JsonParser.parseString("[1, null]"), "bad.json"));
    assertEquals(list.problems(), List.of("Cannot read bad.json: null inside a list"));
    SettingsException notObject =
        expectThrows(SettingsException.class, () -> Layer.json("models.json", JsonParser.parseString("[1]")));
    assertEquals(notObject.problems(), List.of("Settings at models.json must be a JSON object"));
  }

  @Test
  public void writeReadsBackTheSameTree() {
    Node.Table tree =
        table(
            "text", text("quote \" back \\ cr \r lf \n tab \t nul \0 ls   ps   é 漢"),
            "numbers", array(number("-0"), number("1E+5"), number("1.0E-5"), number("9223372036854775807")),
            "flags", array(bool(true), bool(false)),
            "nested", table("empty", table(), "list", array(), "a.b", table("c", text(""))));
    String written = Json.write(tree);
    assertFalse(written.contains("\n"), written);
    assertEquals(Json.read(written, "again"), tree);
    assertEquals(Toml.read(Toml.write(Json.read(written, "again")), "again"), tree);
    assertEquals(Json.read(Json.write(Json.read(JSON, "x.json")), "again"), Json.read(JSON, "x.json"));
  }

  @Test
  public void numbersCompareByValue() {
    assertEquals(number("1"), number("1.0"));
    assertEquals(number("1").hashCode(), number("1.000").hashCode());
    assertEquals(number("100").hashCode(), number("1E2").hashCode());
    assertEquals(number("0").hashCode(), number("0.0").hashCode());
    assertFalse(number("1").equals(text("1")));
    assertEquals(new Node.Numeral("1", "a"), new Node.Numeral("1", "b"));
    expectThrows(IllegalArgumentException.class, () -> new Node.Numeral("1_0", "x"));
  }
}
