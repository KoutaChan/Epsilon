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

import java.util.List;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 設定の TOML の読み書き。読める全ての書き方、誤った書き方ごとのエラーの行番号、書いて読み直すと同じ木に戻ることを確かめる。 */
public class TomlTest {

  private static final String DOCUMENT =
      """
      # 頭のコメント
      title = "a \\"q\\" \\\\ \\n \\t \\u00e9 # not a comment"   # 行末のコメント
      raw = 'C:\\dir\\x'
      count = 1_000
      negative = -42
      positive = +7
      ratio = 1.5e-3
      big = 6.02E23
      zero = 0
      yes = true
      no = false
      list = [
        1, # one
        2,
      ]
      nested = [[1, 2], ["a"], []]
      inline = {a = 1, b.c = "x", empty = {}}
      dotted.key = 3
      dotted.more = 4
      "quoted key" = 5
      'literal key' = 6

      [section]
      value = 1
      inner.leaf = 2

      [ section . child ]
      deep = "d"

      [section.inner.extra]
      ok = true

      [[items]]
      name = "first"

      [[items]]
      name = "second"
      [items.sub]
      flag = true
      """;

  private static final Node.Table EXPECTED =
      table(
          "title", text("a \"q\" \\ \n \t \u00e9 # not a comment"),
          "raw", text("C:\\dir\\x"),
          "count", number("1000"),
          "negative", number("-42"),
          "positive", number("7"),
          "ratio", number("0.0015"),
          "big", number("6.02E23"),
          "zero", number("0"),
          "yes", bool(true),
          "no", bool(false),
          "list", array(number("1"), number("2")),
          "nested", array(array(number("1"), number("2")), array(text("a")), array()),
          "inline", table("a", number("1"), "b", table("c", text("x")), "empty", table()),
          "dotted", table("key", number("3"), "more", number("4")),
          "quoted key", number("5"),
          "literal key", number("6"),
          "section",
              table(
                  "value", number("1"),
                  "inner", table("leaf", number("2"), "extra", table("ok", bool(true))),
                  "child", table("deep", text("d"))),
          "items",
              array(
                  table("name", text("first")),
                  table("name", text("second"), "sub", table("flag", bool(true)))));

  @Test
  public void readsEveryForm() {
    Node.Table tree = Toml.read(DOCUMENT, "t.toml");
    assertEquals(tree, EXPECTED);
    assertEquals(tree.get("title").origin(), "t.toml:2");
    assertEquals(tree.get("list").origin(), "t.toml:12");
    assertEquals(tree.get("nested").origin(), "t.toml:16");
    assertEquals(tree.get("section").origin(), "t.toml:23");
    assertEquals(((Node.Table) tree.get("section")).get("child").origin(), "t.toml:27");
    Node.Array items = (Node.Array) tree.get("items");
    assertEquals(items.items().get(1).origin(), "t.toml:36");
    assertEquals(((Node.Table) items.items().get(1)).get("sub").origin(), "t.toml:38");
    assertTrue(((Node.Numeral) tree.get("count")).integer());
    assertFalse(((Node.Numeral) tree.get("ratio")).integer());
  }

  @Test
  public void crlfReadsLikeLf() {
    String lf = "a = 1\nb = [\n  2,\n]\n[c]\nd = 'x'\n";
    Node.Table crlf = Toml.read(lf.replace("\n", "\r\n"), "w.toml");
    assertEquals(crlf, Toml.read(lf, "u.toml"));
    assertEquals(((Node.Table) crlf.get("c")).get("d").origin(), "w.toml:6");
  }

  @Test
  public void emptyDocumentIsAnEmptyTable() {
    assertEquals(Toml.read("", "e.toml"), table());
    assertEquals(Toml.read("\n# only a comment\n\n", "e.toml"), table());
  }

  @DataProvider
  public Object[][] errors() {
    return new Object[][] {
      {"a = 1\na = 2", 2, "duplicate key a"},
      {"a.b = 1\na.b = 2", 2, "duplicate key a.b"},
      {"[a]\nb = 1\n[a]", 3, "duplicate table [a]"},
      {"[a]\nb = 1\n\n[a.b]", 4, "duplicate table [a.b]"},
      {"a.b = 1\n[a]", 2, "duplicate table [a]"},
      {"[a]\nx.y = 1\n[a.x]", 3, "duplicate table [a.x]"},
      {"a = 1\n[a.b]", 2, "a is not a table"},
      {"[[t]]\n[t]", 2, "duplicate table [t]"},
      {"[t]\n[[t]]", 2, "duplicate key t"},
      {"a = {b = 1}\n[a]", 2, "duplicate table [a]"},
      {"a = {b = 1}\na.c = 2", 2, "duplicate key a.c"},
      {"a = {b = 1, b = 2}", 1, "duplicate key b"},
      {"x = 1\ny = inf", 2, "inf and nan are not allowed"},
      {"y = -nan", 1, "inf and nan are not allowed"},
      {"\nd = 1979-05-27", 2, "dates are not supported"},
      {"d = 1979-05-27T07:32:00Z", 1, "dates are not supported"},
      {"d = 07:32:00", 1, "dates are not supported"},
      {"s = \"\"\"x\"\"\"", 1, "multi-line strings are not supported"},
      {"s = '''x'''", 1, "multi-line strings are not supported"},
      {"a = [\n  1,\n  2", 3, "unterminated list"},
      {"a = [1 2]", 1, "expected , or ] in the list but found '2'"},
      {"a = {b = 1,\n c = 2}", 1, "an inline table must fit on one line"},
      {"a = {b = 1\n}", 1, "an inline table must fit on one line"},
      {"a = {b = 1,}", 1, "expected a key but found '}'"},
      {"a = {b = 1 c = 2}", 1, "expected , or } in the inline table but found 'c'"},
      {"a = 01", 1, "invalid number 01"},
      {"a = 1__0", 1, "invalid number 1__0"},
      {"a = 1.", 1, "invalid number 1."},
      {"a = 0x1F", 1, "invalid number 0x1F"},
      {"a = \"x\\q\"", 1, "invalid escape \\q"},
      {"a = \"\\uD800\"", 1, "invalid escape \\uD800"},
      {"a = \"\\u00g0\"", 1, "invalid escape \\u00g0"},
      {"a = \"\\r\"", 1, "invalid escape \\r"},
      {"a = bare", 1, "expected a value but found bare (strings need quotes)"},
      {"a = \"open\nb = 1", 1, "unterminated string"},
      {"a = 'open", 1, "unterminated string"},
      {"a = \"tab\u0001\"", 1, "control character in a string"},
      {"a 1", 1, "expected = but found '1'"},
      {"a = 1 b = 2", 1, "expected the end of the line but found 'b'"},
      {"= 1", 1, "expected a key but found '='"},
      {"a =", 1, "expected a value"},
      {"a = \nb = 1", 1, "expected a value but found the end of the line"},
      {"[a", 1, "expected ] at the end"},
      {"[[a]\n", 1, "expected ] but found the end of the line"},
      {"a = 1\rb = 2", 1, "a carriage return must be followed by a line feed"},
      {"# c\n\n  # d\nx = 1\n\nx = 2", 6, "duplicate key x"},
      {"a = [\n  1, # one\n  2,\n]\nb = [", 5, "unterminated list"},
    };
  }

  @Test(dataProvider = "errors")
  public void errorsNameTheLine(String text, int line, String detail) {
    SettingsException thrown = expectThrows(SettingsException.class, () -> Toml.read(text, "e.toml"));
    assertEquals(thrown.problems(), List.of("Cannot read e.toml at line " + line + ": " + detail));
  }

  @Test
  public void writeReadsBackTheSameTree() {
    Node.Table tree = Toml.read(DOCUMENT, "t.toml");
    assertEquals(Toml.read(Toml.write(tree), "again"), tree);
  }

  @Test
  public void writeEscapesAnyTextAndKey() {
    Node.Table tree =
        table(
            "plain", text("x"),
            "a.b", text("quote \" back \\ cr \r lf \n tab \t bell \u0007 del \u007f é"),
            "with space", array(text(""), text("'")),
            "", number("1"),
            "deep",
                table(
                    "only",
                    table("tables", table("leaf", bool(false)), "empty", table(), "list", array())));
    String written = Toml.write(tree);
    assertEquals(Toml.read(written, "again"), tree);
    assertTrue(written.contains("\"a.b\" = "), written);
    assertTrue(written.contains("[deep.only]"), written);
  }


  @Test
  public void commandLineValues() {
    assertEquals(Toml.value("prism:/runs/x/snapshots/00824", "--set"), text("prism:/runs/x/snapshots/00824"));
    assertEquals(Toml.value("nano:/x y", "--set"), text("nano:/x y"));
    assertEquals(Toml.value("  padded  ", "--set"), text("padded"));
    assertEquals(Toml.value("", "--set"), text(""));
    assertEquals(Toml.value("1e-3", "--set"), number("0.001"));
    assertEquals(Toml.value("+1_000", "--set"), number("1000"));
    assertEquals(Toml.value("0123", "--set"), text("0123"));
    assertEquals(Toml.value("inf", "--set"), text("inf"));
    assertEquals(Toml.value("2026-11-12T03:00:00Z", "--set"), text("2026-11-12T03:00:00Z"));
    assertEquals(Toml.value("a # b", "--set"), text("a # b"));
    assertEquals(Toml.value("1 2", "--set"), text("1 2"));
    assertEquals(Toml.value("true", "--set"), bool(true));
    assertEquals(Toml.value("\"true\"", "--set"), text("true"));
    assertEquals(Toml.value("[cpu, 2, \"x y\"]", "--set"), array(text("cpu"), number("2"), text("x y")));
    assertEquals(Toml.value("{a = b, c = [1]}", "--set"), table("a", text("b"), "c", array(number("1"))));
    assertEquals(Toml.value("1", "--set").origin(), "--set");
  }

  @Test
  public void unreadableCommandLineValues() {
    for (String value : List.of("[1, 2", "{a = 1", "\"open", "[1] x", "{a = }")) {
      SettingsException thrown = expectThrows(SettingsException.class, () -> Toml.value(value, "--set"));
      assertTrue(thrown.getMessage().startsWith("Cannot read " + value + " at --set: "), thrown.getMessage());
    }
  }

  @Test
  public void assignmentsBuildOneTree() {
    Node.Table tree = Toml.assignments(List.of("a.b=1", "a.c = x", "\"d.e\"=[1]"), "--set");
    assertEquals(tree, table("a", table("b", number("1"), "c", text("x")), "d.e", array(number("1"))));
    for (List<String> twice : List.of(List.of("a=1", "a=2"), List.of("a.b=1", "a=2"), List.of("a=1", "a.b=2"))) {
      SettingsException thrown = expectThrows(SettingsException.class, () -> Toml.assignments(twice, "--set"));
      assertTrue(thrown.getMessage().contains("duplicate key"), thrown.getMessage());
    }
    expectThrows(SettingsException.class, () -> Toml.assignments(List.of("novalue"), "--set"));
  }

  @Test
  public void inlineWritesOneLine() {
    assertEquals(
        Toml.inline(table("a", array(number("1"), text("x")), "b c", table())), "{a = [1, \"x\"], \"b c\" = {}}");
  }
}
