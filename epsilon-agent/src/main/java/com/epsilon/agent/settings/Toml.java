package com.epsilon.agent.settings;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * 設定の TOML の読み書き。
 *
 * <p>読めるのは TOML の一部だけ。テーブル {@code [a.b]}、テーブルの配列 {@code [[a.b]]}、ドットでつないだキー、行末の {@code #} の
 * コメントと、値では文字列・整数・小数・真偽値・配列・インラインテーブルを読む。{@code "…"} の文字列では {@code \"}・{@code \\}・
 * {@code \n}・{@code \t} と、u に 16 進 4 桁を続けた形のエスケープを使え、{@code '…'} の文字列はそのまま読む。整数は {@code _} で
 * 区切ってよく、小数は指数を付けてよい。配列は行をまたいでよく、最後にカンマを置いてよい。同じキーを 2 回書くこと、日時、複数行の文字列、
 * inf と nan、ほかの書き方は、行の番号を添えたエラーになる。
 */
public final class Toml {

  private static final Pattern NUMBER =
      Pattern.compile("[+-]?(0|[1-9](_?[0-9])*)(\\.[0-9](_?[0-9])*)?([eE][+-]?[0-9](_?[0-9])*)?");
  private static final Pattern DATE = Pattern.compile("\\d{4}-\\d{2}-\\d{2}.*|\\d{2}:\\d{2}.*");

  private Toml() {}

  /**
   * 設定ファイルの中身を読む。
   *
   * @param origin 書かれた場所の前半（ファイルのパスか、同梱の層の名前）。各ノードの書かれた場所は {@code <origin>:<行>} になる
   * @throws SettingsException 読めない書き方があった
   */
  public static Node.Table read(String text, String origin) {
    return new Parser(text, origin, true).document();
  }

  /**
   * コマンドラインの値と、注釈に書いた既定値を読む。TOML の値の書き方に加えて、引用符の無い語も文字列として読む（配列とインラインテーブルの
   * 中でも）。全体を 1 つの値として読めないとき、括弧か引用符で始まるならエラーにし、そうでなければ前後の空白を除いた全体を 1 つの文字列にする。
   * 空の値は空の文字列になる。
   *
   * @param origin 書かれた場所（{@code --set}、{@code default} など）
   * @throws SettingsException 括弧か引用符で始まる値が読めない
   */
  public static Node value(String text, String origin) {
    Parser parser = new Parser(text, origin, false);
    parser.skipBlank();
    if (parser.atEnd()) {
      return new Node.Text("", origin);
    }
    char first = parser.peek();
    boolean structured = first == '[' || first == '{' || first == '"' || first == '\'';
    try {
      Node node = parser.value();
      parser.skipBlank();
      if (parser.atEnd()) {
        return node;
      }
      if (structured) {
        throw parser.error("unexpected text after the value");
      }
    } catch (SettingsException unreadable) {
      if (structured) {
        throw unreadable;
      }
    }
    return new Node.Text(text.strip(), origin);
  }

  /** 木を TOML に書く。読み直すと同じ木になる。 */
  public static String write(Node.Table root) {
    StringBuilder out = new StringBuilder();
    table(out, root, "");
    return out.toString();
  }

  /** 値を、TOML で 1 行に書く形にする。数は書いたままの形で書く。 */
  public static String inline(Node node) {
    return switch (node) {
      case Node.Text text -> quote(text.value());
      case Node.Numeral numeral -> numeral.written();
      case Node.Bool bool -> Boolean.toString(bool.value());
      case Node.Array array -> array.items().stream().map(Toml::inline).collect(Collectors.joining(", ", "[", "]"));
      case Node.Table table ->
          table.entries().entrySet().stream()
              .map(entry -> Keys.quote(entry.getKey()) + " = " + inline(entry.getValue()))
              .collect(Collectors.joining(", ", "{", "}"));
    };
  }

  /** 「パス=値」の並びを 1 つの木にする。値は {@link #value} と同じ書き方で読む。 */
  static Node.Table assignments(List<String> assignments, String origin) {
    Building root = new Building(origin, State.HEADER);
    for (String assignment : assignments) {
      Parser parser = new Parser(assignment, origin, false);
      parser.skipBlank();
      List<String> keys = parser.key();
      parser.skipBlank();
      parser.expect('=');
      root.assign(keys, value(assignment.substring(parser.pos), origin), parser);
    }
    return root.freeze();
  }

  /** ドットでつないだパスを、キーごとに分ける。 */
  static List<String> path(String path) {
    Parser parser = new Parser(path, path, false);
    parser.skipBlank();
    List<String> keys = parser.key();
    parser.skipBlank();
    if (!parser.atEnd()) {
      throw parser.error("unexpected text after the key");
    }
    return keys;
  }

  /** 文字列を引用符で囲み、エスケープした形にする。 */
  static String quote(String value) {
    StringBuilder out = new StringBuilder(value.length() + 2).append('"');
    for (int index = 0; index < value.length(); index++) {
      char c = value.charAt(index);
      switch (c) {
        case '"' -> out.append("\\\"");
        case '\\' -> out.append("\\\\");
        case '\n' -> out.append("\\n");
        case '\t' -> out.append("\\t");
        default -> {
          if (c < 0x20 || c == 0x7f) {
            out.append(String.format("\\u%04X", (int) c));
          } else {
            out.append(c);
          }
        }
      }
    }
    return out.append('"').toString();
  }

  private static void table(StringBuilder out, Node.Table table, String path) {
    for (Map.Entry<String, Node> entry : table.entries().entrySet()) {
      Node node = entry.getValue();
      if (section(node)) {
        continue;
      }
      out.append(Keys.quote(entry.getKey())).append(" = ").append(inline(node)).append('\n');
    }
    for (Map.Entry<String, Node> entry : table.entries().entrySet()) {
      if (entry.getValue() instanceof Node.Table child && section(child)) {
        String childPath = Keys.join(path, entry.getKey());
        if (child.entries().values().stream().anyMatch(node -> !section(node))) {
          if (!out.isEmpty()) {
            out.append('\n');
          }
          out.append('[').append(childPath).append("]\n");
        }
        table(out, child, childPath);
      }
    }
  }

  /** 見出しを付けたセクションとして書くテーブルか。空のテーブルは {@code {}} で書く。 */
  private static boolean section(Node node) {
    return node instanceof Node.Table table && !table.entries().isEmpty();
  }

  private enum State {
    /** 見出しで作ったテーブル。 */
    HEADER,
    /** 見出しのパスの途中で作ったテーブル。後から同じパスの見出しを 1 回だけ書ける。 */
    IMPLICIT,
    /** ドットでつないだキーで作ったテーブル。見出しでは開き直せない。ドットでつないだキーで中に足せるのは、この種類のテーブルだけ。 */
    DOTTED
  }

  /** 読んでいる途中のテーブル。値は {@link Building}・{@link TableList}・確定した {@link Node} のどれか。 */
  private static final class Building {
    private final Map<String, Object> entries = new LinkedHashMap<>();
    private final String origin;
    private State state;

    Building(String origin, State state) {
      this.origin = origin;
      this.state = state;
    }

    Building openTable(List<String> keys, String origin, Parser parser) {
      Building parent = descend(keys.subList(0, keys.size() - 1), origin, parser);
      Object existing = parent.entries.get(keys.getLast());
      if (existing == null) {
        Building table = new Building(origin, State.HEADER);
        parent.entries.put(keys.getLast(), table);
        return table;
      }
      if (existing instanceof Building table && table.state == State.IMPLICIT) {
        table.state = State.HEADER;
        return table;
      }
      throw parser.error("duplicate table [" + dotted(keys) + "]");
    }

    Building openList(List<String> keys, String origin, Parser parser) {
      Building parent = descend(keys.subList(0, keys.size() - 1), origin, parser);
      Object existing = parent.entries.get(keys.getLast());
      TableList list;
      if (existing == null) {
        list = new TableList(origin);
        parent.entries.put(keys.getLast(), list);
      } else if (existing instanceof TableList tables) {
        list = tables;
      } else {
        throw parser.error("duplicate key " + dotted(keys));
      }
      Building table = new Building(origin, State.HEADER);
      list.tables.add(table);
      return table;
    }

    private Building descend(List<String> keys, String origin, Parser parser) {
      Building table = this;
      for (String key : keys) {
        Object existing = table.entries.get(key);
        if (existing == null) {
          Building created = new Building(origin, State.IMPLICIT);
          table.entries.put(key, created);
          table = created;
        } else if (existing instanceof Building child) {
          table = child;
        } else if (existing instanceof TableList list) {
          table = list.tables.getLast();
        } else {
          throw parser.error(key + " is not a table");
        }
      }
      return table;
    }

    void assign(List<String> keys, Node value, Parser parser) {
      Building table = this;
      for (String key : keys.subList(0, keys.size() - 1)) {
        Object existing = table.entries.get(key);
        if (existing == null) {
          Building created = new Building(value.origin(), State.DOTTED);
          table.entries.put(key, created);
          table = created;
        } else if (existing instanceof Building child && child.state == State.DOTTED) {
          table = child;
        } else {
          throw parser.error("duplicate key " + dotted(keys));
        }
      }
      if (table.entries.putIfAbsent(keys.getLast(), value) != null) {
        throw parser.error("duplicate key " + dotted(keys));
      }
    }

    Node.Table freeze() {
      Map<String, Node> frozen = new LinkedHashMap<>();
      entries.forEach(
          (key, value) ->
              frozen.put(
                  key,
                  switch (value) {
                    case Building table -> table.freeze();
                    case TableList list -> list.freeze();
                    case Node node -> node;
                    default -> throw new IllegalStateException("Unexpected entry " + value);
                  }));
      return new Node.Table(frozen, origin);
    }

    private static String dotted(List<String> keys) {
      return keys.stream().map(Keys::quote).collect(Collectors.joining("."));
    }
  }

  /** テーブルの配列（{@code [[a.b]]}）。 */
  private static final class TableList {
    private final List<Building> tables = new ArrayList<>();
    private final String origin;

    TableList(String origin) {
      this.origin = origin;
    }

    Node.Array freeze() {
      return new Node.Array(tables.stream().map(Building::freeze).map(Node.class::cast).toList(), origin);
    }
  }

  /**
   * 文書全体か、1 つの値を読む。文書は TOML のとおりに読み、書かれた場所に行の番号を付ける。値では引用符の無い語も文字列として読み、
   * コメントは扱わない。
   */
  private static final class Parser {
    private final String text;
    private final String label;
    private final boolean document;
    private int pos;
    private int line = 1;

    Parser(String text, String label, boolean document) {
      this.text = text;
      this.label = label;
      this.document = document;
    }

    Node.Table document() {
      Building root = new Building(origin(), State.HEADER);
      Building current = root;
      while (true) {
        skipBlank();
        if (atEnd()) {
          return root.freeze();
        }
        char c = peek();
        if (c == '\n' || c == '\r') {
          newline();
          continue;
        }
        if (c == '#') {
          skipComment();
          continue;
        }
        if (c == '[') {
          current = header(root);
        } else {
          List<String> keys = key();
          skipBlank();
          expect('=');
          skipBlank();
          current.assign(keys, value(), this);
        }
        endOfLine();
      }
    }

    private Building header(Building root) {
      String origin = origin();
      pos++;
      boolean list = !atEnd() && peek() == '[';
      if (list) {
        pos++;
      }
      skipBlank();
      List<String> keys = key();
      skipBlank();
      expect(']');
      if (list) {
        expect(']');
      }
      return list ? root.openList(keys, origin, this) : root.openTable(keys, origin, this);
    }

    List<String> key() {
      List<String> keys = new ArrayList<>();
      while (true) {
        skipBlank();
        keys.add(simpleKey());
        skipBlank();
        if (atEnd() || peek() != '.') {
          return keys;
        }
        pos++;
      }
    }

    private String simpleKey() {
      if (atEnd()) {
        throw error("expected a key");
      }
      char c = peek();
      if (c == '"') {
        return basicString();
      }
      if (c == '\'') {
        return literalString();
      }
      int start = pos;
      while (!atEnd() && Keys.bareChar(peek())) {
        pos++;
      }
      if (pos == start) {
        throw error("expected a key but found " + describe(c));
      }
      return text.substring(start, pos);
    }

    Node value() {
      if (atEnd()) {
        throw error("expected a value");
      }
      String origin = origin();
      char c = peek();
      if (c == '"' || c == '\'') {
        if (text.startsWith(c == '"' ? "\"\"\"" : "'''", pos)) {
          throw error("multi-line strings are not supported");
        }
        return new Node.Text(c == '"' ? basicString() : literalString(), origin);
      }
      if (c == '[') {
        return array(origin);
      }
      if (c == '{') {
        return inlineTable(origin);
      }
      int start = pos;
      while (!atEnd() && !delimiter(peek())) {
        pos++;
      }
      String token = text.substring(start, pos);
      if (token.isEmpty()) {
        throw error("expected a value but found " + describe(c));
      }
      if (token.equals("true") || token.equals("false")) {
        return new Node.Bool(token.equals("true"), origin);
      }
      if (NUMBER.matcher(token).matches()) {
        String literal = token.replace("_", "");
        return new Node.Numeral(literal.startsWith("+") ? literal.substring(1) : literal, token, origin);
      }
      if (!document) {
        return new Node.Text(token, origin);
      }
      String bare = token.startsWith("+") || token.startsWith("-") ? token.substring(1) : token;
      if (bare.equals("inf") || bare.equals("nan")) {
        throw error("inf and nan are not allowed");
      }
      if (DATE.matcher(token).matches()) {
        throw error("dates are not supported");
      }
      if (!bare.isEmpty() && Character.isDigit(bare.charAt(0))) {
        throw error("invalid number " + token);
      }
      throw error("expected a value but found " + token + " (strings need quotes)");
    }

    private Node.Array array(String origin) {
      pos++;
      List<Node> items = new ArrayList<>();
      while (true) {
        skipSpace();
        if (atEnd()) {
          throw error("unterminated list");
        }
        if (peek() == ']') {
          pos++;
          return new Node.Array(items, origin);
        }
        items.add(value());
        skipSpace();
        if (atEnd()) {
          throw error("unterminated list");
        }
        if (peek() == ',') {
          pos++;
        } else if (peek() != ']') {
          throw error("expected , or ] in the list but found " + describe(peek()));
        }
      }
    }

    private Node.Table inlineTable(String origin) {
      pos++;
      Building table = new Building(origin, State.HEADER);
      skipBlank();
      if (!atEnd() && peek() == '}') {
        pos++;
        return table.freeze();
      }
      while (true) {
        skipBlank();
        if (atEnd() || peek() == '\n' || peek() == '\r') {
          throw error(atEnd() ? "unterminated inline table" : "an inline table must fit on one line");
        }
        List<String> keys = key();
        skipBlank();
        expect('=');
        skipBlank();
        table.assign(keys, value(), this);
        skipBlank();
        if (atEnd()) {
          throw error("unterminated inline table");
        }
        if (peek() == '\n' || peek() == '\r') {
          throw error("an inline table must fit on one line");
        }
        if (peek() == '}') {
          pos++;
          return table.freeze();
        }
        if (peek() != ',') {
          throw error("expected , or } in the inline table but found " + describe(peek()));
        }
        pos++;
      }
    }

    private String basicString() {
      pos++;
      StringBuilder out = new StringBuilder();
      while (true) {
        if (atEnd() || peek() == '\n' || peek() == '\r') {
          throw error("unterminated string");
        }
        char c = text.charAt(pos++);
        if (c == '"') {
          return out.toString();
        }
        if (c != '\\') {
          if (c < 0x20 && c != '\t') {
            throw error("control character in a string");
          }
          out.append(c);
          continue;
        }
        if (atEnd()) {
          throw error("unterminated string");
        }
        char escaped = text.charAt(pos++);
        switch (escaped) {
          case '"' -> out.append('"');
          case '\\' -> out.append('\\');
          case 'n' -> out.append('\n');
          case 't' -> out.append('\t');
          case 'u' -> out.append(unicode());
          default -> throw error("invalid escape \\" + escaped);
        }
      }
    }

    private char unicode() {
      if (pos + 4 > text.length()) {
        throw error("invalid escape \\u");
      }
      String hex = text.substring(pos, pos + 4);
      if (!hex.chars().allMatch(c -> Character.digit(c, 16) >= 0)) {
        throw error("invalid escape \\u" + hex);
      }
      char c = (char) Integer.parseInt(hex, 16);
      if (Character.isSurrogate(c)) {
        throw error("invalid escape \\u" + hex);
      }
      pos += 4;
      return c;
    }

    private String literalString() {
      int start = ++pos;
      while (!atEnd() && peek() != '\'') {
        if (peek() == '\n' || peek() == '\r') {
          throw error("unterminated string");
        }
        pos++;
      }
      if (atEnd()) {
        throw error("unterminated string");
      }
      return text.substring(start, pos++);
    }

    /** 値の終わりを示す文字か。値を読むときは空白・改行・カンマ・閉じ括弧だけで、文書を読むときは {@code #} も含む。 */
    private boolean delimiter(char c) {
      return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ',' || c == ']' || c == '}'
          || (document && c == '#');
    }

    void skipBlank() {
      while (!atEnd() && (peek() == ' ' || peek() == '\t')) {
        pos++;
      }
    }

    /** 配列の中で、空白・改行・コメントを読み飛ばす。 */
    private void skipSpace() {
      while (!atEnd()) {
        char c = peek();
        if (c == ' ' || c == '\t') {
          pos++;
        } else if (c == '\n' || c == '\r') {
          newline();
        } else if (c == '#' && document) {
          skipComment();
        } else {
          return;
        }
      }
    }

    private void skipComment() {
      while (!atEnd() && peek() != '\n' && peek() != '\r') {
        pos++;
      }
    }

    private void newline() {
      if (peek() == '\r') {
        pos++;
        if (atEnd() || peek() != '\n') {
          throw error("a carriage return must be followed by a line feed");
        }
      }
      pos++;
      line++;
    }

    private void endOfLine() {
      skipBlank();
      if (!atEnd() && peek() == '#') {
        skipComment();
      }
      if (atEnd()) {
        return;
      }
      if (peek() != '\n' && peek() != '\r') {
        throw error("expected the end of the line but found " + describe(peek()));
      }
      newline();
    }

    void expect(char c) {
      if (atEnd() || peek() != c) {
        throw error("expected " + c + (atEnd() ? " at the end" : " but found " + describe(peek())));
      }
      pos++;
    }

    boolean atEnd() {
      return pos >= text.length();
    }

    char peek() {
      return text.charAt(pos);
    }

    private String origin() {
      return document ? label + ":" + line : label;
    }

    private static String describe(char c) {
      return c == '\n' || c == '\r' ? "the end of the line" : "'" + c + "'";
    }

    SettingsException error(String detail) {
      return new SettingsException(
          List.of(
              document
                  ? "Cannot read " + label + " at line " + line + ": " + detail
                  : "Cannot read " + text + " at " + label + ": " + detail));
    }
  }
}
