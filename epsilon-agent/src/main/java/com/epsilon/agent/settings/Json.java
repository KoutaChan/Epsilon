package com.epsilon.agent.settings;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.google.gson.Strictness;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonToken;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * 設定と、プログラムが書くファイルの JSON を読み書きする。オブジェクトはテーブル、配列は配列のノードにし、数はリテラルのまま持つ。
 * 値が null のフィールドは、無いものとして読む。同じキーが 2 回出るのと、配列の中の null はエラー。
 */
public final class Json {

  private Json() {}

  /**
   * JSON のオブジェクトを読む。
   *
   * @param origin 書かれた場所。全てのノードがこれを持つ
   * @throws SettingsException 読めないか、オブジェクトでないとき
   */
  public static Node.Table read(String text, String origin) {
    try (JsonReader reader = new JsonReader(new StringReader(text))) {
      reader.setStrictness(Strictness.STRICT);
      Node node = value(reader, origin);
      if (reader.peek() != JsonToken.END_DOCUMENT) {
        throw problem(origin, "unexpected text after the JSON value");
      }
      if (node instanceof Node.Table table) {
        return table;
      }
      throw problem(origin, "expected a JSON object");
    } catch (IOException | IllegalStateException unreadable) {
      throw problem(origin, unreadable.getMessage());
    }
  }

  /**
   * Gson の木を設定の木にする。
   *
   * @param origin 書かれた場所。全てのノードがこれを持つ
   * @throws SettingsException 配列の中に null があるとき
   */
  public static Node tree(JsonElement element, String origin) {
    if (element.isJsonObject()) {
      JsonObject object = element.getAsJsonObject();
      Map<String, Node> entries = new LinkedHashMap<>();
      object.entrySet().stream()
          .filter(entry -> !entry.getValue().isJsonNull())
          .forEach(entry -> entries.put(entry.getKey(), tree(entry.getValue(), origin)));
      return new Node.Table(entries, origin);
    }
    if (element.isJsonArray()) {
      List<Node> items = new ArrayList<>();
      for (JsonElement item : element.getAsJsonArray()) {
        if (item.isJsonNull()) {
          throw problem(origin, "null inside a list");
        }
        items.add(tree(item, origin));
      }
      return new Node.Array(items, origin);
    }
    if (element.isJsonPrimitive()) {
      JsonPrimitive primitive = element.getAsJsonPrimitive();
      if (primitive.isBoolean()) {
        return new Node.Bool(primitive.getAsBoolean(), origin);
      }
      if (primitive.isNumber()) {
        return new Node.Numeral(primitive.getAsNumber().toString(), origin);
      }
      return new Node.Text(primitive.getAsString(), origin);
    }
    throw problem(origin, "expected a value but found null");
  }

  /** 木を 1 行の JSON にする。キーは木の順に書く。 */
  public static String write(Node node) {
    StringBuilder out = new StringBuilder();
    write(out, node);
    return out.toString();
  }

  private static void write(StringBuilder out, Node node) {
    switch (node) {
      case Node.Table table -> {
        out.append('{');
        boolean first = true;
        for (Map.Entry<String, Node> entry : table.entries().entrySet()) {
          if (!first) {
            out.append(',');
          }
          first = false;
          string(out, entry.getKey());
          out.append(':');
          write(out, entry.getValue());
        }
        out.append('}');
      }
      case Node.Array array -> {
        out.append('[');
        for (int index = 0; index < array.items().size(); index++) {
          if (index > 0) {
            out.append(',');
          }
          write(out, array.items().get(index));
        }
        out.append(']');
      }
      case Node.Text text -> string(out, text.value());
      case Node.Numeral numeral -> out.append(numeral.literal());
      case Node.Bool bool -> out.append(bool.value());
    }
  }

  private static void string(StringBuilder out, String value) {
    out.append('"');
    for (int index = 0; index < value.length(); index++) {
      char c = value.charAt(index);
      switch (c) {
        case '"' -> out.append("\\\"");
        case '\\' -> out.append("\\\\");
        case '\n' -> out.append("\\n");
        case '\r' -> out.append("\\r");
        case '\t' -> out.append("\\t");
        default -> {
          if (c < 0x20 || c == 0x2028 || c == 0x2029) {
            out.append(String.format("\\u%04x", (int) c));
          } else {
            out.append(c);
          }
        }
      }
    }
    out.append('"');
  }

  private static Node value(JsonReader reader, String origin) throws IOException {
    return switch (reader.peek()) {
      case BEGIN_OBJECT -> {
        reader.beginObject();
        Map<String, Node> entries = new LinkedHashMap<>();
        Set<String> names = new HashSet<>();
        while (reader.hasNext()) {
          String name = reader.nextName();
          if (!names.add(name)) {
            throw problem(origin, "duplicate key " + name + " at " + reader.getPath());
          }
          if (reader.peek() == JsonToken.NULL) {
            reader.nextNull();
          } else {
            entries.put(name, value(reader, origin));
          }
        }
        reader.endObject();
        yield new Node.Table(entries, origin);
      }
      case BEGIN_ARRAY -> {
        reader.beginArray();
        List<Node> items = new ArrayList<>();
        while (reader.hasNext()) {
          if (reader.peek() == JsonToken.NULL) {
            throw problem(origin, "null inside a list at " + reader.getPath());
          }
          items.add(value(reader, origin));
        }
        reader.endArray();
        yield new Node.Array(items, origin);
      }
      case STRING -> new Node.Text(reader.nextString(), origin);
      case NUMBER -> new Node.Numeral(reader.nextString(), origin);
      case BOOLEAN -> new Node.Bool(reader.nextBoolean(), origin);
      default -> throw problem(origin, "unexpected " + reader.peek() + " at " + reader.getPath());
    };
  }

  private static SettingsException problem(String origin, String detail) {
    return new SettingsException(List.of("Cannot read " + origin + ": " + detail));
  }
}
