package com.epsilon.agent.settings;

import com.epsilon.agent.settings.Schema.ChoiceCodec;
import com.epsilon.agent.settings.Schema.Codec;
import com.epsilon.agent.settings.Schema.Component;
import com.epsilon.agent.settings.Schema.ListCodec;
import com.epsilon.agent.settings.Schema.MapCodec;
import com.epsilon.agent.settings.Schema.RecordCodec;
import com.epsilon.agent.settings.Schema.ScalarCodec;
import com.epsilon.agent.settings.Schema.TreeCodec;
import java.lang.reflect.InvocationTargetException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.time.Instant;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Supplier;

/**
 * 重ねた木を record に結び付ける。エラーは全て集め、呼び出し側がまとめて投げる。結び付けた値ごとに、効いている木と、葉のキーごとの
 * 変えてよい時期を作る。効いている木では、既定値を埋め、種類を選ぶセクションをテーブルに直し、秘密の値を伏せる。文字列とパスのキーの
 * 値は、数や真偽値として書かれていても文字列のノードにする。
 */
final class Binder {

  /** 既定値から入れた値の、書かれた場所。 */
  static final String DEFAULT_ORIGIN = "default";

  /** 書き出しで、秘密の値の代わりに書く文字列。 */
  static final String MASK = "********";

  /** 結び付け方。 */
  enum Mode {
    /** 設定。コンポーネントは全て {@link Key} の注釈を持ち、無いキーは既定値で埋める。 */
    SETTINGS,
    /** プログラムが書くファイル。フィールドは全て必要で、{@link Key} の注釈に書いた既定値は初期値として入れる。 */
    RECORD
  }

  /**
   * 結び付けた値と、その値の効いているノード。
   *
   * @param node 効いているノード。ABSENT のコンポーネントで値が無ければ null
   */
  record Bound(Object value, Node node) {}

  private static final Object FAILED = new Object();

  private final Mode mode;
  private final Supplier<Set<String>> known;
  private final List<String> problems = new ArrayList<>();
  private final List<String> added = new ArrayList<>();
  private final Map<String, Change> changes = new HashMap<>();
  private Set<String> knownPaths;
  private int insideList;

  /** @param known 全体のキーのパス。知らないキーに近いキーを探すのに使い、要るときに 1 回だけ作る */
  Binder(Mode mode, Supplier<Set<String>> known) {
    this.mode = mode;
    this.known = known;
  }

  List<String> problems() {
    return problems;
  }

  /** 初期値を入れたフィールドのパス（プログラムが書くファイルのとき）。 */
  List<String> added() {
    return added;
  }

  /** 葉のキーごとの変えてよい時期。配列は 1 つの葉として扱う。 */
  Map<String, Change> changes() {
    return changes;
  }

  /**
   * テーブルを record に結び付ける。
   *
   * @param change 親のキーを変えてよい時期。いちばん外の record では null（そのときコンポーネントが INHERIT ならエラー）
   * @return 結び付けた値。エラーがあれば null
   */
  Bound record(RecordCodec codec, Node.Table table, String path, Change change) {
    return record(codec, table, path, change, null, null);
  }

  /** 1 つの値を、変えてよい時期を見ずに結び付ける。既定値と、層を選ぶキーを確かめるのに使う。 */
  Bound value(Codec codec, Node node, String path) {
    return value(codec, node, path, Change.INHERIT, false);
  }

  private Bound record(
      RecordCodec codec, Node.Table table, String path, Change change, ChoiceCodec choice, String chosen) {
    List<Component> components = codec.components();
    Object[] arguments = new Object[components.size()];
    Map<String, Node> effective = new LinkedHashMap<>();
    boolean failed = false;
    for (int index = 0; index < components.size(); index++) {
      Component component = components.get(index);
      String child = Keys.join(path, component.name());
      Key key = component.key();
      if (mode == Mode.SETTINGS && key == null) {
        throw new IllegalStateException("Setting " + child + " has no @Key");
      }
      Change childChange = key == null || key.change() == Change.INHERIT ? change : key.change();
      if (childChange == null) {
        throw new IllegalStateException("Top-level setting " + child + " must say when it may change");
      }
      Node node = table.get(component.name());
      if (node == null) {
        if (key != null && key.presence() == Presence.ABSENT) {
          continue;
        }
        node = key == null || key.presence() == Presence.REQUIRED ? null : component.fallback();
        if (node == null) {
          problems.add(
              "Missing " + noun() + " " + child + (mode == Mode.RECORD ? " in " + table.origin() : ""));
          failed = true;
          continue;
        }
        if (mode == Mode.RECORD) {
          added.add(child);
        }
      } else if (mode == Mode.SETTINGS) {
        node = Merge.withDefault(component, node);
      }
      Bound bound = value(component.codec(), node, child, childChange, key != null && key.secret());
      if (bound == null) {
        failed = true;
        continue;
      }
      arguments[index] = bound.value();
      effective.put(component.name(), bound.node());
    }
    for (Map.Entry<String, Node> entry : table.entries().entrySet()) {
      String name = entry.getKey();
      if (codec.component(name) == null && (choice == null || !name.equals(choice.key()))) {
        unknown(codec, path, name, entry.getValue(), choice, chosen);
      }
    }
    if (failed) {
      return null;
    }
    Node.Table node = new Node.Table(effective, table.origin());
    Object value = construct(codec, arguments, path, node);
    return value == null ? null : new Bound(value, node);
  }

  private Object construct(RecordCodec codec, Object[] arguments, String path, Node.Table node) {
    try {
      return codec.constructor().newInstance(arguments);
    } catch (InvocationTargetException failure) {
      Throwable cause = failure.getCause();
      if (cause instanceof InvalidSetting invalid) {
        String key = path.isEmpty() ? invalid.component() : path + "." + invalid.component();
        Node value = Keys.find(node, Arrays.asList(invalid.component().split("\\.")));
        problems.add(
            "Invalid " + noun() + " " + key + (value == null ? "" : "=" + Keys.text(value)) + " at "
                + (value == null ? node.origin() : value.origin()) + ": " + invalid.getMessage());
        return null;
      }
      if (cause instanceof RuntimeException runtime) {
        problems.add(
            "Invalid " + noun() + (path.isEmpty() ? "s" : " " + path) + " at " + node.origin() + ": "
                + runtime.getMessage());
        return null;
      }
      if (cause instanceof Error error) {
        throw error;
      }
      throw new IllegalStateException("Cannot construct " + codec.type().getName(), cause);
    } catch (ReflectiveOperationException failure) {
      throw new IllegalStateException("Cannot construct " + codec.type().getName(), failure);
    }
  }

  private void unknown(RecordCodec codec, String path, String name, Node node, ChoiceCodec choice, String chosen) {
    String child = Keys.join(path, name);
    if (choice != null) {
      for (RecordCodec other : choice.cases().values()) {
        if (other.component(name) != null) {
          problems.add(
              "Setting " + child + " at " + node.origin() + " does not apply to "
                  + Keys.join(path, choice.key()) + "=" + chosen);
          return;
        }
      }
    }
    List<String> candidates = new ArrayList<>();
    if (mode == Mode.SETTINGS) {
      if (knownPaths == null) {
        knownPaths = known.get();
      }
      candidates.addAll(knownPaths);
      codec.components().forEach(component -> candidates.add(Keys.join(path, component.name())));
    }
    unknownLeaves(child, node, candidates);
  }

  /** 知らないテーブルの中の葉を、1 つずつ知らないキーとして知らせる。 */
  private void unknownLeaves(String path, Node node, List<String> candidates) {
    if (node instanceof Node.Table table && !table.entries().isEmpty()) {
      table.entries().forEach((key, child) -> unknownLeaves(Keys.join(path, key), child, candidates));
      return;
    }
    String similar = candidates.isEmpty() ? null : Keys.similar(path, candidates);
    problems.add(
        "Unknown " + noun() + " " + path + " at " + node.origin()
            + (similar == null ? "" : " (did you mean " + similar + "?)"));
  }

  private Bound value(Codec codec, Node node, String path, Change change, boolean secret) {
    return switch (codec) {
      case ScalarCodec scalar -> scalar(scalar, node, path, change, secret);
      case RecordCodec record ->
          node instanceof Node.Table table
              ? record(record, table, path, change, null, null)
              : invalid(path, node, "a table");
      case ChoiceCodec choice -> choice(choice, node, path, change);
      case ListCodec list -> list(list, node, path, change);
      case MapCodec map -> map(map, node, path, change);
      case TreeCodec tree -> {
        if (mode == Mode.SETTINGS) {
          throw new IllegalStateException("Setting " + path + " holds a raw tree, which only program files may hold");
        }
        yield !tree.table() || node instanceof Node.Table ? new Bound(node, node) : invalid(path, node, "a table");
      }
    };
  }

  private Bound scalar(ScalarCodec codec, Node node, String path, Change change, boolean secret) {
    Object value =
        switch (codec.kind()) {
          case INT -> integer(node, path, codec.range(), true);
          case LONG -> integer(node, path, codec.range(), false);
          case FLOAT, DOUBLE -> decimal(node, path, codec.range(), codec.kind() == Schema.Scalar.FLOAT);
          case BOOLEAN ->
              node instanceof Node.Bool bool
                  ? bool.value()
                  : fail(path, node, "true or false" + quoted(node, Node.Bool.class));
          case STRING -> string(node, path);
          case PATH -> path(node, path);
          case ENUM -> constant(codec.type(), node, path);
          case INSTANT -> instant(node, path);
        };
    if (value == FAILED) {
      return null;
    }
    leaf(path, change);
    if (secret) {
      return new Bound(value, new Node.Text(MASK, node.origin()));
    }
    boolean text = codec.kind() == Schema.Scalar.STRING || codec.kind() == Schema.Scalar.PATH;
    return new Bound(value, text ? new Node.Text((String) string(node, path), node.origin()) : node);
  }

  /** @param small int なら true、long なら false */
  private Object integer(Node node, String path, Range range, boolean small) {
    long min = small ? Integer.MIN_VALUE : Long.MIN_VALUE;
    long max = small ? Integer.MAX_VALUE : Long.MAX_VALUE;
    String bounds = Schema.bounds(range, true);
    String expected = "an integer" + (bounds.isEmpty() ? "" : " " + inBounds(bounds));
    if (!(node instanceof Node.Numeral numeral) || !numeral.integer()) {
      return fail(path, node, expected + quoted(node, Node.Numeral.class));
    }
    BigInteger value = new BigInteger(numeral.literal());
    if (value.compareTo(BigInteger.valueOf(min)) < 0 || value.compareTo(BigInteger.valueOf(max)) > 0) {
      return fail(path, node, bounds.isEmpty() ? "an integer in [" + min + ", " + max + "]" : expected);
    }
    if (!inRange(range, new BigDecimal(value))) {
      return fail(path, node, expected);
    }
    return small ? (Object) value.intValue() : (Object) value.longValue();
  }

  private Object decimal(Node node, String path, Range range, boolean single) {
    String bounds = Schema.bounds(range, false);
    String expected = "a number" + (bounds.isEmpty() ? "" : " " + inBounds(bounds));
    if (!(node instanceof Node.Numeral numeral)) {
      return fail(path, node, expected + quoted(node, Node.Numeral.class));
    }
    double value = single ? Float.parseFloat(numeral.literal()) : Double.parseDouble(numeral.literal());
    if (!Double.isFinite(value)) {
      return fail(path, node, "a finite number");
    }
    if (!inRange(range, new BigDecimal(value), single)) {
      return fail(path, node, expected);
    }
    return single ? (Object) (float) value : (Object) value;
  }

  /**
   * 数か真偽値を引用符で囲み、文字列として書いた値（JSON の {@code "0.004"} など）に添える注意書き。エラーメッセージは文字列を
   * 引用符なしで書くので、添えないと正しい値に見えてしまう。
   *
   * @param wanted 求めるノードの種類
   * @return 注意書き。文字列を値として読み直しても求める種類にならなければ空
   */
  private static String quoted(Node node, Class<? extends Node> wanted) {
    if (!(node instanceof Node.Text text)) {
      return "";
    }
    try {
      return wanted.isInstance(Toml.value(text.value(), node.origin())) ? " (written as a string: drop the quotes)" : "";
    } catch (SettingsException unreadable) {
      return "";
    }
  }

  private static String inBounds(String bounds) {
    return bounds.startsWith("[") || bounds.startsWith("(") ? "in " + bounds : bounds;
  }

  private static boolean inRange(Range range, BigDecimal value) {
    return inRange(range, value, false);
  }

  /** @param single 値が float なら true。そのときは範囲の端も float に丸めて比べる（0.3 と書いた値が、端の 0.3 を越えたことにならないように） */
  private static boolean inRange(Range range, BigDecimal value, boolean single) {
    if (range == null) {
      return true;
    }
    if (range.min() != Double.NEGATIVE_INFINITY) {
      int low = value.compareTo(new BigDecimal(single ? (float) range.min() : range.min()));
      if (range.minInclusive() ? low < 0 : low <= 0) {
        return false;
      }
    }
    if (range.max() != Double.POSITIVE_INFINITY) {
      int high = value.compareTo(new BigDecimal(single ? (float) range.max() : range.max()));
      return range.maxInclusive() ? high <= 0 : high < 0;
    }
    return true;
  }

  /** 文字列を読む。数と真偽値も、書いたままの文字列として受け付ける（コマンドラインの {@code --room 1234} など）。 */
  private Object string(Node node, String path) {
    return switch (node) {
      case Node.Text text -> text.value();
      case Node.Numeral numeral -> numeral.written();
      case Node.Bool bool -> Boolean.toString(bool.value());
      default -> fail(path, node, "a string");
    };
  }

  private Object path(Node node, String path) {
    Object text = string(node, path);
    if (text == FAILED) {
      return FAILED;
    }
    try {
      return Path.of((String) text);
    } catch (InvalidPathException invalid) {
      return fail(path, node, "a path");
    }
  }

  private Object constant(Class<?> type, Node node, String path) {
    Object[] constants = type.getEnumConstants();
    if (node instanceof Node.Text text) {
      for (Object constant : constants) {
        if (((Enum<?>) constant).name().equals(text.value())) {
          return constant;
        }
      }
    }
    return fail(path, node, "one of " + names(type));
  }

  private static String names(Class<?> type) {
    return Arrays.stream(type.getEnumConstants()).map(constant -> ((Enum<?>) constant).name()).toList().toString();
  }

  private Object instant(Node node, String path) {
    if (node instanceof Node.Text text) {
      try {
        return Instant.parse(text.value());
      } catch (DateTimeParseException invalid) {
        return fail(path, node, "an instant such as 2026-11-12T03:00:00Z");
      }
    }
    return fail(path, node, "an instant such as 2026-11-12T03:00:00Z");
  }

  private Bound choice(ChoiceCodec codec, Node node, String path, Change change) {
    Node.Table table = codec.normalize(node);
    if (table == null) {
      return invalid(path, node, "a table or one of " + codec.names());
    }
    String keyPath = Keys.join(path, codec.key());
    Node kind = table.get(codec.key());
    if (kind == null) {
      problems.add("Missing " + noun() + " " + keyPath);
      return null;
    }
    String name = ChoiceCodec.name(kind);
    RecordCodec chosen = name == null ? null : codec.cases().get(name);
    if (chosen == null) {
      return invalid(keyPath, kind, "one of " + codec.names());
    }
    leaf(keyPath, change);
    Bound bound = record(chosen, table, path, change, codec, name);
    if (bound == null) {
      return null;
    }
    Map<String, Node> effective = new LinkedHashMap<>();
    effective.put(codec.key(), kind);
    effective.putAll(((Node.Table) bound.node()).entries());
    return new Bound(bound.value(), new Node.Table(effective, table.origin()));
  }

  private Bound list(ListCodec codec, Node node, String path, Change change) {
    if (!(node instanceof Node.Array array)) {
      return invalid(path, node, "a list");
    }
    List<Object> values = new ArrayList<>();
    List<Node> effective = new ArrayList<>();
    boolean failed = false;
    insideList++;
    try {
      for (int index = 0; index < array.items().size(); index++) {
        Bound bound = value(codec.element(), array.items().get(index), path + "[" + index + "]", change, false);
        if (bound == null) {
          failed = true;
        } else {
          values.add(bound.value());
          effective.add(bound.node());
        }
      }
    } finally {
      insideList--;
    }
    if (failed) {
      return null;
    }
    leaf(path, change);
    if (!codec.set()) {
      return new Bound(List.copyOf(values), new Node.Array(effective, array.origin()));
    }
    Set<Object> distinct = new LinkedHashSet<>();
    for (int index = 0; index < values.size(); index++) {
      if (!distinct.add(values.get(index))) {
        problems.add(
            "Invalid " + noun() + " " + path + "=" + Keys.text(array) + " at " + array.origin() + ": "
                + Keys.text(array.items().get(index)) + " appears twice");
        return null;
      }
    }
    return new Bound(Collections.unmodifiableSet(distinct), new Node.Array(effective, array.origin()));
  }

  private Bound map(MapCodec codec, Node node, String path, Change change) {
    if (!(node instanceof Node.Table table)) {
      return invalid(path, node, "a table");
    }
    boolean named = codec.keys() == String.class;
    Map<Object, Object> values = named ? new TreeMap<>() : enumMap(codec.keys());
    Map<Object, Node> nodes = named ? new TreeMap<>() : enumMap(codec.keys());
    boolean failed = false;
    for (Map.Entry<String, Node> entry : table.entries().entrySet()) {
      String child = Keys.join(path, entry.getKey());
      Object key = named ? entry.getKey() : enumKey(codec.keys(), entry.getKey());
      if (key == null) {
        problems.add(
            "Unknown " + noun() + " " + child + " at " + entry.getValue().origin() + " (expected a key in "
                + names(codec.keys()) + ")");
        failed = true;
        continue;
      }
      Bound bound = value(codec.value(), entry.getValue(), child, change, false);
      if (bound == null) {
        failed = true;
        continue;
      }
      values.put(key, bound.value());
      nodes.put(key, bound.node());
    }
    if (failed) {
      return null;
    }
    Map<String, Node> effective = new LinkedHashMap<>();
    nodes.forEach((key, value) -> effective.put(named ? (String) key : ((Enum<?>) key).name(), value));
    return new Bound(Collections.unmodifiableMap(values), new Node.Table(effective, table.origin()));
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static <V> Map<Object, V> enumMap(Class<?> keys) {
    return new EnumMap(keys);
  }

  private static Object enumKey(Class<?> keys, String name) {
    for (Object constant : keys.getEnumConstants()) {
      if (((Enum<?>) constant).name().equals(name)) {
        return constant;
      }
    }
    return null;
  }

  /** 葉のキーを変えてよい時期を控える。配列の中の値は、配列という 1 つの葉に含める。 */
  private void leaf(String path, Change change) {
    if (insideList == 0) {
      changes.put(path, change);
    }
  }

  private Object fail(String path, Node node, String expected) {
    problems.add(
        "Invalid " + noun() + " " + path + "=" + Keys.text(node) + " at " + node.origin() + ": expected " + expected);
    return FAILED;
  }

  private Bound invalid(String path, Node node, String expected) {
    fail(path, node, expected);
    return null;
  }

  private String noun() {
    return mode == Mode.SETTINGS ? "setting" : "field";
  }
}
