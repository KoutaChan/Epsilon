package com.epsilon.agent.settings;

import java.lang.reflect.AnnotatedParameterizedType;
import java.lang.reflect.AnnotatedType;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.lang.reflect.Type;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * 設定と記録の record の型を読み、結び付けと書き出しに使える形にする。型ごとに 1 回だけ作り、作るときに全ての既定値を確かめる。
 */
final class Schema {

  private static final Map<Class<?>, RecordCodec> RECORDS = new ConcurrentHashMap<>();
  private static final Map<Class<?>, ChoiceCodec> CHOICES = new ConcurrentHashMap<>();
  private static final ThreadLocal<Set<Class<?>>> BUILDING = ThreadLocal.withInitial(HashSet::new);

  private Schema() {}

  /** 1 つの型の値を、どう読みどう書くか。 */
  sealed interface Codec permits ScalarCodec, RecordCodec, ChoiceCodec, ListCodec, MapCodec, TreeCodec {}

  /** 単一の値の型。 */
  enum Scalar {
    INT,
    LONG,
    FLOAT,
    DOUBLE,
    BOOLEAN,
    STRING,
    PATH,
    ENUM,
    INSTANT;

    boolean number() {
      return this == INT || this == LONG || this == FLOAT || this == DOUBLE;
    }

    boolean integral() {
      return this == INT || this == LONG;
    }

    /** @return 単一の値の型でなければ null（enum かどうかは呼び出し側が見る） */
    static Scalar of(Class<?> type) {
      if (type == int.class || type == Integer.class) {
        return INT;
      }
      if (type == long.class || type == Long.class) {
        return LONG;
      }
      if (type == float.class || type == Float.class) {
        return FLOAT;
      }
      if (type == double.class || type == Double.class) {
        return DOUBLE;
      }
      if (type == boolean.class || type == Boolean.class) {
        return BOOLEAN;
      }
      if (type == String.class) {
        return STRING;
      }
      if (type == Path.class) {
        return PATH;
      }
      if (type == Instant.class) {
        return INSTANT;
      }
      return null;
    }
  }

  /** @param range 数の範囲。無ければ null */
  record ScalarCodec(Scalar kind, Class<?> type, Range range) implements Codec {}

  /**
   * record のコンポーネント 1 つ。
   *
   * @param key {@link Key} の注釈。プログラムが書くファイルのコンポーネントでは null のことがある
   * @param fallback キーが無いときに使うノード。既定値のリテラルを読んだものか、既定値の無いセクション・テーブル・配列なら空のもの。
   *     どちらでもなければ null
   */
  record Component(String name, Codec codec, Key key, Node fallback, Method accessor) {}

  record RecordCodec(
      Class<?> type, List<Component> components, Map<String, Component> named, Constructor<?> constructor)
      implements Codec {

    /** @return その名前のコンポーネント。無ければ null */
    Component component(String name) {
      return named.get(name);
    }
  }

  /**
   * @param doc 種類のキーの説明（{@link Choice#doc}）
   * @param cases 種類のキーの値から、その種類の record への対応。並びは permits に書いたクラスの順
   */
  record ChoiceCodec(Class<?> type, String key, String doc, Map<String, RecordCodec> cases) implements Codec {

    /** 値だけで書いた種類を、種類のキーだけを持つテーブルに直す。テーブルでも文字列・真偽値でもなければ null。 */
    Node.Table normalize(Node node) {
      return switch (node) {
        case null -> null;
        case Node.Table table -> table;
        case Node.Text text -> new Node.Table(Map.of(key, text), text.origin());
        case Node.Bool bool -> new Node.Table(Map.of(key, bool), bool.origin());
        default -> null;
      };
    }

    /** @return 種類の名前（種類のキーの値を文字列にしたもの）。単一の値でなければ null */
    static String name(Node kind) {
      return switch (kind) {
        case null -> null;
        case Node.Text text -> text.value();
        case Node.Bool bool -> Boolean.toString(bool.value());
        case Node.Numeral numeral -> numeral.literal();
        default -> null;
      };
    }

    String names() {
      return cases.keySet().toString();
    }

    String caseName(Object value) {
      for (Map.Entry<String, RecordCodec> entry : cases.entrySet()) {
        if (entry.getValue().type() == value.getClass()) {
          return entry.getKey();
        }
      }
      throw new IllegalArgumentException("Not a case of " + type.getName() + ": " + value.getClass().getName());
    }
  }

  /** @param set 要素が重ならない配列（{@code Set}）か */
  record ListCodec(Codec element, boolean set) implements Codec {}

  /** @param keys キーの型（String か enum） */
  record MapCodec(Class<?> keys, Codec value) implements Codec {}

  /**
   * 設定の木のノードをそのまま持つ（設定の行が持つ設定、変えたキーの変更前後の値）。使うのはプログラムが書くファイルの record だけ。
   *
   * @param table テーブルに限るか（{@link Node.Table} のコンポーネント）
   */
  record TreeCodec(boolean table) implements Codec {}

  /**
   * record の型を読んだもの。同じ型には同じものを返す。
   *
   * @throws IllegalStateException 型が設定に使えないか、注釈か既定値が誤っている
   */
  static RecordCodec record(Class<?> type) {
    RecordCodec cached = RECORDS.get(type);
    if (cached != null) {
      return cached;
    }
    if (!type.isRecord()) {
      throw new IllegalStateException("Settings bind to records: " + type.getName());
    }
    Set<Class<?>> building = BUILDING.get();
    if (!building.add(type)) {
      throw new IllegalStateException("Recursive settings type " + type.getName());
    }
    try {
      RecordCodec built = build(type);
      RecordCodec previous = RECORDS.putIfAbsent(type, built);
      return previous != null ? previous : built;
    } finally {
      building.remove(type);
    }
  }

  private static RecordCodec build(Class<?> type) {
    RecordComponent[] declared = type.getRecordComponents();
    List<Component> components = new ArrayList<>();
    Map<String, Component> named = new LinkedHashMap<>();
    Class<?>[] parameters = new Class<?>[declared.length];
    for (int index = 0; index < declared.length; index++) {
      RecordComponent component = declared[index];
      String where = type.getSimpleName() + "." + component.getName();
      Key key = component.getAnnotation(Key.class);
      Codec codec = codec(component.getAnnotatedType(), where);
      Method accessor = component.getAccessor();
      accessor.setAccessible(true);
      Node fallback = fallback(key, codec, component.getType(), where);
      Component built = new Component(component.getName(), codec, key, fallback, accessor);
      components.add(built);
      named.put(built.name(), built);
      parameters[index] = component.getType();
    }
    Constructor<?> constructor;
    try {
      constructor = type.getDeclaredConstructor(parameters);
      constructor.setAccessible(true);
    } catch (NoSuchMethodException missing) {
      throw new IllegalStateException("No canonical constructor in " + type.getName(), missing);
    }
    RecordCodec codec =
        new RecordCodec(type, List.copyOf(components), Collections.unmodifiableMap(named), constructor);
    for (Component component : codec.components()) {
      if (component.key() != null && !component.key().value().isEmpty()) {
        Binder check = new Binder(Binder.Mode.RECORD, Set::of);
        if (check.value(component.codec(), component.fallback(), component.name()) == null) {
          throw new IllegalStateException(
              "Invalid default of " + type.getSimpleName() + "." + component.name() + ": "
                  + String.join("; ", check.problems()));
        }
      }
    }
    return codec;
  }

  private static Node fallback(Key key, Codec codec, Class<?> type, String where) {
    if (key == null) {
      return null;
    }
    if (key.presence() == Presence.ABSENT && type.isPrimitive()) {
      throw new IllegalStateException("Setting " + where + " may be absent, so it needs a reference type");
    }
    if (key.secret() && !(codec instanceof ScalarCodec scalar && scalar.kind() == Scalar.STRING)) {
      throw new IllegalStateException("Secret setting " + where + " must be a string");
    }
    if (!key.value().isEmpty()) {
      if (key.presence() != Presence.DEFAULT) {
        throw new IllegalStateException("Setting " + where + " has a default but presence " + key.presence());
      }
      try {
        return Toml.value(key.value(), Binder.DEFAULT_ORIGIN);
      } catch (SettingsException unreadable) {
        throw new IllegalStateException("Unreadable default of " + where + ": " + unreadable.getMessage());
      }
    }
    if (key.presence() != Presence.DEFAULT) {
      return null;
    }
    return switch (codec) {
      case RecordCodec record -> new Node.Table(Map.of(), Binder.DEFAULT_ORIGIN);
      case MapCodec map -> new Node.Table(Map.of(), Binder.DEFAULT_ORIGIN);
      case ListCodec list -> new Node.Array(List.of(), Binder.DEFAULT_ORIGIN);
      default ->
          throw new IllegalStateException(
              "Setting " + where + " needs a default, or presence REQUIRED or ABSENT");
    };
  }

  private static Codec codec(AnnotatedType annotated, String where) {
    Type type = annotated.getType();
    Codec codec;
    if (type instanceof Class<?> plain) {
      codec = plain(plain, where);
    } else if (type instanceof ParameterizedType parameterized
        && annotated instanceof AnnotatedParameterizedType arguments) {
      codec = parameterized(parameterized, arguments.getAnnotatedActualTypeArguments(), where);
    } else {
      throw new IllegalStateException("Unsupported settings type " + type.getTypeName() + " at " + where);
    }
    Range range = annotated.getAnnotation(Range.class);
    if (range == null) {
      return codec;
    }
    if (!(codec instanceof ScalarCodec scalar) || !scalar.kind().number()) {
      throw new IllegalStateException("@Range on a value that is not a number at " + where);
    }
    if (!(range.min() <= range.max())) {
      throw new IllegalStateException("Empty @Range at " + where);
    }
    return new ScalarCodec(scalar.kind(), scalar.type(), range);
  }

  private static Codec plain(Class<?> type, String where) {
    Scalar scalar = Scalar.of(type);
    if (scalar != null) {
      return new ScalarCodec(scalar, type, null);
    }
    if (type.isEnum()) {
      return new ScalarCodec(Scalar.ENUM, type, null);
    }
    if (type == Node.Table.class || type == Node.class) {
      return new TreeCodec(type == Node.Table.class);
    }
    if (type.isRecord()) {
      return record(type);
    }
    if (type.isInterface() && type.isSealed() && type.isAnnotationPresent(Choice.class)) {
      return choice(type);
    }
    throw new IllegalStateException("Unsupported settings type " + type.getName() + " at " + where);
  }

  private static Codec parameterized(ParameterizedType type, AnnotatedType[] arguments, String where) {
    Type raw = type.getRawType();
    if (raw == List.class || raw == Set.class) {
      return new ListCodec(codec(arguments[0], where + "[]"), raw == Set.class);
    }
    if (raw == Map.class) {
      Type keys = arguments[0].getType();
      if (keys != String.class && !(keys instanceof Class<?> constants && constants.isEnum())) {
        throw new IllegalStateException("Table keys must be strings or enums at " + where);
      }
      return new MapCodec((Class<?>) keys, codec(arguments[1], where + ".<key>"));
    }
    throw new IllegalStateException("Unsupported settings type " + type.getTypeName() + " at " + where);
  }

  private static ChoiceCodec choice(Class<?> type) {
    ChoiceCodec cached = CHOICES.get(type);
    if (cached != null) {
      return cached;
    }
    Choice annotation = type.getAnnotation(Choice.class);
    String key = annotation.key();
    Map<String, RecordCodec> cases = new LinkedHashMap<>();
    for (Class<?> permitted : type.getPermittedSubclasses()) {
      Case value = permitted.getAnnotation(Case.class);
      if (!permitted.isRecord() || value == null) {
        throw new IllegalStateException(
            "Every case of " + type.getName() + " must be a record with @Case: " + permitted.getName());
      }
      RecordCodec codec = record(permitted);
      if (codec.component(key) != null) {
        throw new IllegalStateException("The case " + permitted.getName() + " has a component named " + key);
      }
      if (cases.put(value.value(), codec) != null) {
        throw new IllegalStateException("Duplicate case " + value.value() + " in " + type.getName());
      }
    }
    ChoiceCodec built = new ChoiceCodec(type, key, annotation.doc(), Collections.unmodifiableMap(cases));
    ChoiceCodec previous = CHOICES.putIfAbsent(type, built);
    return previous != null ? previous : built;
  }

  /** セクションに分けず、キーと同じ行に書く型か（単一の値、配列、木のノード、それらを値に持つテーブル）。 */
  static boolean inline(Codec codec) {
    return switch (codec) {
      case ScalarCodec scalar -> true;
      case ListCodec list -> true;
      case TreeCodec tree -> true;
      case MapCodec map -> inline(map.value());
      case RecordCodec record -> false;
      case ChoiceCodec choice -> false;
    };
  }

  /** 値を木にする。null のコンポーネントは書かない。 */
  static Node node(Codec codec, Object value, String origin) {
    return switch (codec) {
      case ScalarCodec scalar -> scalar(scalar, value, origin);
      case RecordCodec record -> {
        Map<String, Node> entries = new LinkedHashMap<>();
        for (Component component : record.components()) {
          Object part = read(component, value);
          if (part != null) {
            entries.put(component.name(), node(component.codec(), part, origin));
          }
        }
        yield new Node.Table(entries, origin);
      }
      case ChoiceCodec choice -> {
        String name = choice.caseName(value);
        Map<String, Node> entries = new LinkedHashMap<>();
        entries.put(
            choice.key(),
            name.equals("true") || name.equals("false")
                ? new Node.Bool(Boolean.parseBoolean(name), origin)
                : new Node.Text(name, origin));
        entries.putAll(((Node.Table) node(choice.cases().get(name), value, origin)).entries());
        yield new Node.Table(entries, origin);
      }
      case ListCodec list ->
          new Node.Array(
              ((Collection<?>) value).stream().map(item -> node(list.element(), item, origin)).toList(), origin);
      case MapCodec map -> {
        Map<String, Node> entries = new LinkedHashMap<>();
        ((Map<?, ?>) value)
            .forEach(
                (key, item) ->
                    entries.put(
                        key instanceof Enum<?> constant ? constant.name() : (String) key,
                        node(map.value(), item, origin)));
        yield new Node.Table(entries, origin);
      }
      case TreeCodec tree -> (Node) value;
    };
  }

  private static Node scalar(ScalarCodec codec, Object value, String origin) {
    return switch (codec.kind()) {
      case INT, LONG -> new Node.Numeral(value.toString(), origin);
      case FLOAT, DOUBLE -> {
        double number = ((Number) value).doubleValue();
        if (!Double.isFinite(number)) {
          throw new IllegalArgumentException("Cannot write a number that is not finite: " + value);
        }
        yield new Node.Numeral(value.toString(), origin);
      }
      case BOOLEAN -> new Node.Bool((Boolean) value, origin);
      case STRING, PATH, INSTANT -> new Node.Text(value.toString(), origin);
      case ENUM -> new Node.Text(((Enum<?>) value).name(), origin);
    };
  }

  private static Object read(Component component, Object record) {
    try {
      return component.accessor().invoke(record);
    } catch (InvocationTargetException failure) {
      throw new IllegalStateException("Cannot read " + component.name(), failure.getCause());
    } catch (IllegalAccessException failure) {
      throw new IllegalStateException("Cannot read " + component.name(), failure);
    }
  }

  /**
   * 選べる値の一覧（ひな形と効いている設定の表示の注）。enum と種類のキーは値を「・」でつなぎ、enum の重ならない列には「から重ならずに」を
   * 添える。
   *
   * @return 選べる値の決まっていない型なら空
   */
  static String choices(Codec codec) {
    return switch (codec) {
      case ScalarCodec scalar when scalar.kind() == Scalar.ENUM ->
          Arrays.stream(scalar.type().getEnumConstants())
              .map(constant -> ((Enum<?>) constant).name())
              .collect(Collectors.joining("・"));
      case ChoiceCodec choice -> String.join("・", choice.cases().keySet());
      case ListCodec list -> {
        String element = choices(list.element());
        yield element.isEmpty() || !list.set() ? element : element + " から重ならずに";
      }
      default -> "";
    };
  }

  /** 値の型の説明（英語）。使い方の表示に出す。 */
  static String helpLabel(Codec codec) {
    return switch (codec) {
      case ScalarCodec scalar ->
          switch (scalar.kind()) {
            case INT, LONG -> "integer";
            case FLOAT, DOUBLE -> "number";
            case BOOLEAN -> "true|false";
            case STRING -> "string";
            case PATH -> "path";
            case INSTANT -> "time";
            case ENUM ->
                Arrays.stream(scalar.type().getEnumConstants())
                    .map(constant -> ((Enum<?>) constant).name())
                    .collect(Collectors.joining("|"));
          };
      case RecordCodec record -> "table";
      case ChoiceCodec choice -> String.join("|", choice.cases().keySet());
      case ListCodec list -> (list.set() ? "distinct " : "") + helpLabel(list.element()) + " list";
      case MapCodec map ->
          helpLabel(map.value()) + " per " + (map.keys() == String.class ? "name" : map.keys().getSimpleName());
      case TreeCodec tree -> tree.table() ? "table" : "value";
    };
  }

  /**
   * 数の範囲を文字列にする（{@code [1, 4]}、{@code (0.0, 1.0]}、{@code >= 1}）。
   *
   * @param integral 整数の範囲か。そうなら端を整数で書く
   * @return 範囲が無ければ空
   */
  static String bounds(Range range, boolean integral) {
    if (range == null) {
      return "";
    }
    boolean low = range.min() != Double.NEGATIVE_INFINITY;
    boolean high = range.max() != Double.POSITIVE_INFINITY;
    String min = bound(range.min(), integral);
    String max = bound(range.max(), integral);
    if (low && high) {
      return (range.minInclusive() ? "[" : "(") + min + ", " + max + (range.maxInclusive() ? "]" : ")");
    }
    if (low) {
      return (range.minInclusive() ? ">= " : "> ") + min;
    }
    if (high) {
      return (range.maxInclusive() ? "<= " : "< ") + max;
    }
    return "";
  }

  private static String bound(double value, boolean integral) {
    return integral && value == Math.rint(value) && Math.abs(value) < 1e18
        ? Long.toString((long) value)
        : Double.toString(value);
  }

  /**
   * 全体の record の全てのキーのパス。名前をキーにしたテーブルの中は含めず、種類を選ぶセクションは全ての種類のキーを含める。
   * 知らないキーに近いキーを探すのに使う。
   */
  static Set<String> paths(RecordCodec root, String prefix) {
    Set<String> paths = new HashSet<>();
    collect(root, prefix, paths);
    return paths;
  }

  private static void collect(RecordCodec codec, String prefix, Set<String> paths) {
    for (Component component : codec.components()) {
      String path = Keys.join(prefix, component.name());
      paths.add(path);
      switch (component.codec()) {
        case RecordCodec record -> collect(record, path, paths);
        case ChoiceCodec choice -> {
          paths.add(Keys.join(path, choice.key()));
          choice.cases().values().forEach(chosen -> collect(chosen, path, paths));
        }
        default -> {}
      }
    }
  }

  /**
   * 秘密のキーのパス。たどるのはセクションと種類を選ぶセクションだけ（秘密のキーは配列と名前をキーにしたテーブルの中に置けない）。
   *
   * @throws IllegalStateException 配列か名前をキーにしたテーブルの中に秘密のキーがある
   */
  static List<List<String>> secrets(RecordCodec root) {
    List<List<String>> secrets = new ArrayList<>();
    secrets(root, List.of(), secrets, false);
    return secrets;
  }

  private static void secrets(Codec codec, List<String> path, List<List<String>> out, boolean collection) {
    switch (codec) {
      case RecordCodec record -> {
        for (Component component : record.components()) {
          List<String> child = new ArrayList<>(path);
          child.add(component.name());
          if (component.key() != null && component.key().secret()) {
            if (collection) {
              throw new IllegalStateException("Secret setting " + String.join(".", child) + " sits in a list or table");
            }
            out.add(List.copyOf(child));
          }
          secrets(component.codec(), child, out, collection);
        }
      }
      case ChoiceCodec choice -> choice.cases().values().forEach(chosen -> secrets(chosen, path, out, collection));
      case ListCodec list -> secrets(list.element(), path, out, true);
      case MapCodec map -> secrets(map.value(), path, out, true);
      default -> {}
    }
  }

  /**
   * セクションをたどった先のコンポーネント。
   *
   * @throws IllegalStateException パスが全体の record のセクションをたどれない
   */
  static Component component(RecordCodec root, List<String> keys) {
    RecordCodec codec = root;
    Component component = null;
    for (int index = 0; index < keys.size(); index++) {
      if (codec == null) {
        throw new IllegalStateException("Not a section: " + String.join(".", keys.subList(0, index)));
      }
      component = codec.component(keys.get(index));
      if (component == null) {
        throw new IllegalStateException("Unknown setting " + String.join(".", keys) + " in " + root.type().getName());
      }
      codec = component.codec() instanceof RecordCodec record ? record : null;
    }
    if (component == null) {
      throw new IllegalStateException("Empty setting path");
    }
    return component;
  }
}
