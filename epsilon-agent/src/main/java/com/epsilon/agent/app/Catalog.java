package com.epsilon.agent.app;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Settings;
import java.lang.reflect.RecordComponent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/** アプリのコマンドの一覧。sealed インターフェースが許す record を、インターフェースを並べた順に並べる。 */
final class Catalog {

  private static final Pattern WORD = Pattern.compile("[a-z][a-z0-9]*(-[a-z0-9]+)*");
  private static final Set<String> OPTIONS = Set.of("settings", "preset", "set");

  /**
   * 1 つのコマンド。
   *
   * @param arguments 引数の名前（{@code stop-after} など）から record のコンポーネントへの対応。コンポーネントの順に並ぶ
   */
  record Entry(String name, Named named, Class<? extends Record> type, Map<String, RecordComponent> arguments) {

    /** 設定ファイルの中で、このコマンドのセクションを指すキーのパス（{@code settings.change} など）。 */
    String section() {
      return name.replace(' ', '.');
    }

    List<String> words() {
      return List.of(name.split(" "));
    }
  }

  /** @param words コマンドの名前に使った語の数 */
  record Match(Entry entry, int words) {}

  private final Map<String, Entry> entries;
  private final Set<String> roots;
  private final Map<String, Set<String>> groups;

  private Catalog(Map<String, Entry> entries) {
    this.entries = entries;
    this.roots = new LinkedHashSet<>();
    this.groups = new LinkedHashMap<>();
    for (Entry entry : entries.values()) {
      List<String> words = entry.words();
      roots.add(words.getFirst());
      if (words.size() == 2) {
        groups.computeIfAbsent(words.getFirst(), first -> new LinkedHashSet<>()).add(words.get(1));
      }
    }
  }

  /**
   * 組み込みのコマンドとアプリのコマンドの表を作る。
   *
   * @throws IllegalStateException インターフェースか record がコマンドとして使えない
   */
  static Catalog of(Application<?> application) {
    Map<String, Entry> entries = new LinkedHashMap<>();
    collect(BuiltinCommand.class, entries);
    for (Class<?> commands : application.commands()) {
      collect(commands, entries);
    }
    Set<String> sections = new LinkedHashSet<>();
    for (RecordComponent component : application.schema().getRecordComponents()) {
      sections.add(component.getName());
    }
    Catalog catalog = new Catalog(entries);
    for (Entry entry : entries.values()) {
      String first = entry.words().getFirst();
      if (sections.contains(first)) {
        throw new IllegalStateException(
            "The command " + entry.name() + " shares its first word with the settings section " + first);
      }
      if (entry.words().size() == 1 && catalog.groups.containsKey(first)) {
        throw new IllegalStateException("The command " + first + " is also the first word of other commands");
      }
    }
    return catalog;
  }

  Collection<Entry> entries() {
    return entries.values();
  }

  /** @return その名前のコマンド。無ければ null */
  Entry named(String name) {
    return entries.get(name);
  }

  /** 名前がいちばん長く合うコマンドを返す。合うものが無ければ null。 */
  Match find(List<String> words) {
    if (words.size() >= 2) {
      Entry two = entries.get(words.get(0) + " " + words.get(1));
      if (two != null) {
        return new Match(two, 2);
      }
    }
    Entry one = words.isEmpty() ? null : entries.get(words.getFirst());
    return one == null ? null : new Match(one, 1);
  }

  /** 設定ファイルで、コマンドのセクションが使ういちばん外側のキー。全体の設定をバインドする前に、層から除く。 */
  Set<String> roots() {
    return roots;
  }

  /** コマンドのセクションを除いた層を返す。 */
  Layer withoutSections(Layer layer) {
    Layer stripped = layer;
    for (String root : roots) {
      stripped = stripped.without(root);
    }
    return stripped;
  }

  /** 2 語のコマンドで 1 語目に当たるセクション（{@code [settings]} など）を見て、その下にある知らないコマンドのセクションを問題として返す。 */
  List<String> unknownSections(Layer layer) {
    List<String> problems = new ArrayList<>();
    groups.forEach(
        (first, seconds) -> {
          Node node = layer.tree().get(first);
          if (node instanceof Node.Table table) {
            for (Map.Entry<String, Node> child : table.entries().entrySet()) {
              if (!seconds.contains(child.getKey())) {
                problems.add(
                    "Unknown setting "
                        + first
                        + "."
                        + child.getKey()
                        + " at "
                        + child.getValue().origin()
                        + suggestion(first + "." + child.getKey(), sections(), " (did you mean [", "]?)"));
              }
            }
          } else if (node != null) {
            problems.add("Invalid setting " + first + " at " + node.origin() + ": expected a table of commands");
          }
        });
    return problems;
  }

  private List<String> sections() {
    return entries.values().stream().map(Entry::section).toList();
  }

  /** 近い名前を添える文を返す。近い名前が無ければ空文字列。 */
  static String suggestion(String unknown, Collection<String> known, String before, String after) {
    String best = null;
    int bestDistance = Math.max(2, unknown.length() / 3) + 1;
    for (String candidate : known) {
      int distance = distance(unknown, candidate);
      if (distance < bestDistance) {
        best = candidate;
        bestDistance = distance;
      }
    }
    return best == null ? "" : before + best + after;
  }

  /** record のコンポーネントの名前を引数の名前にする（{@code stopAfter} なら {@code stop-after}）。 */
  static String argument(String component) {
    StringBuilder out = new StringBuilder(component.length() + 4);
    for (int index = 0; index < component.length(); index++) {
      char c = component.charAt(index);
      if (Character.isUpperCase(c)) {
        out.append('-').append(Character.toLowerCase(c));
      } else {
        out.append(c);
      }
    }
    return out.toString();
  }

  private static void collect(Class<?> type, Map<String, Entry> entries) {
    if (type.isRecord()) {
      Entry entry = entry(type);
      if (entries.putIfAbsent(entry.name(), entry) != null) {
        throw new IllegalStateException("Two commands are named " + entry.name());
      }
      return;
    }
    if (!type.isInterface() || !type.isSealed() || !Command.class.isAssignableFrom(type)) {
      throw new IllegalStateException("Commands are records under a sealed command interface: " + type.getName());
    }
    for (Class<?> permitted : type.getPermittedSubclasses()) {
      collect(permitted, entries);
    }
  }

  private static Entry entry(Class<?> type) {
    Named named = type.getAnnotation(Named.class);
    if (named == null) {
      throw new IllegalStateException("The command " + type.getName() + " has no @Named");
    }
    String[] words = named.value().split(" ", -1);
    if (words.length > 2 || !Arrays.stream(words).allMatch(word -> WORD.matcher(word).matches())) {
      throw new IllegalStateException("Invalid command name '" + named.value() + "' on " + type.getName());
    }
    @SuppressWarnings("unchecked")
    Class<? extends Record> record = (Class<? extends Record>) type;
    Map<String, RecordComponent> arguments = new LinkedHashMap<>();
    for (RecordComponent component : record.getRecordComponents()) {
      if (component.getAnnotation(Key.class) == null) {
        throw new IllegalStateException("The argument " + type.getSimpleName() + "." + component.getName() + " has no @Key");
      }
      String argument = argument(component.getName());
      if (OPTIONS.contains(argument) || arguments.putIfAbsent(argument, component) != null) {
        throw new IllegalStateException("The argument --" + argument + " of " + named.value() + " is taken");
      }
    }
    Settings.describe(record);
    return new Entry(named.value(), named, record, arguments);
  }

  private static int distance(String a, String b) {
    int[] previous = new int[b.length() + 1];
    int[] current = new int[b.length() + 1];
    for (int j = 0; j <= b.length(); j++) {
      previous[j] = j;
    }
    for (int i = 1; i <= a.length(); i++) {
      current[0] = i;
      for (int j = 1; j <= b.length(); j++) {
        int replace = previous[j - 1] + (a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1);
        current[j] = Math.min(replace, Math.min(previous[j], current[j - 1]) + 1);
      }
      int[] swap = previous;
      previous = current;
      current = swap;
    }
    return previous[b.length()];
  }
}
