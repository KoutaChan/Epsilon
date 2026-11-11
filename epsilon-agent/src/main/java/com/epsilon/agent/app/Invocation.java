package com.epsilon.agent.app;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.agent.settings.Toml;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.lang.reflect.Type;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * 読んだコマンドライン。形は {@code <コマンド> [--名前 値 …]}。{@code --settings FILE}、{@code --preset 名前}（くり返せる）、
 * {@code --set パス=値}（くり返せる）は、どこに書いてもよい。値は省けない（真偽値も {@code --replace true} と書く）。
 *
 * @param words {@code --} で始まらない語。コマンドの名前と、{@code help} の後に続くコマンドの名前
 * @param settings {@code --settings} のファイル。無ければ null
 * @param presets {@code --preset} の名前。書いた順
 * @param sets {@code --set} の「パス=値」。書いた順
 * @param arguments コマンドの引数。書いた順
 */
record Invocation(List<String> words, Path settings, List<String> presets, List<String> sets, List<Argument> arguments) {

  /** 伏せた値の代わりに書く文字列。 */
  static final String MASK = "********";

  /** コマンドの引数から作る層の出どころ。 */
  static final String ORIGIN = "command line";

  /** @param name {@code --} を除いた名前 */
  record Argument(String name, String value) {}

  Invocation {
    words = List.copyOf(words);
    presets = List.copyOf(presets);
    sets = List.copyOf(sets);
    arguments = List.copyOf(arguments);
  }

  /**
   * コマンドラインを読む。
   *
   * @throws SettingsException 値の無い {@code --名前} があるか、{@code --settings} が 2 回ある
   */
  static Invocation parse(String[] args) {
    List<String> words = new ArrayList<>();
    List<String> presets = new ArrayList<>();
    List<String> sets = new ArrayList<>();
    List<Argument> arguments = new ArrayList<>();
    List<String> problems = new ArrayList<>();
    Path settings = null;
    for (int index = 0; index < args.length; index++) {
      String token = args[index];
      if (!token.startsWith("--")) {
        words.add(token);
        continue;
      }
      String name = token.substring(2);
      if (index + 1 >= args.length || args[index + 1].startsWith("--")) {
        problems.add("Argument " + token + " needs a value");
        continue;
      }
      String value = args[++index];
      switch (name) {
        case "settings" -> {
          if (settings != null) {
            problems.add("Argument --settings is given twice");
          }
          settings = Path.of(value);
        }
        case "preset" -> presets.add(value);
        case "set" -> sets.add(value);
        default -> arguments.add(new Argument(name, value));
      }
    }
    if (!problems.isEmpty()) {
      throw new SettingsException(problems);
    }
    return new Invocation(words, settings, presets, sets, arguments);
  }

  /** {@code --settings}・{@code --preset}・{@code --set} のどれかを受けたか。 */
  boolean hasLayers() {
    return settings != null || !presets.isEmpty() || !sets.isEmpty();
  }

  /**
   * コマンドの record を作る。引数をコマンドのセクションの層にして、渡した層の上に重ね、record にバインドする。
   *
   * @param match 選んだコマンドと、名前に使った語の数
   * @param layers 下から順に並べた層。引数の層はこの上に重ねる
   * @throws SettingsException 知らない引数、くり返せない引数の 2 回目、余った語、バインドの誤りのどれかがある
   */
  Record command(Catalog.Match match, List<Layer> layers) {
    Catalog.Entry entry = match.entry();
    List<String> problems = new ArrayList<>();
    List<Argument> given = new ArrayList<>(arguments);
    List<String> extra = words.subList(match.words(), words.size());
    if (!extra.isEmpty()) {
      if (entry.type() == Help.class) {
        given.add(new Argument("command", String.join(" ", extra)));
      } else {
        problems.add("Unexpected argument " + String.join(" ", extra) + " for " + entry.name());
      }
    }
    Map<String, Node> values = new LinkedHashMap<>();
    Map<String, List<Node>> repeated = new LinkedHashMap<>();
    for (Argument argument : given) {
      RecordComponent component = entry.arguments().get(argument.name());
      if (component == null) {
        problems.add(
            "Unknown argument --"
                + argument.name()
                + " for "
                + entry.name()
                + Catalog.suggestion(argument.name(), entry.arguments().keySet(), " (did you mean --", "?)"));
        continue;
      }
      String origin = "--" + argument.name();
      Type element = element(component);
      try {
        if (element != null) {
          repeated.computeIfAbsent(component.getName(), name -> new ArrayList<>()).add(node(element, argument.value(), origin));
        } else if (values.putIfAbsent(component.getName(), node(component.getGenericType(), argument.value(), origin))
            != null) {
          problems.add("Argument --" + argument.name() + " is given twice");
        }
      } catch (SettingsException unreadable) {
        problems.addAll(unreadable.problems());
      }
    }
    repeated.forEach((name, items) -> values.put(name, new Node.Array(items, "--" + Catalog.argument(name))));
    if (!problems.isEmpty()) {
      throw new SettingsException(problems);
    }
    Node.Table section = new Node.Table(values, ORIGIN);
    List<String> path = entry.words();
    for (int index = path.size() - 1; index >= 0; index--) {
      section = new Node.Table(Map.of(path.get(index), section), ORIGIN);
    }
    List<Layer> stacked = new ArrayList<>(layers);
    stacked.add(new Layer(ORIGIN, Layer.Source.COMMAND_LINE, section, List.of()));
    try {
      return Settings.section(entry.type(), entry.section(), stacked).value();
    } catch (SettingsException invalid) {
      throw new SettingsException(arguments(entry, invalid.problems()));
    }
  }

  /**
   * 始めのログの行に書くコマンドライン。秘密の引数と、秘密のキーへの {@code --set} の値は伏せる。
   *
   * @param secretSets 秘密のキーのパスに合う正規表現（{@link #secretPatterns}）
   */
  String describe(Catalog.Match match, Set<String> secretSets) {
    Catalog.Entry entry = match.entry();
    StringBuilder out = new StringBuilder(String.join(" ", words));
    for (Argument argument : arguments) {
      RecordComponent component = entry.arguments().get(argument.name());
      boolean secret = component != null && component.getAnnotation(Key.class).secret();
      out.append(" --").append(argument.name()).append(' ').append(secret ? MASK : argument.value());
    }
    if (settings != null) {
      out.append(" --settings ").append(settings);
    }
    presets.forEach(preset -> out.append(" --preset ").append(preset));
    for (String set : sets) {
      int equals = set.indexOf('=');
      String key = equals < 0 ? set : set.substring(0, equals).strip();
      out.append(" --set ").append(secretSets.stream().anyMatch(key::matches) ? key + "=" + MASK : set);
    }
    return out.toString();
  }

  /** 全体の設定と、選んだコマンドのセクションのうち、秘密のキーのパスに合う正規表現を返す。名前で引く表では、パスの {@code <name>} が点を含まないキー 1 つに合う。 */
  static Set<String> secretPatterns(Class<? extends Record> schema, Catalog.Entry entry) {
    return Stream.concat(secretPaths(schema, ""), secretPaths(entry.type(), entry.section() + "."))
        .map(path -> Pattern.quote(path).replace("<name>", "\\E[^.]+\\Q"))
        .collect(Collectors.toUnmodifiableSet());
  }

  /** record の秘密のキーのパスを、セクションのパスを先頭に付けて返す。 */
  private static Stream<String> secretPaths(Class<? extends Record> type, String prefix) {
    return Settings.describe(type).stream().filter(info -> info.secret()).map(info -> prefix + info.path());
  }

  /** List か Set のコンポーネントなら、要素の型を返す。ほかのコンポーネントでは null。 */
  private static Type element(RecordComponent component) {
    if (component.getGenericType() instanceof ParameterizedType parameterized) {
      Type raw = parameterized.getRawType();
      if (raw == List.class || raw == Set.class) {
        return parameterized.getActualTypeArguments()[0];
      }
    }
    return null;
  }

  /** 文字列とパスの値はそのまま使い、ほかの型の値は TOML の値として読む。 */
  private static Node node(Type type, String value, String origin) {
    if (type == String.class || type == Path.class) {
      return new Node.Text(value, origin);
    }
    return Toml.value(value, origin);
  }

  /**
   * バインドのエラーメッセージで、コマンドのセクションのキーのパスを引数の名前に直す（{@code Missing setting train.run} なら
   * {@code Missing argument --run}）。
   */
  private static List<String> arguments(Catalog.Entry entry, List<String> problems) {
    Pattern key = Pattern.compile("(Missing|Invalid) setting " + Pattern.quote(entry.section() + ".") + "(\\w+)");
    List<String> rewritten = new ArrayList<>(problems.size());
    for (String problem : problems) {
      Matcher matcher = key.matcher(problem);
      rewritten.add(
          matcher.lookingAt()
              ? matcher.group(1) + " argument --" + Catalog.argument(matcher.group(2)) + problem.substring(matcher.end())
              : problem);
    }
    return rewritten;
  }
}
