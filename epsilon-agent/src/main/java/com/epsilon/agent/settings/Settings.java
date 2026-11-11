package com.epsilon.agent.settings;

import com.epsilon.agent.settings.Schema.ChoiceCodec;
import com.epsilon.agent.settings.Schema.Component;
import com.epsilon.agent.settings.Schema.MapCodec;
import com.epsilon.agent.settings.Schema.RecordCodec;
import com.epsilon.agent.settings.Schema.ScalarCodec;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Function;
import java.util.function.UnaryOperator;

/**
 * 層を下から重ねて作った全体の record と、効いているキーの木。
 *
 * <p>重ね方は 1 通りで、下から既定値、四人打ち・三人打ちごとの既定値、プリセット、設定ファイル、{@code --set}、コマンドの引数の順に重ねる。
 * テーブルはキーごとに重ね、配列と値は上の層のもので置き換える。種類を選ぶセクションは、上の層が種類のキーを違う値にしたら丸ごと置き換える。
 * 秘密のキーは設定ファイルの層からだけ受け付ける。
 *
 * <p>エラーは全て集め、1 行に 1 つずつ並べた {@link SettingsException} にまとめる。集めるのは、型、範囲、enum の名前、必須のキーの欠け、
 * 知らないキー（近いキーを添える）、選んだ種類に無いキー、record の検査のエラー。どの行にも、キーのパス・値・期待した値・書かれた場所を書く。
 *
 * @param <R> 全体の record
 */
public final class Settings<R extends Record> {

  /** 既定値と同じ値を探して結び付け直すときの層の出どころ。結び付け直した木の書かれた場所は使わない。 */
  private static final String ROW = "settings row";

  private final R value;
  private final Node.Table tree;
  private final Map<String, Change> changes;

  private Settings(R value, Node.Table tree, Map<String, Change> changes) {
    this.value = value;
    this.tree = tree;
    this.changes = Collections.unmodifiableMap(new HashMap<>(changes));
  }

  /**
   * 層を重ねて全体の record を作る。
   *
   * @param root 全体の record の型。いちばん外のキーには、全て変えてよい時期を書いておく
   * @param layers 下から順の層
   * @throws SettingsException 全てのエラーを集めたもの
   * @throws IllegalStateException record の型か注釈が設定に使えない
   */
  public static <R extends Record> Settings<R> bind(Class<R> root, List<Layer> layers) {
    RecordCodec codec = Schema.record(root);
    List<String> problems = new ArrayList<>();
    Node.Table merged = Merge.layers(codec, layers, "", problems);
    Binder binder = new Binder(Binder.Mode.SETTINGS, () -> Schema.paths(codec, ""));
    binder.problems().addAll(problems);
    Binder.Bound bound = binder.record(codec, merged, "", null);
    if (!binder.problems().isEmpty()) {
      throw new SettingsException(binder.problems());
    }
    return new Settings<>(root.cast(bound.value()), (Node.Table) bound.node(), binder.changes());
  }

  /**
   * いちばん下の層（四人打ち・三人打ちごとの既定値）を選んでから、層を重ねて全体の record を作る。下の層は、上の層だけを重ねた生の木で
   * 1 つのキーの値を見て選ぶ。
   *
   * @throws SettingsException 層を選ぶキーが無いか読めない、層を作れない、または record に結び付けられない
   */
  public static <R extends Record> Settings<R> bind(Class<R> root, List<Layer> layers, BaseLayer base) {
    return bind(root, withBase(root, layers, base));
  }

  /**
   * 層の並びのいちばん下に層を 1 つ足す。足す層は、渡された層を重ねた生の木で 1 つのキーの値を見て選ぶ。そのキーを書いた層が無ければ、
   * キーの既定値で選ぶ。
   *
   * @return 選んだ層を先頭に足した、下から順の層
   * @throws SettingsException キーが無いか読めないか、その値で層を作れない
   */
  public static List<Layer> withBase(Class<? extends Record> root, List<Layer> layers, BaseLayer base) {
    RecordCodec codec = Schema.record(root);
    List<String> keys = Toml.path(base.key());
    Component component = Schema.component(codec, keys);
    if (!(component.codec() instanceof ScalarCodec)) {
      throw new IllegalStateException("The base layer key " + base.key() + " must hold a single value");
    }
    Node node = Keys.find(Merge.layers(codec, layers, "", new ArrayList<>()), keys);
    if (node == null) {
      node = component.fallback();
    }
    if (node == null) {
      throw new SettingsException(List.of("Missing setting " + base.key()));
    }
    Binder binder = new Binder(Binder.Mode.SETTINGS, Collections::emptySet);
    if (binder.value(component.codec(), node, base.key()) == null) {
      throw new SettingsException(binder.problems());
    }
    String name = Keys.text(node);
    Layer below;
    try {
      below = Objects.requireNonNull(base.layer().apply(name), "base layer");
    } catch (IllegalArgumentException unknown) {
      throw new SettingsException(
          List.of(
              "Invalid setting " + base.key() + "=" + name + " at " + node.origin() + ": " + unknown.getMessage()));
    }
    List<Layer> stacked = new ArrayList<>(layers.size() + 1);
    stacked.add(below);
    stacked.addAll(layers);
    return List.copyOf(stacked);
  }

  /**
   * 層の中の 1 つのセクション（コマンドのセクション）を record に結び付ける。エラーメッセージのキーのパスは、セクションのパスから書く。
   * セクションのキーの変えてよい時期は見ない。
   *
   * @param path セクションのパス（{@code train}、{@code settings.change}）
   * @throws SettingsException 全てのエラーを集めたもの
   */
  public static <R extends Record> Settings<R> section(Class<R> type, String path, List<Layer> layers) {
    RecordCodec codec = Schema.record(type);
    List<String> keys = Toml.path(path);
    List<String> problems = new ArrayList<>();
    List<Layer> sections = new ArrayList<>();
    for (Layer layer : layers) {
      Node node = Keys.find(layer.tree(), keys);
      if (node instanceof Node.Table table) {
        sections.add(new Layer(layer.origin(), layer.source(), table, List.of()));
      } else if (node != null) {
        problems.add("Invalid setting " + path + "=" + Keys.text(node) + " at " + node.origin() + ": expected a table");
      }
    }
    Node.Table merged = Merge.layers(codec, sections, path, problems);
    Binder binder = new Binder(Binder.Mode.SETTINGS, () -> Schema.paths(codec, path));
    binder.problems().addAll(problems);
    Binder.Bound bound = binder.record(codec, merged, path, Change.INHERIT);
    if (!binder.problems().isEmpty()) {
      throw new SettingsException(binder.problems());
    }
    return new Settings<>(type.cast(bound.value()), (Node.Table) bound.node(), binder.changes());
  }

  /**
   * 組み立て済みの record から作る（テストと、プログラムが組んだ設定に使う）。値の書かれた場所は {@code record} になる。
   *
   * @throws SettingsException record の値が結び付けの検査を通らない（範囲の外など）
   */
  public static <R extends Record> Settings<R> of(R value) {
    @SuppressWarnings("unchecked")
    Class<R> type = (Class<R>) value.getClass();
    RecordCodec codec = Schema.record(type);
    Binder binder = new Binder(Binder.Mode.SETTINGS, () -> Schema.paths(codec, ""));
    Binder.Bound bound = binder.record(codec, (Node.Table) Schema.node(codec, value, "record"), "", null);
    if (!binder.problems().isEmpty()) {
      throw new SettingsException(binder.problems());
    }
    return new Settings<>(type.cast(bound.value()), (Node.Table) bound.node(), binder.changes());
  }

  public R value() {
    return value;
  }

  /**
   * 効いているキーの木。既定値も含めた全ての値と、値ごとの書かれた場所を持つ。種類を選ぶセクションは種類のキーを持つテーブルに直し、
   * 秘密の値は伏せる。設定の行にはこの木を書く。読み直すと、秘密のキーを除いて同じ record になる。
   */
  public Node.Table tree() {
    return tree;
  }

  /** 効いている全ての葉のキーと値。配列は 1 つの葉として扱う。キーのパスの順に並ぶ。 */
  public SortedMap<String, Node> effective() {
    SortedMap<String, Node> leaves = new TreeMap<>();
    flatten(tree, "", leaves);
    return Collections.unmodifiableSortedMap(leaves);
  }

  /**
   * 葉のキーを変えてよい時期。
   *
   * @return 効いているキーでなければ null
   */
  public Change change(String key) {
    return changes.get(key);
  }

  /** 前の設定から変わったキー。キーのパスの順に並ぶ。 */
  public List<Difference> since(Settings<R> previous) {
    SortedMap<String, Node> before = previous.effective();
    SortedMap<String, Node> after = effective();
    TreeSet<String> keys = new TreeSet<>(before.keySet());
    keys.addAll(after.keySet());
    List<Difference> differences = new ArrayList<>();
    for (String key : keys) {
      Node old = before.get(key);
      Node now = after.get(key);
      if (!Objects.equals(old, now)) {
        Change change = changes.containsKey(key) ? changes.get(key) : previous.changes.get(key);
        differences.add(new Difference(key, old, now, change));
      }
    }
    return List.copyOf(differences);
  }

  /**
   * 効いている値の TOML。キーとセクションの見出しの上にひな形と同じ説明を書き、既定値でない値には、書かれた場所を行の番号を除いて行末の
   * コメントに添える。
   */
  public String toml() {
    return Listing.write(Schema.record(value.getClass()), tree);
  }

  /** 効いている値の 1 行の JSON（設定の行の中身）。 */
  public String json() {
    return Json.write(tree);
  }

  /**
   * 全ての値を持つ木（設定の行の中身）から、既定値と同じ値を除いた木を返す。既定値は、コードの既定値と、{@code stack} が入れる四人打ち・
   * 三人打ちごとの層の値。値（葉と空のテーブル）を 1 つずつ除いて結び付け直し、効いている値が全て変わらなければ除く。既定値の無い値と、
   * 既定値と違う値は残る。
   *
   * @param root 全体の record の型
   * @param tree 全ての値を持つ木
   * @param stack 層のいちばん下に四人打ち・三人打ちごとの層を入れる（そのような層の無いアプリなら、層をそのまま返す）
   */
  public static <R extends Record> Node.Table beyondDefaults(
      Class<R> root, Node.Table tree, UnaryOperator<List<Layer>> stack) {
    SortedMap<String, Node> effective = bind(root, stack.apply(List.of(Layer.tree(ROW, tree)))).effective();
    Node.Table kept = tree;
    for (List<String> value : values(tree, List.of(), new ArrayList<>())) {
      Node.Table without = Keys.remove(kept, value);
      try {
        if (bind(root, stack.apply(List.of(Layer.tree(ROW, without)))).effective().equals(effective)) {
          kept = without;
        }
      } catch (SettingsException noDefault) {
        // 既定値の無い値（必須のキーなど）は残す
      }
    }
    return kept;
  }

  /**
   * 木の値（葉と空のテーブル）ごとに、書かれた場所を付け直した木を返す。
   *
   * @param origin キーのパス（{@link #effective} と同じ書き方）から、その値の書かれた場所を返す
   */
  public static Node.Table relabelValues(Node.Table tree, Function<String, String> origin) {
    return relabelValues(tree, "", origin);
  }

  private static Node.Table relabelValues(Node.Table table, String path, Function<String, String> origin) {
    Map<String, Node> entries = new LinkedHashMap<>();
    table.entries().forEach(
        (key, node) -> {
          String child = Keys.join(path, key);
          entries.put(
              key,
              node instanceof Node.Table inner && !inner.entries().isEmpty()
                  ? relabelValues(inner, child, origin)
                  : node.relabel(origin.apply(child)));
        });
    return new Node.Table(entries, table.origin());
  }

  /** 木の値（葉と空のテーブル）のキーの並びを、木の順に足す。 */
  private static List<List<String>> values(Node.Table table, List<String> path, List<List<String>> values) {
    table.entries().forEach(
        (key, node) -> {
          List<String> child = new ArrayList<>(path);
          child.add(key);
          if (node instanceof Node.Table inner && !inner.entries().isEmpty()) {
            values(inner, child, values);
          } else {
            values.add(List.copyOf(child));
          }
        });
    return values;
  }

  /**
   * キーの一覧（説明・型・既定値・範囲・変えてよい時期）。使い方の表示がこれを読む。種類を選ぶセクションは、種類のキーと全ての種類の
   * キーを並べる。名前をキーにしたテーブルは、{@code <name>} の項目のキーを並べる。親に従うキーは親の時期にする。いちばん外で親に従う
   * キー（コマンドの引数）は INHERIT のままにする。
   *
   * @throws IllegalStateException {@link Key} の注釈の無いコンポーネントがある
   */
  public static List<KeyInfo> describe(Class<? extends Record> root) {
    List<KeyInfo> keys = new ArrayList<>();
    describe(Schema.record(root), "", Change.INHERIT, "", keys);
    return List.copyOf(keys);
  }

  /**
   * ひな形の TOML。全てのキーを説明つきで書く。設定ファイルとコマンドラインの層が書いた値はそのまま書き、ほかの値（既定値、プリセットと
   * 四人打ち・三人打ちごとの層の値）はコメントにする。
   *
   * @param layers 下から順の層（四人打ち・三人打ちごとの層を入れるなら、{@link #withBase} を通した後のもの）
   * @param presets 先頭の {@value Layer#PRESETS} に書くプリセットの名前
   */
  public static String template(Class<? extends Record> root, List<Layer> layers, List<String> presets) {
    RecordCodec codec = Schema.record(root);
    List<String> ignored = new ArrayList<>();
    Node.Table merged = Merge.layers(codec, layers, "", ignored);
    List<Layer> writtenLayers =
        layers.stream()
            .filter(layer -> layer.source() == Layer.Source.FILE || layer.source() == Layer.Source.COMMAND_LINE)
            .toList();
    Node.Table written = Merge.layers(codec, writtenLayers, "", ignored);
    return Template.write(codec, merged, written, presets);
  }

  private static void flatten(Node node, String path, SortedMap<String, Node> leaves) {
    if (node instanceof Node.Table table) {
      table.entries().forEach((key, child) -> flatten(child, Keys.join(path, key), leaves));
    } else {
      leaves.put(path, node);
    }
  }

  private static void describe(
      RecordCodec codec, String path, Change change, String condition, List<KeyInfo> keys) {
    for (Component component : codec.components()) {
      String child = Keys.join(path, component.name());
      Key key = component.key();
      if (key == null) {
        throw new IllegalStateException("Setting " + child + " has no @Key");
      }
      Change resolved = key.change() == Change.INHERIT ? change : key.change();
      switch (component.codec()) {
        case RecordCodec record -> {
          keys.add(info(child, key, component, resolved, condition));
          describe(record, child, resolved, condition, keys);
        }
        case ChoiceCodec choice -> {
          String discriminator = Keys.join(child, choice.key());
          keys.add(info(discriminator, key, component, resolved, condition));
          choice.cases().forEach((name, chosen) -> describe(chosen, child, resolved, discriminator + "=" + name, keys));
        }
        case MapCodec map when map.value() instanceof RecordCodec record -> {
          keys.add(info(child, key, component, resolved, condition));
          describe(record, child + ".<name>", resolved, condition, keys);
        }
        default -> keys.add(info(child, key, component, resolved, condition));
      }
    }
  }

  private static KeyInfo info(String path, Key key, Component component, Change change, String condition) {
    String range =
        component.codec() instanceof ScalarCodec scalar ? Schema.bounds(scalar.range(), scalar.kind().integral()) : "";
    return new KeyInfo(
        path,
        key.doc(),
        Schema.helpLabel(component.codec()),
        key.value(),
        key.presence(),
        range,
        change,
        key.secret(),
        condition);
  }
}
