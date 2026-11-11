package com.epsilon.agent.settings;

import com.epsilon.agent.settings.Schema.ChoiceCodec;
import com.epsilon.agent.settings.Schema.Component;
import com.epsilon.agent.settings.Schema.MapCodec;
import com.epsilon.agent.settings.Schema.RecordCodec;
import com.epsilon.agent.settings.Schema.ScalarCodec;
import java.util.ArrayList;
import java.util.List;
import java.util.TreeMap;

/**
 * ひな形の TOML。全てのキーを説明つきで書く。設定ファイルとコマンドラインの層が書いた値はそのまま書き、ほかの値（既定値、プリセットと
 * 四人打ち・三人打ちごとの層の値）はコメントにする。コメントを外すと、その値を書いたことになる。
 *
 * <p>種類を選ぶセクションと、名前をキーにしたテーブルの項目は、設定ファイルとコマンドラインの層が書いていなければ、セクションごと
 * コメントにする。見出しだけを残すと、種類のキーの無いセクションや空の項目になってしまうため。
 */
final class Template {

  private final StringBuilder out = new StringBuilder();

  private Template() {}

  /**
   * @param merged 全ての層を重ねた木
   * @param written 設定ファイルとコマンドラインの層だけを重ねた木
   * @param presets 先頭に書くプリセットの名前
   */
  static String write(RecordCodec root, Node.Table merged, Node.Table written, List<String> presets) {
    Template template = new Template();
    if (!presets.isEmpty()) {
      List<Node> names = presets.stream().map(name -> (Node) new Node.Text(name, "")).toList();
      template.out.append(Layer.PRESETS).append(" = ").append(Toml.inline(new Node.Array(names, ""))).append('\n');
    }
    template.section(root, merged, written, "", null, false);
    return template.out.toString();
  }

  private void section(
      RecordCodec codec, Node.Table merged, Node.Table written, String path, Change change, boolean commented) {
    for (Component component : codec.components()) {
      if (Schema.inline(component.codec())) {
        Change resolved = resolve(component, change, path);
        line(component, entry(merged, component), entry(written, component), resolved, change, commented);
      }
    }
    for (Component component : codec.components()) {
      if (Schema.inline(component.codec())) {
        continue;
      }
      String child = Keys.join(path, component.name());
      Change resolved = resolve(component, change, path);
      switch (component.codec()) {
        case RecordCodec record -> {
          header(child, component, resolved, commented);
          Node.Table mergedSection = table(entry(merged, component));
          section(record, mergedSection, table(entry(written, component)), child, resolved, commented);
        }
        case ChoiceCodec choice -> choice(component, choice, merged, written, child, resolved, commented);
        case MapCodec map -> names(component, map, merged, written, child, resolved, commented);
        default -> throw new IllegalStateException("Unexpected section " + child);
      }
    }
  }

  private void choice(
      Component component,
      ChoiceCodec codec,
      Node.Table merged,
      Node.Table written,
      String path,
      Change change,
      boolean commented) {
    Node.Table chosen = codec.normalize(Merge.withDefault(component, entry(merged, component)));
    Node.Table chosenWritten = codec.normalize(entry(written, component));
    boolean comment = commented || chosenWritten == null;
    Node kind = chosen == null ? null : chosen.get(codec.key());
    String name = ChoiceCodec.name(kind);
    header(path, component, change, comment);
    String discriminator = Keys.quote(codec.key());
    out.append("# ").append(kindDoc(codec, path)).append('\n');
    if (kind == null) {
      out.append("# ").append(discriminator).append(" =\n");
      return;
    }
    out.append(comment ? "# " : "").append(discriminator).append(" = ").append(Toml.inline(kind));
    origin(kind, comment);
    RecordCodec chosenCase = name == null ? null : codec.cases().get(name);
    if (chosenCase != null) {
      section(chosenCase, chosen, chosenWritten, path, change, comment);
    }
  }

  private void names(
      Component component,
      MapCodec codec,
      Node.Table merged,
      Node.Table written,
      String path,
      Change change,
      boolean commented) {
    if (!(codec.value() instanceof RecordCodec value)) {
      throw new IllegalStateException("Unexpected table of names " + path);
    }
    Node.Table entries = table(entry(merged, component));
    Node.Table writtenEntries = table(entry(written, component));
    if (entries == null || entries.entries().isEmpty()) {
      String example = path + ".<名前>";
      header(example, component, change, true);
      section(value, null, null, example, change, true);
      return;
    }
    for (String name : new TreeMap<>(entries.entries()).keySet()) {
      Node.Table writtenEntry = writtenEntries == null ? null : table(writtenEntries.get(name));
      boolean comment = commented || writtenEntry == null;
      String child = Keys.join(path, name);
      header(child, component, change, comment);
      section(value, table(entries.get(name)), writtenEntry, child, change, comment);
    }
  }

  private void line(Component component, Node value, Node written, Change change, Change table, boolean commented) {
    Key key = component.key();
    out.append("# ").append(doc(component, change, table, true)).append('\n');
    Node shown = Merge.withDefault(component, value);
    String name = Keys.quote(component.name());
    if (shown == null) {
      out.append("# ").append(name).append(" =\n");
      return;
    }
    boolean comment = commented || written == null;
    out.append(comment ? "# " : "").append(name).append(" = ").append(key.secret() ? "\"\"" : Toml.inline(shown));
    origin(shown, comment);
  }

  private void origin(Node node, boolean comment) {
    if (comment && !node.origin().equals(Binder.DEFAULT_ORIGIN)) {
      out.append("  # ").append(node.origin());
    }
    out.append('\n');
  }

  private void header(String path, Component component, Change change, boolean comment) {
    if (!out.isEmpty()) {
      out.append('\n');
    }
    out.append("# ").append(sectionDoc(component, change)).append('\n');
    out.append(comment ? "# " : "").append('[').append(path).append("]\n");
  }

  /** セクションの見出しの上に書く説明。キーの説明に、変えてよい時期を添える。 */
  static String sectionDoc(Component component, Change change) {
    String label = label(change);
    return component.key().doc() + (label.isEmpty() ? "" : "（" + label + "）");
  }

  /**
   * キーの行の上に書く説明。キーの説明に、値の行からは分からないことの注を添える: 選べる値、範囲、秘密のキーであること、セクションと違う
   * ときだけの変えてよい時期。ひな形では、必須か省略できるかも添える。注に書くことが無ければ説明だけにする。
   *
   * @param change キーの変えてよい時期
   * @param table キーのあるセクションの変えてよい時期。いちばん外のキーでは null
   * @param template ひな形に書くか
   */
  static String doc(Component component, Change change, Change table, boolean template) {
    List<String> parts = new ArrayList<>();
    String choices = Schema.choices(component.codec());
    if (!choices.isEmpty()) {
      parts.add(choices);
    }
    if (component.codec() instanceof ScalarCodec scalar && scalar.range() != null) {
      parts.add(Schema.bounds(scalar.range(), scalar.kind().integral()));
    }
    if (template) {
      switch (component.key().presence()) {
        case REQUIRED -> parts.add("必須");
        case ABSENT -> parts.add("省略できる");
        case DEFAULT -> {}
      }
    }
    if (component.key().secret()) {
      parts.add("秘密。設定のファイルにだけ書く");
    }
    if (change != table) {
      String label = label(change);
      if (!label.isEmpty()) {
        parts.add(label);
      }
    }
    String doc = component.key().doc();
    return parts.isEmpty() ? doc : doc + "（" + String.join("、", parts) + "）";
  }

  /**
   * 種類のキーの行の上に書く説明（何を選ぶかと、選べる種類）。
   *
   * @throws IllegalStateException 種類を選ぶセクションに {@link Choice#doc} が無い
   */
  static String kindDoc(ChoiceCodec codec, String path) {
    if (codec.doc().isEmpty()) {
      throw new IllegalStateException("Setting " + Keys.join(path, codec.key()) + " has no doc in its @Choice");
    }
    return codec.doc() + "（" + Schema.choices(codec) + "）";
  }

  private static String label(Change change) {
    if (change == null) {
      return "";
    }
    return switch (change) {
      case FIXED -> "run の間は変えない";
      case COLLECT -> "次の反復の収集から効く";
      case FIT -> "次の反復の当てはめから効く";
      case MACHINE -> "次の起動から効く";
      case SCHEDULE -> "次に公開するバージョンの仕事から効く";
      case INHERIT -> "";
    };
  }

  private static Change resolve(Component component, Change parent, String path) {
    Change change = component.key().change() == Change.INHERIT ? parent : component.key().change();
    if (change == null) {
      throw new IllegalStateException(
          "Top-level setting " + Keys.join(path, component.name()) + " must say when it may change");
    }
    return change;
  }

  private static Node entry(Node.Table table, Component component) {
    if (component.key() == null) {
      throw new IllegalStateException("Setting " + component.name() + " has no @Key");
    }
    return table == null ? null : table.get(component.name());
  }

  private static Node.Table table(Node node) {
    return node instanceof Node.Table table ? table : null;
  }
}
