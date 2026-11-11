package com.epsilon.agent.settings;

import com.epsilon.agent.settings.Schema.ChoiceCodec;
import com.epsilon.agent.settings.Schema.Codec;
import com.epsilon.agent.settings.Schema.Component;
import com.epsilon.agent.settings.Schema.MapCodec;
import com.epsilon.agent.settings.Schema.RecordCodec;
import java.util.Map;
import java.util.regex.Pattern;

/**
 * 効いている設定の TOML（{@code settings show}）。キーの並びは {@link Toml#write} と同じで、キーとセクションの見出しの上に、ひな形と
 * 同じ説明の行を書く。既定値でない値には、行末に書かれた場所を行の番号を除いて添える。
 */
final class Listing {

  /** 書かれた場所の末尾の行の番号（{@code run.toml:5} の {@code :5}）。 */
  private static final Pattern LINE = Pattern.compile(":\\d+$");

  private final StringBuilder out = new StringBuilder();

  private Listing() {}

  /**
   * テーブル 1 つの説明の引き方。
   *
   * @param codec テーブルの型（record、種類を選ぶセクション、名前か enum をキーにしたテーブル）。説明の無いテーブルなら null
   * @param change テーブルの変えてよい時期。いちばん外では null
   * @param header 見出しの上に書く説明。無ければ null
   * @param entries 名前をキーにしたテーブルの項目の見出しの上に書く説明。ほかのテーブルでは null
   */
  private record Scope(Codec codec, Change change, String header, String entries) {}

  /**
   * @param root 全体の record の型
   * @param tree 効いているキーの木
   */
  static String write(RecordCodec root, Node.Table tree) {
    Listing listing = new Listing();
    listing.table(new Scope(root, null, null, null), tree, "");
    return listing.out.toString();
  }

  private void table(Scope scope, Node.Table table, String path) {
    for (Map.Entry<String, Node> entry : table.entries().entrySet()) {
      Node node = entry.getValue();
      if (section(node)) {
        continue;
      }
      String doc = doc(scope, table, path, entry.getKey());
      if (doc != null) {
        out.append("# ").append(doc).append('\n');
      }
      out.append(Keys.quote(entry.getKey())).append(" = ").append(Toml.inline(node));
      if (!node.origin().equals(Binder.DEFAULT_ORIGIN)) {
        out.append("  # ").append(LINE.matcher(node.origin()).replaceFirst(""));
      }
      out.append('\n');
    }
    for (Map.Entry<String, Node> entry : table.entries().entrySet()) {
      if (entry.getValue() instanceof Node.Table child && section(child)) {
        String childPath = Keys.join(path, entry.getKey());
        Scope inner = scope(scope, table, entry.getKey());
        if (child.entries().values().stream().anyMatch(node -> !section(node))) {
          if (!out.isEmpty()) {
            out.append('\n');
          }
          if (inner.header() != null) {
            out.append("# ").append(inner.header()).append('\n');
          }
          out.append('[').append(childPath).append("]\n");
        }
        table(inner, child, childPath);
      }
    }
  }

  /** テーブルのキーの行の上に書く説明。説明の無いキー（値をキーごとに持つテーブルの項目）なら null。 */
  private static String doc(Scope scope, Node.Table table, String path, String name) {
    if (scope.codec() instanceof ChoiceCodec choice && name.equals(choice.key())) {
      return Template.kindDoc(choice, path);
    }
    Component component = component(scope.codec(), table, name);
    if (component == null || component.key() == null) {
      return null;
    }
    return Template.doc(component, resolve(component, scope.change()), scope.change(), false);
  }

  /** セクションに書くキーの説明の引き方。 */
  private static Scope scope(Scope scope, Node.Table table, String name) {
    if (scope.codec() instanceof MapCodec map) {
      return new Scope(map.value(), scope.change(), scope.entries(), null);
    }
    Component component = component(scope.codec(), table, name);
    if (component == null) {
      return new Scope(null, scope.change(), null, null);
    }
    Change change = resolve(component, scope.change());
    return switch (component.codec()) {
      case MapCodec map when Schema.inline(map) ->
          new Scope(map, change, Template.doc(component, change, scope.change(), false), null);
      case MapCodec map -> new Scope(map, change, null, Template.sectionDoc(component, change));
      default -> new Scope(component.codec(), change, Template.sectionDoc(component, change), null);
    };
  }

  /** テーブルのキーのコンポーネント。種類を選ぶセクションでは、選んだ種類のコンポーネント。無ければ null。 */
  private static Component component(Codec codec, Node.Table table, String name) {
    return switch (codec) {
      case RecordCodec record -> record.component(name);
      case ChoiceCodec choice -> {
        RecordCodec chosen = choice.cases().get(ChoiceCodec.name(table.get(choice.key())));
        yield chosen == null ? null : chosen.component(name);
      }
      case null, default -> null;
    };
  }

  private static Change resolve(Component component, Change parent) {
    return component.key() == null || component.key().change() == Change.INHERIT ? parent : component.key().change();
  }

  /** 見出しを付けたセクションとして書くテーブルか。空のテーブルは {@code {}} で書く。 */
  private static boolean section(Node node) {
    return node instanceof Node.Table table && !table.entries().isEmpty();
  }
}
