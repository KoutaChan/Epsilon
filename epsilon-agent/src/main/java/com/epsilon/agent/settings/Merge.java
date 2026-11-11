package com.epsilon.agent.settings;

import com.epsilon.agent.settings.Schema.ChoiceCodec;
import com.epsilon.agent.settings.Schema.Codec;
import com.epsilon.agent.settings.Schema.Component;
import com.epsilon.agent.settings.Schema.MapCodec;
import com.epsilon.agent.settings.Schema.RecordCodec;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 層を下から重ねる。テーブルはキーごとに重ね、配列と値は上の層のもので置き換える。種類を選ぶセクションは、上の層が種類のキーを違う値に
 * したら丸ごと置き換える。全体の record の型をたどって重ねるので、種類を選ぶセクションや名前をキーにしたテーブルの中でも同じように重なる。
 */
final class Merge {

  private Merge() {}

  /**
   * @param prefix 層の木が全体のどのセクションか（コマンドのセクションなら {@code train}。全体なら空）
   * @param problems 設定ファイル以外の層に秘密のキーがあれば、そのエラーをここに足す。そのキーは層から取り除く
   */
  static Node.Table layers(RecordCodec root, List<Layer> layers, String prefix, List<String> problems) {
    List<List<String>> secrets = Schema.secrets(root);
    Node.Table merged = new Node.Table(Map.of(), Binder.DEFAULT_ORIGIN);
    for (Layer layer : layers) {
      Node.Table tree = layer.tree();
      if (layer.source() != Layer.Source.FILE) {
        for (List<String> secret : secrets) {
          Node written = Keys.find(tree, secret);
          if (written != null) {
            String path = String.join(".", secret);
            problems.add(
                "Setting " + (prefix.isEmpty() ? path : prefix + "." + path) + " at " + written.origin()
                    + " is secret and is read only from the settings file");
            tree = Keys.remove(tree, secret);
          }
        }
      }
      merged = tables(merged, tree, root, null);
    }
    return merged;
  }

  /**
   * キーの既定値を、いちばん下の層として重ねる。重なるのはテーブルと種類を選ぶセクションだけで、値と配列は層の値のまま。
   *
   * @param node 層を重ねたノード。無ければ null
   * @return 既定値を重ねたノード。どちらも無ければ null
   */
  static Node withDefault(Component component, Node node) {
    Node fallback = component.fallback();
    if (fallback == null || node == null) {
      return node == null ? fallback : node;
    }
    Codec codec = component.codec();
    boolean table = codec instanceof RecordCodec || codec instanceof MapCodec || codec instanceof ChoiceCodec;
    return table ? node(codec, fallback, fallback, node) : node;
  }

  /**
   * 1 つのキーについて、下の層のノードに上の層のノードを重ねる。
   *
   * @param fallback キーの既定値。種類を選ぶセクションで、下の層が種類のキーを書いていないときに、下の層の種類を知るのに使う
   */
  private static Node node(Codec codec, Node fallback, Node lower, Node upper) {
    if (lower == null) {
      return upper;
    }
    return switch (codec) {
      case RecordCodec record when lower instanceof Node.Table below && upper instanceof Node.Table above ->
          tables(below, above, record, null);
      case MapCodec map when lower instanceof Node.Table below && upper instanceof Node.Table above ->
          tables(below, above, null, map.value());
      case ChoiceCodec choice -> choice(choice, fallback, lower, upper);
      default -> upper;
    };
  }

  /**
   * 2 つのテーブルをキーごとに重ねる。
   *
   * @param record テーブルが record なら、その型。キーごとの型と既定値をここから引く
   * @param values 名前をキーにしたテーブルなら、値の型
   */
  private static Node.Table tables(Node.Table lower, Node.Table upper, RecordCodec record, Codec values) {
    Map<String, Node> entries = new LinkedHashMap<>(lower.entries());
    upper.entries().forEach(
        (name, node) -> {
          Component component = record == null ? null : record.component(name);
          Codec codec = component != null ? component.codec() : values;
          Node fallback = component == null ? null : component.fallback();
          entries.put(name, codec == null ? node : node(codec, fallback, entries.get(name), node));
        });
    return new Node.Table(entries, upper.origin());
  }

  private static Node choice(ChoiceCodec codec, Node fallback, Node lower, Node upper) {
    Node.Table below = codec.normalize(lower);
    Node.Table above = codec.normalize(upper);
    if (below == null || above == null) {
      return upper;
    }
    Node upperKind = above.get(codec.key());
    Node lowerKind = below.get(codec.key());
    if (lowerKind == null) {
      Node.Table defaults = codec.normalize(fallback);
      lowerKind = defaults == null ? null : defaults.get(codec.key());
    }
    if (upperKind != null
        && lowerKind != null
        && !Objects.equals(ChoiceCodec.name(upperKind), ChoiceCodec.name(lowerKind))) {
      return above;
    }
    String name = ChoiceCodec.name(upperKind != null ? upperKind : lowerKind);
    return tables(below, above, name == null ? null : codec.cases().get(name), null);
  }
}
