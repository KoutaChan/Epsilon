package com.epsilon.agent.settings;

import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** キーのパスを組み立てて木を引く。知らないキーには、近いキーを探して示す。 */
final class Keys {

  private Keys() {}

  /** キーの中で引用符なしに書ける文字か。 */
  static boolean bareChar(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
  }

  /** キーを、パスの中に書く形にする。引用符が要るキーだけを引用符で囲む。 */
  static String quote(String key) {
    boolean bare = !key.isEmpty();
    for (int index = 0; bare && index < key.length(); index++) {
      bare = bareChar(key.charAt(index));
    }
    return bare ? key : Toml.quote(key);
  }

  /** 親のパスにキーを 1 つ足す。 */
  static String join(String parent, String key) {
    return parent.isEmpty() ? quote(key) : parent + "." + quote(key);
  }

  /** @return パスの先のノード。途中がテーブルでないか、ノードが無ければ null */
  static Node find(Node.Table root, List<String> keys) {
    Node node = root;
    for (String key : keys) {
      if (!(node instanceof Node.Table table)) {
        return null;
      }
      node = table.get(key);
      if (node == null) {
        return null;
      }
    }
    return node;
  }

  /** パスの先のノードを除いた木。途中がテーブルでなければ、木をそのまま返す。 */
  static Node.Table remove(Node.Table root, List<String> keys) {
    if (keys.size() == 1) {
      return root.without(keys.getFirst());
    }
    if (!(root.get(keys.getFirst()) instanceof Node.Table child)) {
      return root;
    }
    Map<String, Node> entries = new LinkedHashMap<>(root.entries());
    entries.put(keys.getFirst(), remove(child, keys.subList(1, keys.size())));
    return new Node.Table(entries, root.origin());
  }

  /** エラーメッセージに書く値。文字列は引用符を付けず、配列とテーブルは 1 行の TOML で書く。 */
  static String text(Node node) {
    return switch (node) {
      case Node.Text text -> text.value();
      case Node.Numeral numeral -> numeral.literal();
      case Node.Bool bool -> Boolean.toString(bool.value());
      case Node.Array array -> Toml.inline(array);
      case Node.Table table -> Toml.inline(table);
    };
  }

  /**
   * 知らないキーに近いキーを探す。知らないキーが {@code .} に続けて知っているキーで終わるなら、当てはまるキーのうち最も長いものを選ぶ。
   * そうでなければ、編集距離が最も小さいキーを選ぶ。ただし距離がそのキーの長さの 3 分の 1（少なくとも 2）を超えるなら選ばない。
   *
   * @return 近いキー。無ければ null
   */
  static String similar(String unknown, Collection<String> known) {
    String suffix = null;
    String best = null;
    int bestDistance = Integer.MAX_VALUE;
    for (String candidate : known) {
      if (unknown.endsWith("." + candidate)) {
        if (suffix == null || candidate.length() > suffix.length()) {
          suffix = candidate;
        }
        continue;
      }
      int distance = distance(unknown, candidate);
      if (distance < bestDistance) {
        best = candidate;
        bestDistance = distance;
      }
    }
    if (suffix != null) {
      return suffix;
    }
    return best != null && bestDistance <= Math.max(2, best.length() / 3) ? best : null;
  }

  /** 2 つの文字列の編集距離（1 文字の挿入・削除・置換の回数）。 */
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
        current[j] = Math.min(Math.min(current[j - 1], previous[j]) + 1, replace);
      }
      int[] swap = previous;
      previous = current;
      current = swap;
    }
    return previous[b.length()];
  }
}
