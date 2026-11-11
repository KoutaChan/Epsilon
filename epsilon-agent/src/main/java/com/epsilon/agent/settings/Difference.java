package com.epsilon.agent.settings;

import java.util.List;
import java.util.Objects;

/**
 * 前の設定から変わった、1 つのキー。
 *
 * @param key キーのパス
 * @param before 前の値。無かったなら null
 * @param after 後の値。無くなったなら null
 * @param change キーを変えてよい時期
 */
public record Difference(String key, Node before, Node after, Change change) {

  public Difference {
    Objects.requireNonNull(key, "key");
    Objects.requireNonNull(change, "change");
    if (before == null && after == null) {
      throw new IllegalArgumentException("A difference needs a value before or after: " + key);
    }
  }

  /**
   * run の間は変えないキーの変化を拒む。
   *
   * @throws SettingsException そのような変化があるとき。変化を全て並べて投げる
   */
  public static void rejectFixed(List<Difference> differences) {
    List<String> problems =
        differences.stream()
            .filter(difference -> difference.change() == Change.FIXED)
            .map(
                difference ->
                    "Setting "
                        + difference.key()
                        + " is fixed for the run: "
                        + (difference.before() == null
                            ? "the run started without it"
                            : "the run started with " + Keys.text(difference.before()))
                        + ", "
                        + (difference.after() == null
                            ? "the change removes it"
                            : "the change asks for " + Keys.text(difference.after())))
            .toList();
    if (!problems.isEmpty()) {
      throw new SettingsException(problems);
    }
  }
}
