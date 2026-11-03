package com.epsilon.prism.run;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.regex.Pattern;

/**
 * 系譜の中の評価 1 つ（設定の {@code [epsilon.prism.evaluation.<名前>]}）: バージョンの打つ網 1 席対 opponent 3 席を games 局打つ（{@link
 * NanoTable}。全員貪欲で、牌山と半荘 ID は全ての評価で同じ）。測るのはバージョンが every の倍数で from 以上のとき（every = 0 なら測らない）で、
 * 出力は {@link EvaluationOutputs}。
 *
 * @param name 名前（英数字。出力のファイル名に入る）
 * @param opponent 相手（{@value NanoTable#HEURISTIC} か {@value NanoTable#NANO}&lt;Nano の checkpoint の絶対パス&gt;）
 * @param games 半荘の数（1 以上）
 * @param every 測るバージョンの間隔（0 以上。0 なら測らない）
 * @param from 測る最初のバージョン（1 以上）
 */
public record Evaluation(String name, String opponent, int games, int every, int from) {

  /** 設定の鍵の接頭辞（{@code epsilon.prism.evaluation.<名前>.<欄>}）。 */
  public static final String SETTINGS = PrismConfig.SETTINGS + "evaluation.";

  /** 名前の書き方。 */
  private static final Pattern NAME = Pattern.compile("[A-Za-z0-9]+");

  /** 設定に書く欄（名前は鍵に入る）。欠けた欄はこの順で最初の 1 つを報告する。 */
  private static final List<String> FIELDS = List.of("opponent", "games", "every", "from");

  /** 範囲を確かめる。 */
  public void validate() {
    if (!NAME.matcher(name).matches())
      throw new IllegalArgumentException("Evaluation names are alphanumeric: '" + name + "'");
    if (!NanoTable.validOpponent(opponent))
      throw new IllegalArgumentException(
          "Evaluation " + name + " opponent must be " + NanoTable.HEURISTIC + " or " + NanoTable.NANO
              + "<absolute path>: '" + opponent + "'");
    if (games < 1 || every < 0 || from < 1)
      throw new IllegalArgumentException(
          "Invalid evaluation " + name + " games/every/from " + games + "/" + every + "/" + from);
  }

  /** latest までで測るバージョン（古い順）。 */
  public List<Integer> versions(int latest) {
    List<Integer> versions = new ArrayList<>();
    if (every == 0) return versions;
    for (int version = Math.ceilDiv(from, every) * every; version <= latest; version += every)
      versions.add(version);
    return versions;
  }

  /** 設定の鍵（接頭辞 {@value #SETTINGS} の後）が {@code <名前>.<欄>} の形なら true。 */
  static boolean settingsKey(String key) {
    int dot = key.lastIndexOf('.');
    return dot > 0 && FIELDS.contains(key.substring(dot + 1));
  }

  /**
   * 設定の値から評価を作る（名前の順）。名前ごとに欄が全てそろっていなければ拒む。
   *
   * @throws IllegalArgumentException 欄が欠けているか、数でない値のとき
   */
  static List<Evaluation> fromSettings(Map<String, String> values) {
    Map<String, Map<String, String>> byName = new TreeMap<>();
    for (var entry : values.entrySet()) {
      if (!entry.getKey().startsWith(SETTINGS)) continue;
      String key = entry.getKey().substring(SETTINGS.length());
      if (!settingsKey(key)) continue;
      int dot = key.lastIndexOf('.');
      byName.computeIfAbsent(key.substring(0, dot), name -> new TreeMap<>()).put(key.substring(dot + 1), entry.getValue());
    }
    List<Evaluation> evaluations = new ArrayList<>();
    byName.forEach(
        (name, fields) -> {
          for (String field : FIELDS)
            if (!fields.containsKey(field))
              throw new IllegalArgumentException("Missing setting " + SETTINGS + name + "." + field);
          evaluations.add(
              new Evaluation(
                  name,
                  fields.get("opponent"),
                  number(name, "games", fields),
                  number(name, "every", fields),
                  number(name, "from", fields)));
        });
    return evaluations;
  }

  private static int number(String name, String field, Map<String, String> fields) {
    try {
      return Integer.parseInt(fields.get(field).trim());
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException(SETTINGS + name + "." + field + " must be an integer: " + fields.get(field));
    }
  }
}
