package com.epsilon.agent.eval;

import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.regex.Pattern;

/**
 * 評価と決闘の設定のセクション。先頭に尺度とリグレッションの扱いを置き、その後に卓を名前ごとに並べる。卓には既定値が無いので、書かなければ
 * 何も測らない。どのキーも、次に公開するバージョンのジョブから効く。
 *
 * <p>尺度の基準の卓を書くときは、尺度に入る卓（相手が {@code version:N} か {@code version:-d} の卓）の判定に使う量を、基準の卓と
 * そろえる。
 *
 * <p>王者を替える卓が書いた確かめの卓は、全てのバージョンで同じ牌山を全て打ち（止め方は固定）、呼ぶ卓と同じ量で判定し、予定では打たない（間隔は
 * 0）。相手は系譜の外の凍結した打ち手（ルールベースの打ち手かチェックポイント）で、学習の相手に入らない。幅は確かめの卓にだけ書く。
 *
 * @param anchor 尺度の基準の卓の名前（無ければ尺度を作らない）
 * @param alarmZ アラートを出すしきい値（標準誤差の倍数）
 * @param dropMargin 基準の卓で低下のアラートを出すのに要る、王者の値からの悪化の幅（判定に使う量の単位）
 * @param onWorse 王者との決闘で「悪い」が続いたときの扱い
 * @param tables 名前ごとの卓（名前の順）
 */
public record EvaluationSettings(
    @Key(doc = "尺度の基準の卓（書かなければ尺度を作らない）", presence = Presence.ABSENT, change = Change.SCHEDULE)
        String anchor,
    @Key(doc = "アラートを出すしきい値（標準誤差の倍数）", value = "3.0", change = Change.SCHEDULE) @Range(min = 0, minInclusive = false)
        double alarmZ,
    @Key(doc = "基準の卓で低下のアラートを出すのに要る、王者の値からの悪化の幅（判定に使う量の単位）", value = "0.01",
            change = Change.SCHEDULE)
        @Range(min = 0)
        double dropMargin,
    @Key(doc = "王者との決闘で「悪い」が続いたときの扱い", value = "REPORT", change = Change.SCHEDULE) OnWorseSettings onWorse,
    @Key(doc = "名前ごとの卓", change = Change.SCHEDULE) Map<String, TableSpec> tables) {

  private static final Pattern NAME = Pattern.compile("[A-Za-z0-9]+");

  public EvaluationSettings {
    Objects.requireNonNull(onWorse, "onWorse");
    tables = Collections.unmodifiableSortedMap(new TreeMap<>(tables));
    for (String name : tables.keySet()) {
      if (!NAME.matcher(name).matches()) {
        throw new InvalidSetting("tables", "table names are alphanumeric, not " + name);
      }
    }
    if (anchor != null) {
      TableSpec base = tables.get(anchor);
      if (base == null) {
        throw new InvalidSetting("anchor", "no table is named " + anchor);
      }
      if (base.wallSet() != WallSet.FIXED
          || base.stop() instanceof StopSettings.Test
          || TableSpec.kind(base.opponent()) != TableSpec.Opponent.VERSION) {
        throw new InvalidSetting(
            "anchor", "the anchor table plays fixed walls against version:N and stops at a fixed size or precision");
      }
      for (Map.Entry<String, TableSpec> entry : tables.entrySet()) {
        if (rated(entry.getValue()) && entry.getValue().score() != base.score()) {
          throw new InvalidSetting(
              "anchor",
              "the tables against version:N and version:-d score " + base.score() + " like the anchor table, not "
                  + entry.getValue().score() + " (" + entry.getKey() + ")");
        }
      }
    }
    long promoting = tables.values().stream().filter(EvaluationSettings::promotes).count();
    if (promoting > 1) {
      throw new InvalidSetting("tables", "at most one table promotes the champion, not " + promoting);
    }
    for (Map.Entry<String, TableSpec> entry : tables.entrySet()) {
      String confirm = entry.getValue().confirm();
      if (confirm != null) {
        requireConfirm(entry.getKey(), entry.getValue(), confirm, tables.get(confirm));
      }
    }
    for (Map.Entry<String, TableSpec> entry : tables.entrySet()) {
      boolean named = tables.values().stream().anyMatch(table -> entry.getKey().equals(table.confirm()));
      if (named != (entry.getValue().confirmMargin() != null)) {
        throw new InvalidSetting(
            "tables." + entry.getKey() + ".confirmMargin",
            named ? "a confirm table needs a confirmMargin" : "only a confirm table has a confirmMargin");
      }
    }
  }

  /** @return 王者を替える卓（判定の止め方で、良いなら王者を候補に替える卓）の名前。無ければ空 */
  public Optional<String> promoting() {
    for (Map.Entry<String, TableSpec> entry : tables.entrySet()) {
      if (promotes(entry.getValue())) {
        return Optional.of(entry.getKey());
      }
    }
    return Optional.empty();
  }

  /**
   * このバージョンを定期的に測る卓を返す。
   *
   * <p>ラダー（相手が {@code version:-d} で、止め方が {@code TEST} でない卓）は、基準の卓（相手が {@code version:N} の卓）の結果で
   * 判定する回（{@link #coveredLadders}）には打たないので、ここには入れない。相手のバージョンが基準の卓の相手と同じでも、席の決め方・
   * ツモの順・判定に使う量のどれかが違えば、ラダーを打つ。
   *
   * @param version バージョン
   * @return 名前ごとの卓（名前の順）
   */
  public SortedMap<String, TableSpec> scheduled(int version) {
    SortedMap<String, TableSpec> scheduled = new TreeMap<>();
    for (Map.Entry<String, TableSpec> entry : tables.entrySet()) {
      TableSpec table = entry.getValue();
      if (table.scheduled(version) && !covered(table, version)) {
        scheduled.put(entry.getKey(), table);
      }
    }
    return scheduled;
  }

  /**
   * 基準の卓がこのバージョンを定期的に測る回に、その結果を判定に使うラダーを返す。入るのは、同じバージョンを定期的に測り、相手のバージョンと、
   * 席の決め方（分け方を含む）、ツモの順、判定に使う量が基準の卓と同じラダー。牌山の集め方と数、止め方は違ってよい。
   *
   * @param baseline 卓の名前
   * @param version バージョン
   * @return ラダーの名前（名前の順）。卓が基準の卓でないか、このバージョンを定期的に測らなければ空
   */
  public List<String> coveredLadders(String baseline, int version) {
    TableSpec base = tables.get(baseline);
    if (base == null) {
      return List.of();
    }
    List<String> ladders = new ArrayList<>();
    for (Map.Entry<String, TableSpec> entry : tables.entrySet()) {
      if (covers(base, entry.getValue(), version)) {
        ladders.add(entry.getKey());
      }
    }
    return ladders;
  }

  /** このバージョンを定期的に測る基準の卓の中に、結果をこの卓の判定に使う卓があるか。 */
  private boolean covered(TableSpec table, int version) {
    for (TableSpec base : tables.values()) {
      if (covers(base, table, version)) {
        return true;
      }
    }
    return false;
  }

  /**
   * 基準の卓の結果を、このバージョンのラダーの判定に使えるか（{@link #coveredLadders} の条件）。卓の種類が違えば false。
   *
   * @param base 基準の卓
   * @param table ラダー
   */
  private static boolean covers(TableSpec base, TableSpec table, int version) {
    return TableSpec.kind(base.opponent()) == TableSpec.Opponent.VERSION
        && ladder(table)
        && base.scheduled(version)
        && table.scheduled(version)
        && table.opponentVersion(version) == base.opponentVersion(version)
        && table.seating().equals(base.seating())
        && table.draws() == base.draws()
        && table.score() == base.score();
  }

  /**
   * 確かめの卓の決まりを確かめる。
   *
   * @param caller 確かめの卓を書いた卓の名前
   * @param calling 確かめの卓を書いた卓
   * @param name 確かめの卓の名前
   * @param table 確かめの卓（その名前の卓が無ければ null）
   */
  private static void requireConfirm(String caller, TableSpec calling, String name, TableSpec table) {
    if (table == null) {
      throw new InvalidSetting("tables." + caller + ".confirm", "no table is named " + name);
    }
    String path = "tables." + name + ".";
    if (table.wallSet() != WallSet.FIXED) {
      throw new InvalidSetting(path + "wallSet", "a confirm table plays the same walls for every version");
    }
    if (!(table.stop() instanceof StopSettings.Fixed)) {
      throw new InvalidSetting(path + "stop", "a confirm table plays all its walls, so it stops FIXED");
    }
    if (table.score() != calling.score()) {
      throw new InvalidSetting(
          path + "score",
          "a confirm table scores " + calling.score() + " like the table " + caller + " that calls it, not "
              + table.score());
    }
    if (table.every() != 0) {
      throw new InvalidSetting(
          path + "every", "a confirm table plays only when a duel calls it, so every is 0, not " + table.every());
    }
    TableSpec.Opponent kind = TableSpec.kind(table.opponent());
    if (kind != TableSpec.Opponent.HEURISTIC && kind != TableSpec.Opponent.CHECKPOINT) {
      throw new InvalidSetting(
          path + "opponent",
          "a confirm table plays a frozen player outside the lineage (heuristic or a checkpoint), not "
              + table.opponent());
    }
  }

  /** 止め方が判定で、「良い」なら王者を候補に替える卓か。 */
  private static boolean promotes(TableSpec table) {
    return table.stop() instanceof StopSettings.Test test && test.promote();
  }

  /** ラダー（相手が {@code version:-d} で、止め方が {@code TEST} でない卓）か。 */
  private static boolean ladder(TableSpec table) {
    return TableSpec.kind(table.opponent()) == TableSpec.Opponent.DISTANCE
        && !(table.stop() instanceof StopSettings.Test);
  }

  /** 尺度に入る卓（相手が {@code version:N} か {@code version:-d} の卓）か。 */
  private static boolean rated(TableSpec table) {
    TableSpec.Opponent kind = TableSpec.kind(table.opponent());
    return kind == TableSpec.Opponent.VERSION || kind == TableSpec.Opponent.DISTANCE;
  }
}
