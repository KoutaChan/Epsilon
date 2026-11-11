package com.epsilon.agent.settings;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.RuntimeSettings;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** 設定のテストが共有する設定の形と、木を組むテスト用のヘルパー。いちばん外側の record は、設定が持てる全ての型を 1 回ずつ使う。 */
final class Fixtures {

  private Fixtures() {}

  enum Variant {
    FOUR,
    THREE
  }

  enum Kind {
    DISCARD,
    CALL,
    KAN
  }

  record Run(
      @Key(doc = "四人打ちか三人打ちか", presence = Presence.REQUIRED) Variant variant,
      @Key(doc = "乱数の種", presence = Presence.REQUIRED) long seed,
      @Key(doc = "名前", value = "lineage") String name,
      @Key(doc = "保存先のディレクトリ", value = "runs/x") Path directory) {}

  @Choice(key = "scheme", doc = "記憶の方式")
  sealed interface Memory {
    @Case("NONE")
    record None() implements Memory {}

    @Case("EMA")
    record Ema(
        @Key(doc = "半減期", value = "8.0") @Range(min = 0, minInclusive = false) double halfLife,
        @Key(doc = "始める反復", value = "16") @Range(min = 1) int from)
        implements Memory {}
  }

  @Choice(key = "enabled", doc = "間引くか")
  sealed interface Thinning {
    @Case("false")
    record Off() implements Thinning {}

    @Case("true")
    record On(@Key(doc = "温度", value = "2.0") double tau) implements Thinning {}
  }

  record Fit(
      @Key(doc = "学習率のピーク", value = "0.0012") @Range(min = 0, minInclusive = false) double lrPeak,
      @Key(doc = "学習率の下限", value = "0.00006") @Range(min = 0) double lrFloor,
      @Key(doc = "判断のクラスごとの温度", value = "{DISCARD = 0.01, CALL = 0.02, KAN = 0.02}")
          Map<Kind, @Range(min = 0, max = 1) Double> temperature,
      @Key(doc = "判断のクラス", value = "[DISCARD, CALL]") Set<Kind> classes) {
    Fit {
      if (lrFloor > lrPeak) {
        throw new InvalidSetting("lrFloor", "lrFloor " + lrFloor + " exceeds lrPeak " + lrPeak);
      }
    }
  }

  record Cohort(
      @Key(doc = "始める反復", presence = Presence.REQUIRED) @Range(min = 1) int from,
      @Key(doc = "半荘の数", presence = Presence.REQUIRED) @Range(min = 1) int games) {}

  @Choice(key = "kind", doc = "止め方")
  sealed interface Stop {
    @Case("FIXED")
    record Fixed() implements Stop {}

    @Case("TEST")
    record Sequential(
        @Key(doc = "互角の幅", presence = Presence.REQUIRED) double evenMargin,
        @Key(doc = "有意水準", value = "0.01") @Range(min = 0, max = 1, minInclusive = false, maxInclusive = false)
            double alpha)
        implements Stop {}
  }

  record Table(
      @Key(doc = "相手", presence = Presence.REQUIRED) String opponent,
      @Key(doc = "牌山の数", presence = Presence.REQUIRED) @Range(min = 1) int walls,
      @Key(doc = "止め方", presence = Presence.REQUIRED) Stop stop,
      @Key(doc = "相手のチェックポイント", presence = Presence.ABSENT) Path checkpoint) {}

  record Evaluation(@Key(doc = "卓") Map<String, Table> tables) {}

  record Riichi(@Key(doc = "トークン", presence = Presence.ABSENT, secret = true) String token) {}

  record Flags(
      @Key(doc = "詳しく書く", value = "true") boolean verbose,
      @Key(doc = "倍率", value = "1.5") float scale,
      @Key(doc = "大きさ", value = "[1, 2]") List<Integer> sizes,
      @Key(doc = "始めた時刻", value = "2026-11-12T03:00:00Z") Instant at,
      @Key(doc = "上限", value = "9000000000") long limit,
      @Key(doc = "名前の表", value = "{a = x}") Map<String, String> labels,
      @Key(doc = "入れ子の列", value = "[[1], [2, 3]]") List<List<Integer>> nested) {}

  record Root(
      @Key(doc = "run", change = Change.FIXED) Run run,
      @Key(doc = "学習", change = Change.FIT) Fit fit,
      @Key(doc = "記憶", value = "EMA", change = Change.FIT) Memory memory,
      @Key(doc = "間引き", value = "false", change = Change.COLLECT) Thinning thinning,
      @Key(doc = "コホートの大きさ", value = "[{from = 1, games = 8192}, {from = 9, games = 16384}]", change = Change.COLLECT)
          List<Cohort> cohorts,
      @Key(doc = "フラグ", change = Change.MACHINE) Flags flags,
      @Key(doc = "推論", change = Change.MACHINE) InferenceSettings inference,
      @Key(doc = "卓", change = Change.MACHINE) TableSettings table,
      @Key(doc = "ランタイム", change = Change.MACHINE) RuntimeSettings runtime,
      @Key(doc = "評価", change = Change.SCHEDULE) Evaluation evaluation,
      @Key(doc = "RiichiLab", change = Change.MACHINE) Riichi riichi) {}

  /** FOUR（四人打ち）の層。学習率のピークを変える。 */
  static final Layer VARIANT = Layer.parse("variant FOUR", Layer.Source.BUNDLED, "[fit]\nlrPeak = 0.002\n");

  /** run.variant の値でルールの層を選ぶ。層を持つのは FOUR だけ。 */
  static final BaseLayer BASE =
      new BaseLayer(
          "run.variant",
          name -> {
            if (!name.equals("FOUR")) {
              throw new IllegalArgumentException("no variant layer for " + name);
            }
            return VARIANT;
          });

  /** スモークテスト用のプリセット。評価の卓を 1 つ足し、卓のスレッド数を変える。 */
  static final Layer PRESET =
      Layer.parse(
          "preset smoke",
          Layer.Source.BUNDLED,
          """
          [table]
          workers = 3
          [evaluation.tables.smoke]
          opponent = "heuristic"
          walls = 48
          stop = "FIXED"
          """);

  static Layer file(String text) {
    return Layer.parse("run.toml", Layer.Source.FILE, text);
  }

  static Layer set(String... assignments) {
    return Layer.assignments("--set", List.of(assignments));
  }

  /** run の必須のキー 2 つだけを書いた設定ファイル。 */
  static Layer minimal() {
    return file("[run]\nvariant = \"FOUR\"\nseed = 20261110\n");
  }

  static Settings<Root> bind(Layer... layers) {
    return Settings.bind(Root.class, Arrays.asList(layers));
  }

  /** 設定の結び付けがエラーで止まり、エラーの一覧が期待とちょうど同じになることを確かめる。 */
  static void assertProblems(List<Layer> layers, String... expected) {
    SettingsException thrown = expectThrows(SettingsException.class, () -> Settings.bind(Root.class, layers));
    assertEquals(thrown.problems(), List.of(expected));
  }

  /** 設定の結び付けがエラーで止まることを確かめ、エラーの一覧を返す。 */
  static List<String> problems(List<Layer> layers) {
    return expectThrows(SettingsException.class, () -> Settings.bind(Root.class, layers)).problems();
  }

  static Node.Text text(String value) {
    return new Node.Text(value, "test");
  }

  static Node.Numeral number(String literal) {
    return new Node.Numeral(literal, "test");
  }

  static Node.Bool bool(boolean value) {
    return new Node.Bool(value, "test");
  }

  static Node.Array array(Node... items) {
    return new Node.Array(List.of(items), "test");
  }

  /** キーとノードを交互に並べた引数から表を作る。 */
  static Node.Table table(Object... entries) {
    Map<String, Node> map = new LinkedHashMap<>();
    for (int index = 0; index < entries.length; index += 2) {
      map.put((String) entries[index], (Node) entries[index + 1]);
    }
    return new Node.Table(map, "test");
  }
}
