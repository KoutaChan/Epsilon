package com.epsilon.prism.settings;

import com.epsilon.agent.eval.EvaluationSettings;
import com.epsilon.agent.eval.Seating;
import com.epsilon.agent.eval.Split;
import com.epsilon.agent.eval.StopSettings;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.eval.WallSet;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.ThreePlayerSettings;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの run で、評価の卓の設定を検査する。2×R・王者・基準の卓（version:N）・ラダー（version:-d）・1 半荘ずつの卓は受け付ける。
 * 四人打ちの席の分け方、3 者に分ける席の分け方（1v1v1）、牌山の数が 3 の倍数でない 1 半荘ずつの卓、王者でない相手と打って昇格させる卓、範囲の外の
 * バージョンは、キーのパスと英語の理由を添えて拒む。
 */
public class ThreePlayerTablesTest {

  private static final Variant THREE = Variant.TENHOU_3P;
  private static final String PRECISION = "{kind = \"PRECISION\", error = 0.005, alpha = 0.01, checkEvery = 512}";
  private static final String TEST =
      "{kind = \"TEST\", evenMargin = 0.01, alpha = 0.01, spending = 3.0, checkEvery = 512, promote = true}";

  /** 卓のキーを全て書いた上書きの並び。 */
  private static List<String> table(
      String name, String opponent, String seating, String wallSet, int walls, String stop, int every, int from) {
    String prefix = "evaluation.tables." + name + ".";
    return List.of(
        prefix + "opponent=" + opponent,
        prefix + "seating=" + seating,
        prefix + "draws=SEAT_STREAMS",
        prefix + "wallSet=" + wallSet,
        prefix + "walls=" + walls,
        prefix + "stop=" + stop,
        prefix + "score=RANK",
        prefix + "every=" + every,
        prefix + "from=" + from);
  }

  private static String every(String... splits) {
    return "{kind = \"EVERY_PLACEMENT\", splits = [\"" + String.join("\", \"", splits) + "\"]}";
  }

  private static List<String> rx2() {
    return table("rx2", "heuristic", every("1v2"), "FIXED", 16384, PRECISION, 1, 1);
  }

  private static List<String> champion() {
    return table("champion", "champion", every("1v2", "2v1"), "FRESH", 4096, TEST, 1, 2);
  }

  @SafeVarargs
  private static String[] concat(List<String>... parts) {
    List<String> all = new ArrayList<>();
    for (List<String> part : parts) {
      all.addAll(part);
    }
    return all.toArray(String[]::new);
  }

  @Test
  public void theThreePlayerTablesOfTheDesignAreAccepted() {
    PrismSettings settings =
        ThreePlayerSettings.of(
            concat(
                rx2(),
                champion(),
                table("reference", "version:32", every("1v2", "2v1"), "FIXED", 8192, PRECISION, 8, 40),
                List.of("evaluation.tables.reference.onPromotion=true"),
                table("ladder8", "version:-8", every("1v2", "2v1"), "FRESH", 8192, PRECISION, 8, 16),
                table("ladder32", "version:-32", every("1v2", "2v1"), "FRESH", 8192, PRECISION, 8, 40),
                table("bygame", "heuristic", "BY_GAME", "FIXED", 9, "FIXED", 1, 1)));
    Map<String, TableSpec> tables = settings.evaluation().tables();
    Assert.assertEquals(tables.keySet(), Set.of("rx2", "champion", "reference", "ladder8", "ladder32", "bygame"));

    TableSpec rx2 = tables.get("rx2");
    Assert.assertEquals(TableSpec.kind(rx2.opponent()), TableSpec.Opponent.HEURISTIC);
    Assert.assertEquals(rx2.seating().seating(), Seating.EVERY_PLACEMENT);
    Assert.assertEquals(rx2.seating().splits(THREE), List.of(Split.oneAgainstRest(THREE)));
    Assert.assertEquals(rx2.wallSet(), WallSet.FIXED);
    Assert.assertEquals(rx2.stop(), new StopSettings.Precision(0.005, 0.01, 512));

    TableSpec champion = tables.get("champion");
    Assert.assertEquals(TableSpec.kind(champion.opponent()), TableSpec.Opponent.CHAMPION);
    Assert.assertEquals(champion.seating().splits(THREE).size(), 2);
    Assert.assertEquals(champion.seating().splits(THREE).get(1).seatsOf(0), 2);
    Assert.assertEquals(champion.wallSet(), WallSet.FRESH);
    Assert.assertTrue(champion.stop() instanceof StopSettings.Test test && test.promote(), champion.stop().toString());

    Assert.assertEquals(TableSpec.kind(tables.get("reference").opponent()), TableSpec.Opponent.VERSION);
    Assert.assertEquals(TableSpec.kind(tables.get("ladder8").opponent()), TableSpec.Opponent.DISTANCE);
    Assert.assertEquals(tables.get("bygame").seating().seating(), Seating.BY_GAME);
    Assert.assertEquals(tables.get("bygame").walls(), 9);
  }

  @Test
  public void tablesThatDoNotFitThreePlayersAreRefusedInEnglish() {
    String t = "evaluation.tables.t";
    // 四人打ちの席の分け方と、3 者に分ける席の分け方
    rejects(t + ".seating.splits", "TENHOU_3P has 3", table("t", "heuristic", every("2v2"), "FIXED", 16, "FIXED", 1, 1));
    rejects(t + ".seating.splits", "TENHOU_3P has 3", table("t", "heuristic", every("1v3", "3v1"), "FIXED", 16, "FIXED", 1, 1));
    rejects(t + ".seating.splits", "has 3 slots", table("t", "heuristic", every("1v1v1"), "FIXED", 16, "FIXED", 1, 1));
    // 1 半荘ずつの卓は、牌山の数が 3 の倍数で、全ての牌山を打つ
    rejects(t + ".walls", "multiple of 3", table("t", "heuristic", "BY_GAME", "FIXED", 8, "FIXED", 1, 1));
    rejects(t + ".walls", "multiple of 3", table("t", "heuristic", "BY_GAME", "FIXED", 4096, "FIXED", 1, 1));
    rejects(t + ".stop", "stops FIXED", table("t", "heuristic", "BY_GAME", "FIXED", 1536, PRECISION, 1, 1));
    // 昇格させる卓の相手は王者だけ
    for (String opponent : List.of("heuristic", "version:1", "version:-1")) {
      rejects(
          t + ".opponent",
          "a table that promotes plays the champion",
          table("t", opponent, every("1v2", "2v1"), "FRESH", 4096, TEST, 1, 2));
    }
    // 昇格させる卓は 1 つまで
    rejects(
        "evaluation.tables",
        "at most one table promotes the champion",
        champion(),
        table("again", "champion", every("1v2"), "FRESH", 4096, TEST, 1, 2));
    // 判定で止める卓は、牌山を毎回作る
    rejects(t + ".wallSet", "fresh walls", table("t", "champion", every("1v2", "2v1"), "FIXED", 4096, TEST, 1, 2));
    // バージョンの範囲
    rejects(t + ".opponent", "1 <= N < from", table("t", "version:0", every("1v2"), "FIXED", 8192, PRECISION, 8, 40));
    rejects(t + ".opponent", "1 <= N < from", table("t", "version:40", every("1v2"), "FIXED", 8192, PRECISION, 8, 40));
    rejects(t + ".opponent", "from - d >= 1", table("t", "version:-40", every("1v2"), "FRESH", 8192, PRECISION, 8, 40));
    rejects(t + ".opponent", "from - d >= 1", table("t", "version:-0", every("1v2"), "FRESH", 8192, PRECISION, 8, 40));
  }

  @Test
  public void onlyABaselineTableSaysWhetherItMeasuresThePromotedVersions() {
    String t = "evaluation.tables.t";
    List<String> yes = List.of(t + ".onPromotion=true");
    List<String> no = List.of(t + ".onPromotion=false");
    // version:N の卓には必ず書き、true も false も受け付ける
    for (List<String> written : List.of(yes, no)) {
      PrismSettings settings =
          ThreePlayerSettings.of(
              concat(table("t", "version:32", every("1v2"), "FIXED", 8192, PRECISION, 8, 40), written));
      Assert.assertEquals(
          settings.evaluation().tables().get("t").onPromotion(), Boolean.valueOf(written == yes), written.toString());
    }
    rejects(t + ".onPromotion", "says whether it also measures", table("t", "version:32", every("1v2"), "FIXED", 8192, PRECISION, 8, 40));
    // 基準の卓は、全てのバージョンで同じ牌山を使う
    rejects(t + ".wallSet", "is a baseline", table("t", "version:32", every("1v2"), "FRESH", 8192, PRECISION, 8, 40), yes);
    // ほかの卓では、true でも false でも拒む
    for (List<String> written : List.of(yes, no)) {
      rejects(t + ".onPromotion", "only a version:N table", table("t", "version:-8", every("1v2"), "FRESH", 8192, PRECISION, 8, 16), written);
      rejects(t + ".onPromotion", "only a version:N table", table("t", "heuristic", every("1v2"), "FIXED", 16384, PRECISION, 1, 1), written);
      rejects(t + ".onPromotion", "only a version:N table", table("t", "champion", every("1v2", "2v1"), "FRESH", 4096, TEST, 1, 2), written);
      rejects(t + ".onPromotion", "only a version:N table", table("t", "prism:/runs/v760", every("1v2"), "FIXED", 8192, PRECISION, 8, 8), written);
    }
  }

  /** 三人打ちの系譜で勧める卓の組。2×R・王者・基準の卓と、8 バージョン前・32 バージョン前と打つラダーで、尺度の基準は reference。 */
  private static PrismSettings design() {
    return ThreePlayerSettings.of(
        concat(
            rx2(),
            champion(),
            table("reference", "version:32", every("1v2", "2v1"), "FIXED", 8192, PRECISION, 8, 40),
            List.of("evaluation.tables.reference.onPromotion=true"),
            table("ladder8", "version:-8", every("1v2", "2v1"), "FRESH", 8192, PRECISION, 8, 16),
            table("ladder32", "version:-32", every("1v2", "2v1"), "FRESH", 8192, PRECISION, 8, 40),
            List.of("evaluation.anchor=reference")));
  }

  @Test
  public void theLadderThatMeetsTheBaselineOpponentIsLeftOutOnlyAtVersions40And64() {
    EvaluationSettings evaluation = design().evaluation();
    Assert.assertEquals(evaluation.scheduled(40).keySet(), new TreeSet<>(List.of("champion", "ladder32", "reference", "rx2")));
    Assert.assertEquals(evaluation.coveredLadders("reference", 40), List.of("ladder8"));
    Assert.assertEquals(evaluation.scheduled(64).keySet(), new TreeSet<>(List.of("champion", "ladder8", "reference", "rx2")));
    Assert.assertEquals(evaluation.coveredLadders("reference", 64), List.of("ladder32"));
    // 省く回の前後。v32 は基準の卓が始まる前で、v48・v56・v72 はラダーと基準の卓の相手が重ならない
    Assert.assertEquals(evaluation.scheduled(32).keySet(), new TreeSet<>(List.of("champion", "ladder8", "rx2")));
    Assert.assertEquals(evaluation.coveredLadders("reference", 32), List.of());
    for (int version : List.of(48, 56, 72)) {
      Assert.assertEquals(
          evaluation.scheduled(version).keySet(),
          new TreeSet<>(List.of("champion", "ladder32", "ladder8", "reference", "rx2")),
          "v" + version);
      Assert.assertEquals(evaluation.coveredLadders("reference", version), List.of(), "v" + version);
    }
    // v1〜v400 でラダーを省くのは v40 の ladder8 と v64 の ladder32 だけ。省いたラダーは coveredLadders が返すものと一致する
    List<String> left = new ArrayList<>();
    for (int version = 1; version <= 400; version++) {
      for (Map.Entry<String, TableSpec> entry : evaluation.tables().entrySet()) {
        if (entry.getValue().scheduled(version) && !evaluation.scheduled(version).containsKey(entry.getKey())) {
          left.add(version + entry.getKey());
        }
      }
      List<String> covered = new ArrayList<>();
      for (String ladder : evaluation.coveredLadders("reference", version)) {
        covered.add(version + ladder);
      }
      Assert.assertTrue(left.containsAll(covered), "v" + version + " " + covered);
      Assert.assertEquals(evaluation.coveredLadders("rx2", version), List.of(), "rx2 is not a baseline");
    }
    Assert.assertEquals(left, List.of("40ladder8", "64ladder32"));
  }

  @Test
  public void aSecondBaselineCoversTheLadderThatMeetsItsOwnOpponent() {
    // 2 つ目の基準の卓（v72 から v64 と打つ）を足すと、v72 の ladder8 と v96 の ladder32 も省く
    PrismSettings settings =
        ThreePlayerSettings.of(
            concat(
                table("reference", "version:32", every("1v2", "2v1"), "FIXED", 8192, PRECISION, 8, 40),
                table("second", "version:64", every("1v2", "2v1"), "FIXED", 8192, PRECISION, 8, 72),
                List.of("evaluation.tables.reference.onPromotion=true", "evaluation.tables.second.onPromotion=true"),
                table("ladder8", "version:-8", every("1v2", "2v1"), "FRESH", 8192, PRECISION, 8, 16),
                table("ladder32", "version:-32", every("1v2", "2v1"), "FRESH", 8192, PRECISION, 8, 40)));
    EvaluationSettings evaluation = settings.evaluation();
    Assert.assertEquals(evaluation.scheduled(72).keySet(), new TreeSet<>(List.of("ladder32", "reference", "second")));
    Assert.assertEquals(evaluation.coveredLadders("second", 72), List.of("ladder8"));
    Assert.assertEquals(evaluation.coveredLadders("reference", 72), List.of());
    Assert.assertEquals(evaluation.scheduled(96).keySet(), new TreeSet<>(List.of("ladder8", "reference", "second")));
    Assert.assertEquals(evaluation.coveredLadders("second", 96), List.of("ladder32"));
  }

  @SafeVarargs
  private static void rejects(String path, String because, List<String>... assignments) {
    String[] written = concat(assignments);
    SettingsException failure = Assert.expectThrows(SettingsException.class, () -> ThreePlayerSettings.of(written));
    List<String> named =
        failure.problems().stream().filter(problem -> problem.contains(path) && problem.contains(because)).toList();
    Assert.assertFalse(named.isEmpty(), String.join(" ", written) + " -> " + failure.problems());
    for (String problem : named) {
      Assert.assertTrue(problem.chars().allMatch(c -> c < 0x80), "not English: " + problem);
    }
  }
}
