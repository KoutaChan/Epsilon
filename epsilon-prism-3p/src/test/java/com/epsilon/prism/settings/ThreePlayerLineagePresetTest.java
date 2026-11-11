package com.epsilon.prism.settings;

import com.epsilon.agent.eval.EvaluationSettings;
import com.epsilon.agent.eval.OnWorseSettings;
import com.epsilon.agent.eval.StopSettings;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.ThreePlayerSettings;
import com.epsilon.prism.grp.GrpRefit;
import com.epsilon.prism.grp.GrpSwitch;
import com.epsilon.prism.grp.PhiEntryKind;
import com.epsilon.prism.grp.PhiSettings;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの系譜のプリセット three-player-lineage のテスト。三人打ちの run で設定に反映でき、評価の卓は全てのキーをプリセットに書いている。
 * どの卓も評価の卓の検査を通り、プリセットに重ねた上書きが検査に引っかかれば、キーのパスと英語の理由を添えて拒む。各卓を測るバージョンは
 * 決めた表のとおりで、ラダーの相手が基準の卓の相手と同じバージョンになる回（v40 の ladder8、v64 の ladder32）だけラダーを打たない。
 */
public class ThreePlayerLineagePresetTest {

  /** プリセットの名前。 */
  static final String PRESET = "three-player-lineage";

  /** プリセットにある卓の名前（名前の順）。 */
  static final Set<String> TABLES = new TreeSet<>(List.of("champion", "ladder32", "ladder8", "reference", "rx2"));

  private static final Variant THREE = Variant.TENHOU_3P;
  private static final String PRECISION_512 = "{kind = \"PRECISION\", error = 0.005, alpha = 0.01, checkEvery = 512}";
  private static final String PRECISION_1024 = "{kind = \"PRECISION\", error = 0.005, alpha = 0.01, checkEvery = 1024}";
  private static final String TEST =
      "{kind = \"TEST\", evenMargin = 0.01, alpha = 0.01, spending = 3.0, checkEvery = 512, promote = true}";

  /** 精度で止める卓に書くキー。卓の名前より後ろのパスで書く。 */
  private static final Set<String> PRECISION_KEYS =
      Set.of(
          "opponent", "seating.kind", "seating.splits", "draws", "wallSet", "walls", "stop.kind", "stop.error",
          "stop.alpha", "stop.checkEvery", "score", "every", "from");

  /** 判定で止める卓に書くキー。卓の名前より後ろのパスで書く。 */
  private static final Set<String> TEST_KEYS =
      Set.of(
          "opponent", "seating.kind", "seating.splits", "draws", "wallSet", "walls", "stop.kind", "stop.evenMargin",
          "stop.alpha", "stop.spending", "stop.checkEvery", "stop.promote", "score", "every", "from");

  /** 卓のキーを全て書いた上書きの並び。 */
  private static List<String> table(
      String name, String opponent, String splits, String wallSet, int walls, String stop, int every, int from) {
    String prefix = "evaluation.tables." + name + ".";
    return List.of(
        prefix + "opponent=" + opponent,
        prefix + "seating={kind = \"EVERY_PLACEMENT\", splits = [" + splits + "]}",
        prefix + "draws=SEAT_STREAMS",
        prefix + "wallSet=" + wallSet,
        prefix + "walls=" + walls,
        prefix + "stop=" + stop,
        prefix + "score=RANK",
        prefix + "every=" + every,
        prefix + "from=" + from);
  }

  @SafeVarargs
  private static String[] concat(List<String>... parts) {
    List<String> all = new ArrayList<>();
    for (List<String> part : parts) {
      all.addAll(part);
    }
    return all.toArray(String[]::new);
  }

  /** 期待する卓を、プリセットを使わずに上書きだけで書いた評価のセクション。 */
  private static EvaluationSettings design() {
    String both = "\"1v2\", \"2v1\"";
    return ThreePlayerSettings.of(
            concat(
                table("rx2", "heuristic", "\"1v2\"", "FIXED", 16384, PRECISION_1024, 1, 1),
                table("champion", "champion", both, "FRESH", 4096, TEST, 1, 2),
                table("reference", "version:32", both, "FIXED", 8192, PRECISION_512, 8, 40),
                List.of("evaluation.tables.reference.onPromotion=true"),
                table("ladder8", "version:-8", both, "FRESH", 8192, PRECISION_512, 8, 16),
                table("ladder32", "version:-32", both, "FRESH", 8192, PRECISION_512, 8, 40),
                List.of("evaluation.anchor=reference")))
        .evaluation();
  }

  private static Settings<PrismSettings> preset(String... assignments) {
    return ThreePlayerSettings.bind(List.of(PRESET), assignments);
  }

  // 設定に反映でき、全ての卓のキーがそろう

  @Test
  public void thePresetBindsOnAThreePlayerRunWithEveryTableKeyWritten() {
    Settings<PrismSettings> bound = preset();
    PrismSettings settings = bound.value();
    Assert.assertEquals(settings.run().variant(), THREE);
    Assert.assertEquals(settings.network(), new NetworkSettings.V2(2));
    Assert.assertTrue(settings.phi() instanceof PhiSettings.SelfGrp, settings.phi().toString());
    PhiSettings.SelfGrp phi = (PhiSettings.SelfGrp) settings.phi();
    Assert.assertEquals(phi.start().kind(), PhiEntryKind.SCORE_RANK);
    Assert.assertEquals(phi.grp().refit(), GrpRefit.SCHEDULE);
    Assert.assertEquals(phi.grp().switching(), GrpSwitch.AUTO);
    Assert.assertEquals(phi.grp().halfLife(), 1.0);
    Assert.assertEquals(phi.grp().trustGames(), 8192);

    EvaluationSettings evaluation = settings.evaluation();
    Assert.assertEquals(evaluation.anchor(), "reference");
    Assert.assertEquals(evaluation.alarmZ(), 3.0);
    Assert.assertEquals(evaluation.dropMargin(), 0.01);
    Assert.assertEquals(evaluation.onWorse(), new OnWorseSettings.Report(2));
    Assert.assertEquals(evaluation.tables().keySet(), TABLES);
    // 卓の値は、上書きだけで書いた期待の卓と同じ
    Assert.assertEquals(evaluation.tables(), design().tables());
    Assert.assertEquals(evaluation.tables().get("rx2").stop(), new StopSettings.Precision(0.005, 0.01, 1024));
    Assert.assertEquals(evaluation.tables().get("reference").onPromotion(), Boolean.TRUE);

    // 評価のセクションのキーは全てプリセットに書いてある。既定値やほかの層から来たキーは無い
    Map<String, Set<String>> written = new TreeMap<>();
    for (Map.Entry<String, Node> leaf : bound.effective().entrySet()) {
      String path = leaf.getKey();
      if (!path.startsWith("evaluation.")) {
        continue;
      }
      Assert.assertTrue(leaf.getValue().origin().startsWith("preset " + PRESET), path + " from " + leaf.getValue().origin());
      if (path.startsWith("evaluation.tables.")) {
        String rest = path.substring("evaluation.tables.".length());
        int dot = rest.indexOf('.');
        written.computeIfAbsent(rest.substring(0, dot), name -> new TreeSet<>()).add(rest.substring(dot + 1));
      }
    }
    for (String head : List.of("evaluation.anchor", "evaluation.alarmZ", "evaluation.dropMargin")) {
      Assert.assertTrue(bound.effective().containsKey(head), head);
    }
    Assert.assertEquals(written.keySet(), TABLES);
    Set<String> reference = new TreeSet<>(PRECISION_KEYS);
    reference.add("onPromotion");
    Assert.assertEquals(written.get("rx2"), PRECISION_KEYS);
    Assert.assertEquals(written.get("champion"), TEST_KEYS);
    Assert.assertEquals(written.get("reference"), reference);
    Assert.assertEquals(written.get("ladder8"), PRECISION_KEYS);
    Assert.assertEquals(written.get("ladder32"), PRECISION_KEYS);
  }

  // 評価の卓の検査

  @Test
  public void theTablesPassTheSectionFiveThreeChecks() {
    EvaluationSettings evaluation = preset().value().evaluation();
    Map<String, Integer> looks = new TreeMap<>();
    int promoting = 0;
    for (Map.Entry<String, TableSpec> entry : evaluation.tables().entrySet()) {
      String name = entry.getKey();
      TableSpec table = entry.getValue();
      switch (TableSpec.kind(table.opponent())) {
        case VERSION -> Assert.assertTrue(table.opponentVersion(table.from()) < table.from(), name);
        case DISTANCE -> {
          // version:-d の d は 1 以上で、最初のバージョンの相手は 1 以上
          Assert.assertTrue(table.from() - table.opponentVersion(table.from()) >= 1, name);
          Assert.assertTrue(table.opponentVersion(table.from()) >= 1, name);
        }
        case HEURISTIC, CHAMPION, CHECKPOINT -> {}
      }
      int checkEvery = table.stop().checkEvery();
      Assert.assertTrue(checkEvery >= 512, name + " checkEvery " + checkEvery);
      Assert.assertEquals(table.walls() % checkEvery, 0, name);
      Assert.assertTrue(table.walls() / checkEvery <= TableSpec.MAX_LOOKS, name);
      looks.put(name, table.walls() / checkEvery);
      if (table.stop() instanceof StopSettings.Test test && test.promote()) {
        promoting++;
        Assert.assertEquals(name, "champion");
      }
    }
    Assert.assertEquals(promoting, 1);
    Assert.assertEquals(looks, Map.of("champion", 8, "ladder32", 16, "ladder8", 16, "reference", 16, "rx2", 16));
    // 最初に測るバージョンでの相手。基準の卓は v32、8 バージョン前と打つ卓は v16 で v8、32 バージョン前と打つ卓は v40 で v8
    Assert.assertEquals(evaluation.tables().get("reference").opponentVersion(40), 32);
    Assert.assertEquals(evaluation.tables().get("ladder8").opponentVersion(16), 8);
    Assert.assertEquals(evaluation.tables().get("ladder32").opponentVersion(40), 8);
  }

  @Test
  public void overridesThatBreakTheChecksAreRefusedInEnglish() {
    String t = "evaluation.tables.";
    // version:N は N < from
    rejects(t + "reference.opponent", "1 <= N < from", t + "reference.from=32");
    Assert.assertEquals(preset(t + "reference.from=33").value().evaluation().tables().get("reference").from(), 33);
    // version:-d は from - d >= 1
    rejects(t + "ladder8.opponent", "from - d >= 1", t + "ladder8.from=8");
    rejects(t + "ladder32.opponent", "from - d >= 1", t + "ladder32.from=32");
    Assert.assertEquals(preset(t + "ladder32.from=33").value().evaluation().tables().get("ladder32").from(), 33);
    // 牌山の数は判定の間隔の倍数で、判定の時点は 32 回まで
    rejects(t + "rx2.walls", "is not a multiple of checkEvery 1024", t + "rx2.walls=16000");
    rejects(t + "rx2.walls", "exceeds 32 looks", t + "rx2.walls=33792");
    Assert.assertEquals(preset(t + "rx2.walls=32768").value().evaluation().tables().get("rx2").walls(), 32768);
    rejects(t + "champion.walls", "is not a multiple of checkEvery 512", t + "champion.walls=4000");
    // 判定の間隔は 512 以上
    rejects(t + "champion.stop.checkEvery", "", t + "champion.stop.checkEvery=256");
    // 王者を替える卓は 1 つまで
    rejects(
        "evaluation.tables",
        "at most one table promotes the champion",
        concat(table("again", "champion", "\"1v2\"", "FRESH", 4096, TEST, 1, 2)));
    // 基準の卓には、昇格したバージョンも測るかを書く必要がある。プリセットの値は消せないので、別の基準の卓を足して確かめる
    rejects(
        t + "second.onPromotion",
        "says whether it also measures",
        concat(table("second", "version:16", "\"1v2\"", "FIXED", 8192, PRECISION_512, 8, 24)));
  }

  /** 尺度の辺には上限を掛けないので、上限のキーは評価の節に無い。 */
  @Test
  public void aLinearCapIsAnUnknownKey() {
    rejects("evaluation.linearCap", "Unknown setting", "evaluation.linearCap=0.05");
  }

  private static void rejects(String path, String because, String... assignments) {
    SettingsException failure = Assert.expectThrows(SettingsException.class, () -> preset(assignments));
    List<String> named =
        failure.problems().stream().filter(problem -> problem.contains(path) && problem.contains(because)).toList();
    Assert.assertFalse(named.isEmpty(), String.join(" ", assignments) + " -> " + failure.problems());
    for (String problem : named) {
      Assert.assertTrue(problem.chars().allMatch(c -> c < 0x80), "not English: " + problem);
    }
  }

  // 各卓を測るバージョン

  @Test
  public void eachTableIsMeasuredAtTheVersionsOfTheDesign() {
    EvaluationSettings evaluation = preset().value().evaluation();
    Map<Integer, List<String>> expected = new TreeMap<>();
    expected.put(1, List.of("rx2"));
    expected.put(2, List.of("champion", "rx2"));
    expected.put(8, List.of("champion", "rx2"));
    expected.put(16, List.of("champion", "ladder8", "rx2"));
    expected.put(24, List.of("champion", "ladder8", "rx2"));
    expected.put(32, List.of("champion", "ladder8", "rx2"));
    // v40: 8 前の相手 v32 が基準の卓の相手と同じなので、ladder8 は打たず基準の卓の結果を使う
    expected.put(40, List.of("champion", "ladder32", "reference", "rx2"));
    expected.put(41, List.of("champion", "rx2"));
    expected.put(48, List.of("champion", "ladder32", "ladder8", "reference", "rx2"));
    expected.put(56, List.of("champion", "ladder32", "ladder8", "reference", "rx2"));
    // v64: 32 前の相手 v32 が基準の卓の相手と同じなので、ladder32 は打たない
    expected.put(64, List.of("champion", "ladder8", "reference", "rx2"));
    expected.put(72, List.of("champion", "ladder32", "ladder8", "reference", "rx2"));
    for (Map.Entry<Integer, List<String>> entry : expected.entrySet()) {
      int version = entry.getKey();
      Assert.assertEquals(new ArrayList<>(evaluation.scheduled(version).keySet()), entry.getValue(), "v" + version);
    }
    Assert.assertEquals(evaluation.coveredLadders("reference", 40), List.of("ladder8"));
    Assert.assertEquals(evaluation.coveredLadders("reference", 64), List.of("ladder32"));
    Assert.assertEquals(evaluation.coveredLadders("reference", 48), List.of());
    // 相手のバージョン
    Assert.assertEquals(evaluation.tables().get("ladder8").opponentVersion(48), 40);
    Assert.assertEquals(evaluation.tables().get("ladder32").opponentVersion(48), 16);
    Assert.assertEquals(evaluation.tables().get("reference").opponentVersion(64), 32);

    // v64 までに測る予定のバージョン。省く回も含む
    Map<String, TableSpec> tables = evaluation.tables();
    Assert.assertEquals(tables.get("rx2").versions(64).size(), 64);
    Assert.assertEquals(tables.get("rx2").versions(64).getFirst(), 1);
    Assert.assertEquals(tables.get("champion").versions(64).size(), 63);
    Assert.assertEquals(tables.get("champion").versions(64).getFirst(), 2);
    Assert.assertEquals(tables.get("reference").versions(64), List.of(40, 48, 56, 64));
    Assert.assertEquals(tables.get("ladder8").versions(64), List.of(16, 24, 32, 40, 48, 56, 64));
    Assert.assertEquals(tables.get("ladder32").versions(64), List.of(40, 48, 56, 64));

    // v1〜v1000 でラダーを省くのは v40 の ladder8 と v64 の ladder32 だけ。省いたラダーは、基準の卓の結果を使うラダーと一致する
    List<String> left = new ArrayList<>();
    List<String> covered = new ArrayList<>();
    for (int version = 1; version <= 1000; version++) {
      for (Map.Entry<String, TableSpec> entry : tables.entrySet()) {
        if (entry.getValue().scheduled(version) && !evaluation.scheduled(version).containsKey(entry.getKey())) {
          left.add(version + entry.getKey());
        }
      }
      for (String ladder : evaluation.coveredLadders("reference", version)) {
        covered.add(version + ladder);
      }
    }
    Assert.assertEquals(left, List.of("40ladder8", "64ladder32"));
    Assert.assertEquals(covered, left);
  }
}
