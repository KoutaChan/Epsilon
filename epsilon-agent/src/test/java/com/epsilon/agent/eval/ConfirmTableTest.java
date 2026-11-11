package com.epsilon.agent.eval;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.testng.annotations.Test;

/**
 * 確かめの卓のキーの検査。王者を替える卓が書いた確かめの卓は、全てのバージョンで同じ牌山を全て打ち、呼ぶ卓と同じ量で判定し、予定では打たず、
 * 系譜の外の打ち手を相手にして、幅を書く。外れた設定は、キーのパスと英語の理由を添えて拒む。
 */
public class ConfirmTableTest {

  private static final String TEST =
      "{kind = \"TEST\", evenMargin = 0.004, alpha = 0.01, spending = 3.0, checkEvery = 1536, promote = true}";
  private static final String PRECISION = "{kind = \"PRECISION\", error = 0.005, alpha = 0.01, checkEvery = 512}";
  private static final String BOTH = "{kind = \"EVERY_PLACEMENT\", splits = [\"1v3\", \"3v1\"]}";
  private static final String ONE = "{kind = \"EVERY_PLACEMENT\", splits = [\"1v3\"]}";
  private static final String V760 = "prism:/runs/prism-v2b/snapshots/00760";

  /** 卓のキーを全て書いた上書き。 */
  private static List<String> table(
      String name, String opponent, String seating, String wallSet, int walls, String stop, String score, int every,
      int from) {
    String prefix = "evaluation.tables." + name + ".";
    return List.of(
        prefix + "opponent=" + opponent,
        prefix + "seating=" + seating,
        prefix + "draws=SEAT_STREAMS",
        prefix + "wallSet=" + wallSet,
        prefix + "walls=" + walls,
        prefix + "stop=" + stop,
        prefix + "score=" + score,
        prefix + "every=" + every,
        prefix + "from=" + from);
  }

  private static List<String> champion() {
    return table("champion", "champion", BOTH, "FRESH", 12288, TEST, "UTILITY", 10, 10);
  }

  private static List<String> outside() {
    return table("outside", V760, ONE, "FIXED", 8192, "FIXED", "UTILITY", 0, 0);
  }

  private static List<String> set(String key, String value) {
    return List.of("evaluation.tables." + key + "=" + value);
  }

  @SafeVarargs
  private static EvaluationSettings bind(List<String>... parts) {
    List<Layer> layers = new ArrayList<>();
    for (List<String> part : parts) {
      for (String assignment : part) {
        layers.add(Layer.assignments("test", List.of(assignment)));
      }
    }
    return Settings.section(EvaluationSettings.class, "evaluation", layers).value();
  }

  @Test
  public void theChampionTableConfirmsWithAFrozenPlayerOnFixedWalls() {
    EvaluationSettings evaluation =
        bind(
            champion(),
            set("champion.confirm", "outside"),
            outside(),
            set("outside.confirmMargin", "0.004"));
    assertEquals(evaluation.promoting(), Optional.of("champion"));
    assertEquals(evaluation.tables().get("champion").confirm(), "outside");
    assertNull(evaluation.tables().get("champion").confirmMargin());
    assertEquals(evaluation.tables().get("outside").confirmMargin(), Double.valueOf(0.004));
    assertNull(evaluation.tables().get("outside").confirm());
    // 確かめの卓は予定では打たない
    for (int version = 1; version <= 100; version++) {
      assertFalse(evaluation.scheduled(version).containsKey("outside"), "v" + version);
    }
    // ルールベースの打ち手とほかの系列のチェックポイントも、系譜の外の打ち手として受ける
    for (String opponent : List.of("heuristic", "major:/models/major/macro614", "nano:/models/nano/ppo17")) {
      EvaluationSettings other =
          bind(
              champion(),
              set("champion.confirm", "outside"),
              outside(),
              set("outside.opponent", opponent),
              set("outside.confirmMargin", "0"));
      assertEquals(other.tables().get("outside").opponent(), opponent);
    }
    // 王者を替える卓が無ければ空
    assertEquals(bind(outside()).promoting(), Optional.empty());
  }

  @Test
  public void confirmTablesThatBreakTheRulesAreRefusedInEnglish() {
    String outside = "evaluation.tables.outside.";
    List<String> calls = set("champion.confirm", "outside");
    List<String> margin = set("outside.confirmMargin", "0.004");
    // 確かめの卓が無い
    rejects("evaluation.tables.champion.confirm", "no table is named nowhere", champion(), set("champion.confirm", "nowhere"));
    // 牌山は全てのバージョンで同じ、止め方は固定
    rejects(outside + "wallSet", "the same walls for every version", champion(), calls, outside(), margin, set("outside.wallSet", "FRESH"));
    rejects(outside + "stop", "stops FIXED", champion(), calls, outside(), margin, set("outside.stop", PRECISION));
    // 呼ぶ卓と同じ量で判定する
    rejects(outside + "score", "scores UTILITY like the table champion", champion(), calls, outside(), margin, set("outside.score", "RANK"));
    // 予定では打たない
    rejects(outside + "every", "every is 0", champion(), calls, outside(), margin, set("outside.every", "10"));
    // 系譜の外の凍結した打ち手を相手にする
    rejects(outside + "opponent", "frozen player outside the lineage", champion(), calls, outside(), margin, set("outside.opponent", "champion"));
    // 幅は確かめの卓に必ず書き、ほかの卓には書かない
    rejects(outside + "confirmMargin", "needs a confirmMargin", champion(), calls, outside());
    rejects(outside + "confirmMargin", "only a confirm table", champion(), outside(), margin);
    rejects("evaluation.tables.champion.confirmMargin", "only a confirm table", champion(), calls, outside(), margin, set("champion.confirmMargin", "0.004"));
    rejects(outside + "confirmMargin", "", champion(), calls, outside(), set("outside.confirmMargin", "-0.001"));
    // 確かめの卓を書けるのは王者を替える卓だけ
    rejects(
        "evaluation.tables.rx3.confirm",
        "only a table that promotes the champion",
        champion(),
        table("rx3", "heuristic", ONE, "FIXED", 16384, PRECISION, "UTILITY", 1, 1),
        set("rx3.confirm", "outside"),
        outside(),
        margin);
    // 順位で判定するなら、幅は 0.5 まで
    rejects(
        outside + "confirmMargin",
        "must not exceed 0.5",
        champion(),
        set("champion.score", "RANK"),
        calls,
        outside(),
        set("outside.score", "RANK"),
        set("outside.confirmMargin", "0.6"));
  }

  @SafeVarargs
  private static void rejects(String path, String because, List<String>... parts) {
    SettingsException failure = expectThrows(SettingsException.class, () -> bind(parts));
    List<String> named =
        failure.problems().stream().filter(problem -> problem.contains(path) && problem.contains(because)).toList();
    assertFalse(named.isEmpty(), path + " " + because + " -> " + failure.problems());
    for (String problem : named) {
      assertTrue(problem.chars().allMatch(c -> c < 0x80), "not English: " + problem);
    }
  }
}
