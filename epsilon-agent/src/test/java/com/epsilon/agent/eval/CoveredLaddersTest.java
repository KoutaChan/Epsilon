package com.epsilon.agent.eval;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;

import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Settings;
import java.util.ArrayList;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 基準の卓の結果でラダーを判定する回（{@link EvaluationSettings#scheduled}・{@link EvaluationSettings#coveredLadders}）。ラダーの相手の
 * バージョンが基準の卓の相手と同じ回に、席の決め方（分け方を含む）・ツモの順・判定に使う量も基準の卓と同じラダーは打たず、基準の卓の結果で
 * 判定する。どれかが違うラダーは打つ。牌山の集め方と数、止め方は違ってよい。
 */
public class CoveredLaddersTest {

  private static final String BOTH = "{kind = \"EVERY_PLACEMENT\", splits = [\"1v3\", \"3v1\"]}";
  private static final String REVERSED = "{kind = \"EVERY_PLACEMENT\", splits = [\"3v1\", \"1v3\"]}";
  private static final String ONE = "{kind = \"EVERY_PLACEMENT\", splits = [\"1v3\"]}";
  private static final String PRECISION = "{kind = \"PRECISION\", error = 0.005, alpha = 0.01, checkEvery = 512}";
  private static final String COARSE = "{kind = \"PRECISION\", error = 0.01, alpha = 0.01, checkEvery = 1024}";

  /** 卓のキーを全て書いた上書き。 */
  private static List<String> table(
      String name, String opponent, String seating, String draws, String wallSet, int walls, String stop, String score,
      int every, int from) {
    String prefix = "evaluation.tables." + name + ".";
    return List.of(
        prefix + "opponent=" + opponent,
        prefix + "seating=" + seating,
        prefix + "draws=" + draws,
        prefix + "wallSet=" + wallSet,
        prefix + "walls=" + walls,
        prefix + "stop=" + stop,
        prefix + "score=" + score,
        prefix + "every=" + every,
        prefix + "from=" + from);
  }

  /** v40 から 8 バージョンごとに v32 と打つ基準の卓。 */
  private static List<String> baseline(String name, String seating) {
    List<String> all =
        new ArrayList<>(table(name, "version:32", seating, "SEAT_STREAMS", "FIXED", 8192, PRECISION, "RANK", 8, 40));
    all.add("evaluation.tables." + name + ".onPromotion=true");
    return all;
  }

  /** v16 から 8 バージョンごとに 8 つ前と打つラダー。v40 で、相手が基準の卓の相手（v32）と同じになる。 */
  private static List<String> ladder(String name, String seating, String draws, String stop, String score) {
    return table(name, "version:-8", seating, draws, "FRESH", 8192, stop, score, 8, 16);
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
  public void aLadderOnTheBaselinePlanIsJudgedByTheBaseline() {
    EvaluationSettings evaluation =
        bind(
            baseline("reference", BOTH),
            table("ladder8", "version:-8", BOTH, "SEAT_STREAMS", "FRESH", 4096, COARSE, "RANK", 8, 16));
    assertFalse(evaluation.scheduled(40).containsKey("ladder8"));
    assertTrue(evaluation.scheduled(40).containsKey("reference"));
    assertEquals(evaluation.coveredLadders("reference", 40), List.of("ladder8"));
    // 相手が重ならない回は打ち、基準の卓の結果では判定しない
    for (int version : List.of(32, 48)) {
      assertTrue(evaluation.scheduled(version).containsKey("ladder8"), "v" + version);
      assertEquals(evaluation.coveredLadders("reference", version), List.of(), "v" + version);
    }
    // 基準の卓でない卓と、無い卓の名前では空
    assertEquals(evaluation.coveredLadders("ladder8", 40), List.of());
    assertEquals(evaluation.coveredLadders("nowhere", 40), List.of());
  }

  @Test
  public void aLadderOnAnotherPlanIsPlayed() {
    // 分け方が違う
    playedAtForty(ladder("ladder8", ONE, "SEAT_STREAMS", PRECISION, "RANK"));
    // 分け方の順が違う（割り当ての番号の振り方が違う）
    playedAtForty(ladder("ladder8", REVERSED, "SEAT_STREAMS", PRECISION, "RANK"));
    // 牌山ごとに 1 半荘打つ
    playedAtForty(ladder("ladder8", "BY_GAME", "SEAT_STREAMS", "FIXED", "RANK"));
    // ツモの順が違う
    playedAtForty(ladder("ladder8", BOTH, "WALL_ORDER", PRECISION, "RANK"));
    // 判定に使う量が違う
    playedAtForty(ladder("ladder8", BOTH, "SEAT_STREAMS", PRECISION, "UTILITY"));
  }

  /** 基準の卓と並べたラダーを、相手が重なる v40 に打ち、基準の卓の結果では判定しないことを確かめる。 */
  private static void playedAtForty(List<String> ladder) {
    EvaluationSettings evaluation = bind(baseline("reference", BOTH), ladder);
    assertTrue(evaluation.scheduled(40).containsKey("ladder8"), ladder.toString());
    assertEquals(evaluation.coveredLadders("reference", 40), List.of(), ladder.toString());
  }

  @Test
  public void eachLadderIsJudgedByTheBaselineOnItsPlan() {
    // 相手が同じで分け方の違う基準の卓 2 つと、v40 で相手が重なるラダー 3 つ。ラダーは、席の決め方・ツモの順・判定に使う量が同じ基準の卓の
    // 結果だけで判定する
    EvaluationSettings evaluation =
        bind(
            baseline("reference", BOTH),
            baseline("single", ONE),
            ladder("both8", BOTH, "SEAT_STREAMS", PRECISION, "RANK"),
            ladder("single8", ONE, "SEAT_STREAMS", PRECISION, "RANK"),
            ladder("wall8", ONE, "WALL_ORDER", PRECISION, "RANK"));
    assertEquals(evaluation.coveredLadders("reference", 40), List.of("both8"));
    assertEquals(evaluation.coveredLadders("single", 40), List.of("single8"));
    assertEquals(new ArrayList<>(evaluation.scheduled(40).keySet()), List.of("reference", "single", "wall8"));
    assertEquals(
        new ArrayList<>(evaluation.scheduled(48).keySet()), List.of("both8", "reference", "single", "single8", "wall8"));
  }
}
