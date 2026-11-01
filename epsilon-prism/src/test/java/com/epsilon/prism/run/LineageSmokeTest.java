package com.epsilon.prism.run;

import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.rows.CohortWriter;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 網の系譜の煙（CPU）: score-rank の Φ・1 コホート 1 局で train を 2 反復回し、止めてから 1 反復を再開する。版 1（白紙の網）で集めて版 2 を当て、
 * 版 2・3 で集めて版 3・4 を当てる。帳簿の補正・根の行・方策の記憶（ema）・学習の網の評価も通る。行・版・記録・state.json がそろい、
 * 再開は書き終えた記録を書き直さない。再開でも設定の知らない鍵は拒む。
 */
public class LineageSmokeTest {

  @Test(timeOut = 1_800_000)
  public void trainsCollectsAndResumesOnCpu() throws Exception {
    Path run = Files.createTempDirectory("prism-lineage");
    try {
      Assert.assertEquals(Lineage.run(run, 2, EpsilonSettings.of(settings())), 0);
      Assert.assertEquals(state(run).get("next").getAsInt(), 3);
      // 再開: 設定の値は使わないが、なくなった鍵は拒む
      Assert.assertThrows(
          IllegalArgumentException.class,
          () -> Lineage.run(run, 1, EpsilonSettings.of(Map.of("epsilon.prism.trainRoots", "true"))));
      // 再開: 設定は run の config.json から読む（版 3 で集めて版 4 を当てる）
      Assert.assertEquals(Lineage.run(run, 1, EpsilonSettings.defaults()), 0);
      JsonObject state = state(run);
      Assert.assertEquals(state.get("next").getAsInt(), 4);
      Assert.assertTrue(state.has("ledgerR2"));
      List<JsonObject> iterations = lines(run.resolve("reports/iterations.jsonl"));
      List<JsonObject> evaluations = lines(run.resolve("reports/evaluations.jsonl"));
      Assert.assertEquals(iterations.size(), 3);
      Assert.assertEquals(evaluations.size(), 3);
      for (int t = 1; t <= 3; t++) {
        JsonObject iteration = iterations.get(t - 1);
        Assert.assertEquals(iteration.get("iteration").getAsInt(), t);
        Assert.assertTrue(iteration.get("collection").isJsonObject(), "cohort " + t + " was collected");
        Assert.assertTrue(iteration.get("totalRows").getAsLong() > 0);
        for (String key : List.of("memory", "ledger", "filter", "fitSingle", "average", "acceptance"))
          Assert.assertTrue(iteration.has(key), "iteration " + t + " records " + key);
        Assert.assertTrue(
            Files.exists(CohortWriter.directory(run, t).resolve(CohortWriter.MANIFEST)),
            "rows-v2 of cohort " + t);
        Path version = snapshot(run, t + 1);
        Assert.assertEquals(
            PrismNetwork.readShape(version.resolve("m0")), PrismShape.standard(2), "version " + (t + 1) + " plays");
        Assert.assertEquals(
            PrismNetwork.readShape(version.resolve("iterate")), PrismShape.standard(2), "version " + (t + 1) + " keeps θ");
        JsonObject evaluation = evaluations.get(t - 1);
        Assert.assertEquals(evaluation.get("version").getAsInt(), t + 1);
        Assert.assertTrue(evaluation.has("heuristic") && evaluation.has("iterate"));
      }
      // K′ の元は前の反復で教師に掛けた補正の保留の R²（版 1 の当てはめは R² = 0 で K′ = K）
      JsonObject first = iterations.get(0).getAsJsonObject("filter");
      Assert.assertEquals(first.get("r2").getAsDouble(), 0.0);
      Assert.assertEquals(first.get("gain").getAsDouble(), first.get("scheduledGain").getAsDouble());
      for (int t = 2; t <= 3; t++) {
        JsonObject ledger = iterations.get(t - 2).getAsJsonObject("ledger");
        double r2 = ledger.getAsJsonObject("all").get("r2").getAsDouble();
        double applied = ledger.get("rows").getAsLong() > 0 && !Double.isNaN(r2) ? r2 : 0;
        Assert.assertEquals(
            iterations.get(t - 1).getAsJsonObject("filter").get("r2").getAsDouble(), applied);
      }
      Assert.assertTrue(
          iterations.get(0).getAsJsonObject("fitSingle").get("steps").getAsInt() > 0, "version 2 was fitted");
      Assert.assertFalse(
          Arrays.equals(weights(snapshot(run, 1)), weights(snapshot(run, 2))), "version 2 learned nothing");
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  /**
   * CPU で回る小さな系譜の設定（本番の学習器の形に、1 コホート 1 局。標準の幅の網は CPU の 1 スレッドでは遅く、収集の判断の大半は分岐なので、
   * ラベル行・ロールインの率と候補集合は本番より小さくする）。
   */
  private static Map<String, String> settings() {
    String[] pairs = {
      "phi", "score-rank", "gamesPerCohort", "1", "gamesSchedule", "", "iterations", "10",
      "collectDevices", "cpu", "learnerDevice", "cpu", "maxActiveGames", "8", "advanceWorkers", "2",
      "encoders", "2", "slots", "1", "readyBatches", "2", "inferBatch", "256",
      "batchWaitMicros", "1000", "rowWriters", "1", "labelRateDiscardEarly", "0.003",
      "labelRateDiscardMid", "0.003", "labelRateDiscardLate", "0.003", "labelRateCommit", "0.03",
      "maxCandidates", "4", "rollinRateDiscard", "0.001", "rollinRateCommit", "0.015", "rhoBranch", "0.2",
      "averaging", "ema", "averageFrom", "2", "evaluateIterate", "true", "batchSize", "16",
      "decodeThreads", "2", "evalEvery", "1", "evalGames", "1"
    };
    Map<String, String> values = new LinkedHashMap<>();
    for (int i = 0; i < pairs.length; i += 2) values.put("epsilon.prism." + pairs[i], pairs[i + 1]);
    return values;
  }

  private static JsonObject state(Path run) throws Exception {
    return JsonParser.parseString(Files.readString(run.resolve("state.json"))).getAsJsonObject();
  }

  private static List<JsonObject> lines(Path path) throws Exception {
    List<JsonObject> result = new ArrayList<>();
    for (String line : Files.readAllLines(path, StandardCharsets.UTF_8))
      if (!line.isBlank()) result.add(JsonParser.parseString(line).getAsJsonObject());
    return result;
  }

  private static Path snapshot(Path run, int version) {
    return run.resolve("snapshots").resolve(String.format("%05d", version));
  }

  private static byte[] weights(Path snapshot) throws Exception {
    return Files.readAllBytes(snapshot.resolve("m0").resolve("weights.bin"));
  }
}
