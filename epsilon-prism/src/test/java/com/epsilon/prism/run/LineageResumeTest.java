package com.epsilon.prism.run;

import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.StopSignal.SafePoint;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.nio.file.attribute.FileTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 系譜の止め方と再開（CPU）: score-rank の Φ・1 コホート 1 局・評価は 3xH（1 局）と x1n3（3×Heuristic の卓 2 局）で、3 つの安全点（封印の後・
 * 反復の後・評価 1 つの後）で止めては起動し直し、落ちた跡（state.json の前の落ち・x1n3 の途中・書きかけの記録の行・集めかけのコホート）と .failed の
 * バージョンを置いてから最後まで回す。どこで止めても、記録と評価はバージョンごとにちょうど 1 回で、測った評価は測り直さない。
 */
public class LineageResumeTest {

  @Test(timeOut = 2_400_000)
  public void stopsAtSafePointsAndResumesWithoutDuplicates() throws Exception {
    Path run = Files.createTempDirectory("prism-resume");
    Path evaluations = run.resolve(EvaluationOutputs.DIRECTORY);
    try {
      // 封印の後: コホート 1 は封印済み、バージョン 2 はまだない
      Assert.assertEquals(Lineage.run(run, Integer.MAX_VALUE, settings(), StopSignal.at(SafePoint.SEALED, 1)), 0);
      Assert.assertEquals(next(run), 1);
      Assert.assertTrue(Files.exists(CohortWriter.directory(run, 1).resolve(CohortWriter.MANIFEST)));
      Assert.assertFalse(Files.exists(run.resolve("reports/iterations.jsonl")));
      Assert.assertFalse(Files.exists(RunFiles.snapshot(run, 2)));
      // 反復の後: 封印したコホートを使い回してバージョン 2 を公開し、評価の前に止まる
      Assert.assertEquals(Lineage.run(run, Integer.MAX_VALUE, settings(), StopSignal.at(SafePoint.ITERATED, 1)), 0);
      Assert.assertEquals(next(run), 2);
      Assert.assertEquals(lines(run.resolve("reports/iterations.jsonl")).get(0).get("collection").getAsString(), "reused the sealed cohort");
      Assert.assertFalse(Files.exists(evaluations));
      // 評価 1 つの後: 起動時にバージョン 2 の 3xH を測って止まる（x1n3 はまだ）
      Assert.assertEquals(Lineage.run(run, Integer.MAX_VALUE, settings(), StopSignal.at(SafePoint.MEASURED, 2)), 0);
      Assert.assertEquals(next(run), 2);
      Assert.assertTrue(Files.exists(new EvaluationOutputs(evaluations, "3xH", 2).done()));
      var v2 = new EvaluationOutputs(evaluations, "x1n3", 2);
      Assert.assertFalse(v2.settled());
      // 反復 2: 起動時にバージョン 2 の x1n3 を測り、封印の後で止め、もう一度反復の後まで回してから、state.json を書く前に落ちたことにする
      Assert.assertEquals(Lineage.run(run, Integer.MAX_VALUE, settings(), StopSignal.at(SafePoint.SEALED, 2)), 0);
      assertMeasured(v2);
      byte[] v2Games = Files.readAllBytes(v2.tsv());
      FileTime v2Done = Files.getLastModifiedTime(v2.done());
      String before = Files.readString(run.resolve("state.json"));
      Assert.assertEquals(Lineage.run(run, Integer.MAX_VALUE, settings(), StopSignal.at(SafePoint.ITERATED, 2)), 0);
      Assert.assertEquals(next(run), 3);
      Files.writeString(run.resolve("state.json"), before);
      // バージョン 3 の x1n3 の途中で落ちた跡、バージョン 4 は測れなかった印、書きかけの記録の行、集めかけのコホート 3
      var v3 = new EvaluationOutputs(evaluations, "x1n3", 3);
      Files.writeString(v3.running(), "1\n");
      Files.writeString(v3.tsv().resolveSibling(v3.tsv().getFileName() + ".tmp"), "partial");
      var v4 = new EvaluationOutputs(evaluations, "x1n3", 4);
      Files.writeString(v4.failed(), "");
      Files.writeString(run.resolve("reports/iterations.jsonl"), "{\"iteration\": 3, \"trunc", StandardOpenOption.APPEND);
      Files.createDirectories(CohortWriter.directory(run, 3));
      Files.writeString(CohortWriter.directory(run, 3).resolve("partial"), "x");
      // 止めずに最後まで（iterations = 3）
      Assert.assertEquals(Lineage.run(run, Integer.MAX_VALUE, settings(), StopSignal.none()), 0);
      Assert.assertEquals(next(run), 4);
      List<JsonObject> iterations = lines(run.resolve("reports/iterations.jsonl"));
      Assert.assertEquals(iterations.stream().map(line -> line.get("iteration").getAsInt()).toList(), List.of(1, 2, 3));
      Assert.assertEquals(iterations.get(1).get("collection").getAsString(), "reused the sealed cohort");
      Assert.assertTrue(iterations.get(2).get("collection").isJsonObject(), "cohort 3 was collected again");
      Assert.assertFalse(Files.exists(CohortWriter.directory(run, 3).resolve("partial")));
      for (int version = 2; version <= 4; version++)
        Assert.assertTrue(Files.exists(new EvaluationOutputs(evaluations, "3xH", version).done()), "3xH v" + version);
      Assert.assertFalse(new EvaluationOutputs(evaluations, "3xH", 1).settled());
      Assert.assertEquals(Files.readAllBytes(v2.tsv()), v2Games, "version 2 was measured again");
      Assert.assertEquals(Files.getLastModifiedTime(v2.done()), v2Done);
      assertMeasured(v3);
      Assert.assertFalse(Files.exists(v3.tsv().resolveSibling(v3.tsv().getFileName() + ".tmp")));
      Assert.assertTrue(Files.exists(v4.failed()));
      Assert.assertFalse(Files.exists(v4.done()) || Files.exists(v4.tsv()) || Files.exists(v4.running()));
      for (int version = 2; version <= 4; version++)
        Assert.assertTrue(Files.isRegularFile(RunFiles.snapshot(run, version).resolve(Snapshot.POLICY)));
      // 合図が立っていれば何もせずに終わる
      StopSignal stopped = StopSignal.none();
      stopped.request("test");
      Assert.assertEquals(Lineage.run(run, Integer.MAX_VALUE, settings(), stopped), 0);
      Assert.assertEquals(next(run), 4);
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  /** バージョンの x1n3 が書き終わっている: .done があり .running はなく、tsv は 1 半荘 1 行、log は evaluate と同じ文言の 3 行。 */
  private static void assertMeasured(EvaluationOutputs outputs) throws Exception {
    Assert.assertTrue(Files.exists(outputs.done()), outputs.done().toString());
    Assert.assertFalse(Files.exists(outputs.running()));
    List<String> games = Files.readAllLines(outputs.tsv());
    Assert.assertEquals(games.size(), 2);
    for (String game : games) Assert.assertTrue(game.matches("\\d+\t[1-4]\t-?[0-9.E-]+"), game);
    String log = Files.readString(outputs.log());
    Assert.assertTrue(log.contains("evaluation workflow started: command=evaluate"), log);
    Assert.assertTrue(log.contains("evaluate play {"), log);
    Assert.assertTrue(
        log.matches(
            "(?s).*evaluation workflow complete: command=evaluate successful=true elapsedMs=\\d+ opponent=heuristic"
                + " games=2 rank=[0-9.]+±[0-9.]+ utility=\\S+ shares=[0-9./]+\n"),
        log);
  }

  /**
   * CPU で回る小さな系譜（LineageSmokeTest と同じ形、方策の記憶なし）に、バージョン 2 からバージョンごとの 3xH（1 局）と x1n3（3×Heuristic の卓、
   * 2 局）。
   */
  private static SettingsLoader settings() {
    String[] pairs = {
      "phiSource", "SCORE_RANK", "phiMonitor", "OFF", "phiDrift", "OFF", "gamesPerCohort", "1",
      "gamesSchedule", "", "iterations", "3",
      "collectDevices", "cpu", "learnerDevice", "cpu", "maxActiveGames", "8", "advanceWorkers", "2",
      "encoders", "2", "slots", "1", "readyBatches", "2", "inferBatch", "256",
      "batchWaitMicros", "1000", "rowWriters", "1", "labelRateDiscardEarly", "0.003",
      "labelRateDiscardMid", "0.003", "labelRateDiscardLate", "0.003", "labelRateCommit", "0.03",
      "maxCandidates", "4", "rollinRateDiscard", "0.001", "rollinRateCommit", "0.015", "rhoBranch", "0.2",
      "batchSize", "16", "decodeThreads", "2", "evaluation.3xH.games", "1", "evaluation.3xH.from", "2",
      "evaluation.x1n3.opponent", NanoTable.HEURISTIC, "evaluation.x1n3.games", "2", "evaluation.x1n3.every", "1",
      "evaluation.x1n3.from", "2"
    };
    Map<String, String> values = new LinkedHashMap<>();
    for (int i = 0; i < pairs.length; i += 2) values.put(PrismConfig.SETTINGS + pairs[i], pairs[i + 1]);
    return EpsilonSettings.of(values);
  }

  private static int next(Path run) throws Exception {
    return JsonParser.parseString(Files.readString(run.resolve("state.json"))).getAsJsonObject().get("next").getAsInt();
  }

  /** jsonl の記録（書きかけで止まった行は読み飛ばす）。 */
  private static List<JsonObject> lines(Path path) throws Exception {
    List<JsonObject> result = new ArrayList<>();
    for (String line : Files.readAllLines(path, StandardCharsets.UTF_8)) {
      if (line.isBlank()) continue;
      try {
        result.add(JsonParser.parseString(line).getAsJsonObject());
      } catch (RuntimeException truncated) {
        Assert.assertEquals(line, "{\"iteration\": 3, \"trunc");
      }
    }
    return result;
  }
}
