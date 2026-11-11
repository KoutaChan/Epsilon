package com.epsilon.prism.run;

import com.epsilon.agent.app.Exit;
import com.epsilon.agent.run.Ending;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.settings.Change;
import com.epsilon.prism.Main;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.eval.EvaluationTable;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import com.epsilon.prism.settings.PrismSettings;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * smoke のプリセットの系譜（CPU、評価のジョブを含む）と、反復の間の設定の変更のテスト。smoke のプリセットだけで作った run を 1 反復回す
 * （帳簿の補正と TEMPO は既定のまま。smoke の卓がバージョン 1・2 を測る）。次に settings change で TEMPO を切り、帳簿を RECORD にし、
 * 席ごとのツモの列を切り、学習のステップ数を減らし、評価の卓を 1 つ足して、もう 1 反復回す。反復 2 は新しい設定の行で回る。教師は生の Y で、
 * ゲイン 1・アンカーなし・学習率の倍率 1。帳簿は行に書くが、当てはめない。足した卓は、行を足した後に公開したバージョン 3 だけを測る
 * （バージョン 1・2 のジョブは作らない）。変えてはいけないキーの変更は拒み、設定の行を足さない。
 */
public class PresetLineageTest {

  @Test(groups = "slow", timeOut = 3_600_000)
  public void smokePresetRunsTheEvaluationJobsAndTakesChangesAtTheNextIteration() throws Exception {
    Path run = Files.createTempDirectory("prism-preset-lineage");
    RunDirectory directory = new RunDirectory(run);
    try {
      PrismSettings created = TestSettings.create(run, List.of("smoke"));
      Assert.assertEquals(created.schedule().iterations(), 2);
      Assert.assertEquals(PrismRun.runner(run, 1, () -> false).train(), Ending.LIMIT);
      Assert.assertEquals(state(run).get("next").getAsInt(), 2);
      for (int version = 1; version <= 2; version++) {
        Path job = directory.job(RunDirectory.jobName(version, "smoke"));
        Assert.assertTrue(Files.exists(job.resolve(Jobs.RESULT)), job.toString());
        Assert.assertEquals(Files.readAllLines(job.resolve(JobOutputs.GAMES)).size(), 12);
      }
      JsonObject first = learn(lines(directory.report(RunDirectory.ITERATIONS)).get(0));
      Assert.assertTrue(first.has("ledger"), "the default ledger is fitted");
      Assert.assertTrue(first.getAsJsonObject("tempo").get("anchorWeight").getAsDouble() > 0, "TEMPO anchors by default");
      int firstSteps = first.getAsJsonObject("fit").get("steps").getAsInt();
      Assert.assertTrue(firstSteps > 0 && firstSteps <= 8, "steps " + firstSteps);

      // 変えてはいけないキーの変更は拒み、設定の行を足さない
      Assert.assertEquals(change(run, "run.seed=7"), Exit.USAGE.code());
      Assert.assertEquals(rows(run).size(), 1);
      List<String> changes =
          new ArrayList<>(
              List.of(
                  "noise.tempo.enabled=false",
                  "noise.ledger.mode=RECORD",
                  "noise.crn.seatStreams=false",
                  "fit.maxSteps=2"));
      changes.addAll(List.of(TestSettings.table("late", EvaluationTable.HEURISTIC, 4, 1, 1)));
      Assert.assertEquals(change(run, changes.toArray(String[]::new)), Exit.OK.code());
      SettingsRow row = rows(run).getLast();
      Assert.assertEquals(row.row(), 2);
      Assert.assertEquals(row.from(), 2);
      Assert.assertTrue(
          row.changes().stream().anyMatch(changed -> changed.key().startsWith("evaluation.tables.late")
              && changed.change() == Change.SCHEDULE),
          row.changes().toString());

      Assert.assertEquals(PrismRun.runner(run, null, () -> false).train(), Ending.FINISHED);
      Assert.assertEquals(state(run).get("next").getAsInt(), 3);
      List<JsonObject> iterations = lines(directory.report(RunDirectory.ITERATIONS));
      Assert.assertEquals(iterations.size(), 2);
      Assert.assertEquals(iterations.get(1).get("iteration").getAsInt(), 2);
      Assert.assertEquals(iterations.get(1).get("settingsRow").getAsInt(), 2);
      JsonObject second = learn(iterations.get(1));
      JsonObject tempo = second.getAsJsonObject("tempo");
      Assert.assertEquals(tempo.get("gain").getAsDouble(), 1.0);
      Assert.assertEquals(tempo.get("anchorWeight").getAsDouble(), 0.0);
      Assert.assertEquals(tempo.get("lrScale").getAsDouble(), 1.0);
      Assert.assertFalse(second.has("ledger"), "RECORD does not fit the ledger");
      int secondSteps = second.getAsJsonObject("fit").get("steps").getAsInt();
      Assert.assertTrue(secondSteps > 0 && secondSteps <= 2, "steps " + secondSteps);
      // RECORD の行は帳簿を持つ
      List<Row> rows = cohortRows(CohortWriter.directory(run, 2));
      Assert.assertFalse(rows.isEmpty());
      Assert.assertTrue(rows.stream().anyMatch(Row::hasLedger), "RECORD writes the ledger");

      // smoke の卓はバージョン 3 も測り、足した卓はバージョン 3 だけを測る
      Assert.assertTrue(Files.exists(directory.job(RunDirectory.jobName(3, "smoke")).resolve(Jobs.RESULT)));
      Assert.assertTrue(Files.exists(directory.job(RunDirectory.jobName(3, "late")).resolve(Jobs.RESULT)));
      for (int version = 1; version <= 2; version++) {
        Path late = directory.job(RunDirectory.jobName(version, "late"));
        Assert.assertFalse(Files.exists(late), late.toString());
      }
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  private static int change(Path run, String... assignments) {
    List<String> arguments = new ArrayList<>(List.of("settings", "change", "--run", run.toString()));
    for (String assignment : assignments) {
      arguments.add("--change");
      arguments.add(assignment);
    }
    arguments.addAll(List.of("--reason", "test"));
    return Main.run(arguments.toArray(String[]::new));
  }

  private static List<SettingsRow> rows(Path run) throws Exception {
    return new SettingsRows(new RunDirectory(run)).read();
  }

  /** 確定したコホートの全ての行を返す。 */
  private static List<Row> cohortRows(Path cohort) throws Exception {
    CohortWriter.Manifest manifest = CohortWriter.readManifest(cohort);
    RowCodec codec = new RowCodec(manifest.ledgerFeatures(), manifest.variant());
    List<Row> rows = new ArrayList<>();
    for (CohortWriter.Shard shard : manifest.shards()) {
      var frames = ByteBuffer.wrap(Files.readAllBytes(cohort.resolve(shard.file())));
      while (frames.hasRemaining()) rows.add(codec.decode(frames));
    }
    Assert.assertEquals(rows.size(), manifest.rows());
    return rows;
  }

  /** 反復の報告から、学習の段の報告を返す。 */
  private static JsonObject learn(JsonObject iteration) {
    return iteration.getAsJsonObject("steps").getAsJsonObject(PrismStep.LEARN.name());
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
}
