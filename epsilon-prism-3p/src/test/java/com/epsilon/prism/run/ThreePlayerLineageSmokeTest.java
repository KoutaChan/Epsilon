package com.epsilon.prism.run;

import com.epsilon.agent.app.Exit;
import com.epsilon.agent.run.Ending;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.grp.train.CountFile;
import com.epsilon.grp.train.CountedHands;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.Main;
import com.epsilon.prism.ThreePlayerSettings;
import com.epsilon.prism.app.FitCohorts;
import com.epsilon.prism.app.PrismApp;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.grp.GrpFiles;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.settings.PrismSettings;
import com.epsilon.prism3p.input.InputSchema;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの系譜のスモークテスト（CPU）。smoke のプリセットに三人打ちと自前の GRP（点数の順位から始める）を重ねた run を 2 反復回す。
 * 初期化したままのバージョン 1 で集めて学習し、バージョン 2・3 を公開できることを確かめる。行・バージョン・Φ の履歴と、smoke の卓の
 * 評価のジョブがそろう。ネットワークの入力とヘッドのフィンガープリントは三人打ちの値になり、北抜きの判断は北抜きのクラスに数えられる。
 * 各コホートの局の数のファイルは grp/counts に移り、その局の数は収集の数と合い、牌譜を再現できる。ファイルのヘッダーにある Φ は、Φ の段で
 * 使った値と同じ。その run のコホートで fit-cohorts を回すと、三人打ちのネットワークができる。
 */
public class ThreePlayerLineageSmokeTest {

  @Test(groups = "slow", timeOut = 3_600_000)
  public void theSmokePresetRunsTwoIterationsOnCpu() throws Exception {
    Path run = Files.createTempDirectory("prism-3p-lineage");
    RunDirectory directory = new RunDirectory(run);
    try {
      PrismSettings created =
          ThreePlayerSettings.create(
              run,
              List.of("smoke"),
              "phi.source=SELF_GRP",
              "phi.start.kind=SCORE_RANK",
              "phi.grp.refit=SCHEDULE",
              "phi.grp.switching=AUTO",
              "phi.grp.threads=1");
      Assert.assertEquals(created.schedule().iterations(), 2);
      Assert.assertEquals(PrismRun.runner(run, null, () -> false).train(), Ending.FINISHED);
      Assert.assertEquals(state(run).get("next").getAsInt(), 3);
      Assert.assertEquals(PrismApp.latest(run).run().variant(), Variant.TENHOU_3P);

      List<JsonObject> iterations = lines(directory.report(RunDirectory.ITERATIONS));
      Assert.assertEquals(iterations.size(), 2);
      boolean nuki = false;
      for (int t = 1; t <= 2; t++) {
        JsonObject steps = iterations.get(t - 1).getAsJsonObject("steps");
        long rows = steps.getAsJsonObject(PrismStep.COLLECT.name()).get("rows").getAsLong();
        Assert.assertTrue(rows > 0, "rows of cohort " + t);
        int fitSteps = steps.getAsJsonObject(PrismStep.LEARN.name()).getAsJsonObject("fit").get("steps").getAsInt();
        Assert.assertTrue(fitSteps > 0 && fitSteps <= 8, "steps " + fitSteps);
        Path cohort = CohortWriter.directory(run, t);
        Assert.assertTrue(Files.exists(cohort.resolve(CohortWriter.MANIFEST)), cohort.toString());
        nuki |= Files.readString(cohort.resolve(CollectReport.CLASSES)).contains("\"NUKI\"");
        // 主経路の局の数のファイルは、コホートを確定する前にコホートへ書かれ、仕上げで grp/counts に移っている
        GrpFiles grp = new GrpFiles(run);
        Assert.assertTrue(grp.sealed(t), "cohort " + t);
        Assert.assertFalse(Files.exists(grp.unmovedCounts(t)), "cohort " + t);
        CountedHands counts = CountFile.read(grp.counts(t));
        JsonObject collect = steps.getAsJsonObject(PrismStep.COLLECT.name());
        Assert.assertEquals(counts.variant(), Variant.TENHOU_3P);
        Assert.assertEquals(counts.cohort(), t);
        Assert.assertEquals(counts.games(), collect.get("games").getAsLong());
        Assert.assertEquals(counts.hands(), collect.get("hands").getAsLong());
        Assert.assertEquals(counts.games(), GameRecordCodec.read(grp.records(t)).games().size());
        Assert.assertEquals(counts.firstMismatch(), -1);
        Assert.assertEquals(counts.exploreTemperature(), created.behaviour().exploreTemperature());
        Assert.assertEquals(counts.phi().mode(), "GRP");
        Assert.assertTrue(Double.isNaN(counts.phi().cutoff()), "the GRP mode has no cutoff");
        String used = steps.getAsJsonObject(PrismStep.PHI.name()).getAsJsonObject("value").get("mode").getAsString();
        Assert.assertEquals(counts.phi().mode(), used);
      }
      Assert.assertTrue(nuki, "no decision was classed as NUKI");

      for (int version = 2; version <= 3; version++) {
        Path model = directory.snapshot(version).resolve(Snapshot.MODEL);
        Assert.assertEquals(PrismNetwork.readShape(model), created.network().shape());
        JsonObject written = JsonParser.parseString(Files.readString(model.resolve("model.json"))).getAsJsonObject();
        Assert.assertEquals(written.get("inputSchema").getAsString(), InputSchema.fingerprint());
        Assert.assertEquals(written.get("headSchema").getAsString(), TargetSchema.of(Variant.TENHOU_3P).fingerprint());
      }
      Assert.assertFalse(Arrays.equals(weights(directory, 1), weights(directory, 2)), "version 2 learned nothing");
      for (int version = 1; version <= 3; version++) {
        Path job = directory.job(RunDirectory.jobName(version, "smoke"));
        Assert.assertTrue(Files.exists(job.resolve(Jobs.RESULT)), job.toString());
        Assert.assertEquals(Files.readAllLines(job.resolve(JobOutputs.GAMES)).size(), 12);
      }
      String phi = Files.readString(new GrpFiles(run).phiHistory());
      Assert.assertTrue(phi.contains("TENHOU_3P"), phi);

      // fit-cohorts は三人打ちのコホートから三人打ちのネットワークを作る
      Path refit = Files.createTempDirectory("prism-3p-refit");
      try {
        String[] arguments = {
          "fit-cohorts", "--run", run.toString(), "--cohorts", "1..2", "--out", refit.toString(), "--max-rows", "32"
        };
        Assert.assertEquals(Main.run(arguments), Exit.OK.code());
        Path model = refit.resolve(FitCohorts.SNAPSHOT).resolve(Snapshot.MODEL);
        Assert.assertEquals(PrismNetwork.readFormat(model).variant(), Variant.TENHOU_3P);
        JsonObject written = JsonParser.parseString(Files.readString(model.resolve("model.json"))).getAsJsonObject();
        Assert.assertEquals(written.get("inputSchema").getAsString(), InputSchema.fingerprint());
      } finally {
        Snapshot.deleteTree(refit);
      }
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  private static byte[] weights(RunDirectory directory, int version) throws Exception {
    return Files.readAllBytes(directory.snapshot(version).resolve(Snapshot.MODEL).resolve("weights.bin"));
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
