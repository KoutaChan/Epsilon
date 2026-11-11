package com.epsilon.prism.run;

import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.eval.StoreEvent;
import com.epsilon.agent.run.Ending;
import com.epsilon.agent.run.JobFile;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.ThreePlayerSettings;
import com.epsilon.prism.app.PrismApp;
import com.epsilon.prism.grp.GrpFiles;
import com.epsilon.prism.grp.PhiEntry;
import com.epsilon.prism.grp.PhiEntryKind;
import com.epsilon.prism.grp.PhiHistory;
import com.epsilon.prism.grp.PhiSettings;
import com.epsilon.prism.settings.PrismSettings;
import com.google.gson.JsonParser;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの系譜のプリセットのスモークテスト（CPU）。smoke と three-player-lineage を重ね、反復を 1 回、2×R の卓を牌山 4 つの固定の
 * 大きさにした run を回す。バージョン 1・2 の 2×R のジョブと、バージョン 2 を対局なしで最初の王者にするジョブができる。Φ の履歴には、
 * 点数の順位から作った g000 の行が書かれる。基準の卓とラダーは、測り始めるバージョンにまだ届かないのでジョブを作らない。
 */
public class ThreePlayerPresetSmokeTest {

  @Test(groups = "slow", timeOut = 3_600_000)
  public void theLineagePresetOnTheSmokePresetRunsOneIterationOnCpu() throws Exception {
    Path run = Files.createTempDirectory("prism-3p-preset");
    RunDirectory directory = new RunDirectory(run);
    try {
      PrismSettings created =
          ThreePlayerSettings.create(
              run,
              List.of("smoke", "three-player-lineage"),
              "schedule.iterations=1",
              "evaluation.tables.rx2.stop=FIXED",
              "evaluation.tables.rx2.walls=4",
              "phi.grp.threads=1");
      Assert.assertEquals(
          created.evaluation().tables().keySet(), Set.of("champion", "ladder32", "ladder8", "reference", "rx2", "smoke"));
      Assert.assertEquals(PrismRun.runner(run, null, () -> false).train(), Ending.FINISHED);
      Assert.assertEquals(
          JsonParser.parseString(Files.readString(directory.state())).getAsJsonObject().get("next").getAsInt(), 2);
      PrismSettings latest = PrismApp.latest(run);
      Assert.assertEquals(latest.run().variant(), Variant.TENHOU_3P);
      Assert.assertEquals(latest.evaluation().anchor(), "reference");
      Assert.assertEquals(((PhiSettings.SelfGrp) latest.phi()).grp().trustGames(), 8192);
      Assert.assertEquals(
          Files.readAllLines(directory.report(RunDirectory.ITERATIONS), StandardCharsets.UTF_8).stream()
              .filter(line -> !line.isBlank())
              .count(),
          1L);

      // ジョブ: v1 は 2×R と smoke、v2 は王者・2×R・smoke。基準の卓（v40 から）とラダー（v16 から）のジョブは無い
      Jobs jobs = new Jobs(directory.jobs(), directory.report(RunDirectory.EVALUATIONS));
      List<String> names = jobs.list().jobs().stream().map(JobFile::name).sorted().toList();
      Assert.assertEquals(
          names, List.of("v00001-rx2", "v00001-smoke", "v00002-champion", "v00002-rx2", "v00002-smoke"));
      for (JobFile job : jobs.list().jobs()) {
        Assert.assertEquals(job.state(), JobFile.State.DONE, job.name() + " " + job.reason());
        Assert.assertTrue(Files.exists(directory.job(job.name()).resolve(Jobs.RESULT)), job.name());
      }
      // 2×R は牌山 4 つを、1 対 2 の席の割り当て 3 通りの全てで打つ
      for (int version = 1; version <= 2; version++) {
        Path job = directory.job(RunDirectory.jobName(version, "rx2"));
        DuelResult result = Jobs.result(job, DuelResult.class);
        Assert.assertEquals(result.walls(), 4, job.toString());
        Assert.assertEquals(result.games(), 12L, job.toString());
        Assert.assertEquals(Files.readAllLines(job.resolve(JobOutputs.GAMES)).size(), 12, job.toString());
      }
      // 王者の保存先は空だったので、v2 を対局なしで最初の王者にする
      List<ChampionStore.Entry> history = ChampionStore.open(directory.champion()).history();
      Assert.assertEquals(history.size(), 1, history.toString());
      Assert.assertTrue(
          history.getFirst().event() instanceof StoreEvent.Initial initial && initial.champion().version() == 2,
          history.toString());

      // Φ の履歴の最初の行は、三人打ちの点数の順位から作った g000 で、反復 1 から使う
      List<PhiEntry> phi = PhiHistory.read(new GrpFiles(run));
      Assert.assertFalse(phi.isEmpty());
      PhiEntry initial = phi.getFirst();
      Assert.assertEquals(initial.id(), PhiEntry.INITIAL_ID);
      Assert.assertEquals(initial.kind(), PhiEntryKind.SCORE_RANK);
      Assert.assertEquals(initial.variant(), Variant.TENHOU_3P);
      Assert.assertEquals(initial.effectiveFrom(), 1);
    } finally {
      Snapshot.deleteTree(run);
    }
  }
}
