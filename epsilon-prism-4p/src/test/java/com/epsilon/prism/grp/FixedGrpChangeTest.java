package com.epsilon.prism.grp;

import com.epsilon.agent.app.Exit;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.Ending;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.prism.Main;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.run.PrismRun;
import com.epsilon.prism.run.Snapshot;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 固定の GRP のファイル（phi.grp）は、学習の途中で変えると次の反復から効くキーであることを確かめる。学習の JVM が回っている間に変えると、
 * 起動し直さなくても次に始まる反復の Φ がその GRP になり、反復の報告の Φ の項目と設定の行に残る（CPU で、1 コホート 1 半荘を 2 反復）。
 */
public class FixedGrpChangeTest {

  @Test(groups = "slow", timeOut = 1_800_000)
  public void aChangedFixedGrpTakesEffectFromTheNextIteration() throws Exception {
    Path root = Files.createTempDirectory("prism-fixed-grp");
    try {
      Path first = GrpFixtures.network(root.resolve("grp-a"), 1);
      Path second = GrpFixtures.network(root.resolve("grp-b"), 2);
      Path run = root.resolve("run");
      TestSettings.create(
          run,
          TestSettings.cpuLineage(
              "phi.source=FIXED_GRP",
              "phi.grp=" + first,
              "schedule.cohorts=[{from = 1, games = 1}]",
              "schedule.iterations=2",
              "fit.maxSteps=0"));
      RunDirectory directory = new RunDirectory(run);
      List<Integer> changed = new ArrayList<>();
      // 反復 1 を閉じた後のセーフポイントで、同じ JVM が回っている間に固定の GRP を替える
      Ending ending =
          PrismRun.runner(
                  run,
                  null,
                  () -> {
                    if (changed.isEmpty() && reports(directory).size() == 1) {
                      changed.add(
                          Main.run(
                              new String[] {
                                "settings", "change", "--run", run.toString(), "--change", "phi.grp=" + second,
                                "--reason", "test"
                              }));
                    }
                    return false;
                  })
              .train();
      Assert.assertEquals(ending, Ending.FINISHED);
      Assert.assertEquals(changed, List.of(Exit.OK.code()));
      List<String> reports = reports(directory);
      Assert.assertEquals(reports.size(), 2);
      JsonObject before = JsonParser.parseString(reports.get(0)).getAsJsonObject();
      JsonObject after = JsonParser.parseString(reports.get(1)).getAsJsonObject();
      Assert.assertEquals(phiSha(before), GrpCheckpoint.sha256(first));
      Assert.assertEquals(phiSha(after), GrpCheckpoint.sha256(second));
      Assert.assertEquals(after.get("settingsRow").getAsInt(), 2);
      Assert.assertEquals(phi(after).get("launchRow").getAsInt(), 1, "the same JVM ran both iterations");
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  /** 反復の報告の行を返す。 */
  private static List<String> reports(RunDirectory run) {
    try {
      return DurableFiles.lines(run.report(RunDirectory.ITERATIONS));
    } catch (IOException unreadable) {
      throw new UncheckedIOException(unreadable);
    }
  }

  /** 反復の報告から、Φ の段の記録を返す。 */
  private static JsonObject phi(JsonObject report) {
    return report.getAsJsonObject("steps").getAsJsonObject("PHI");
  }

  /** 反復の報告で、Φ の段が決めた項目にある GRP の sha256 を返す。 */
  private static String phiSha(JsonObject report) {
    return phi(report).getAsJsonObject("value").getAsJsonObject("entry").get("sha256").getAsString();
  }
}
