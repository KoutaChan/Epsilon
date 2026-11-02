package com.epsilon.prism.policy;

import static org.testng.Assert.assertEquals;

import ai.djl.engine.Engine;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.eval.Arena;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.workflow.EvaluateWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * GPU の煙試験（rocm の組。GPU 1 枚）。版（-Dprism.snapshot。なければ標準の形・深さ 2 の白紙の版を作る）を evaluate で Nano の 3 席
 * （-Dprism.nano の Decision チェックポイント）と -Dprism.games 局（既定 2048）打たせ、平着と秒を出す。推論の道は -Dprism.inference
 * （EAGER か HIP、既定 EAGER。設定 epsilon.prism.inference）。PRISM の席の判断/秒は、方策が閉じるときの記録
 * 「PRISM policy v&lt;版&gt; (&lt;道&gt;)」に出る。
 */
public class NetworkPolicyGpuSmokeTest {
  private static final Logger log = LoggerFactory.getLogger(NetworkPolicyGpuSmokeTest.class);

  @Test(groups = "rocm")
  public void playsAgainstNano() throws Exception {
    String nano = System.getProperty("prism.nano");
    if (Engine.getInstance().getGpuCount() == 0 || nano == null) {
      throw new SkipException("needs a GPU and -Dprism.nano");
    }
    int games = Integer.getInteger("prism.games", 2048);
    InferencePath inference =
        InferencePath.valueOf(System.getProperty("prism.inference", InferencePath.EAGER.name()));
    Path root = Files.createTempDirectory("prism-gpu");
    try {
      String given = System.getProperty("prism.snapshot");
      Path snapshot = given == null ? root.resolve("1") : Path.of(given);
      if (given == null) {
        NetworkPolicyTest.write(snapshot, PrismShape.standard(), false);
      }
      Path perGame = root.resolve("per-game.tsv");
      Arena.Result result =
          new EvaluateWorkflow()
              .execute(
                  new WorkflowArguments(
                      "evaluate",
                      new String[] {
                        "--snapshot", snapshot.toString(),
                        "--opponent", "nano:" + nano,
                        "--nano-device", "gpu:0",
                        "--games", Integer.toString(games),
                        "--per-game", perGame.toString()
                      },
                      EpsilonSettings.of(
                          Map.of(
                              "epsilon.prism.collectDevices", "gpu0",
                              "epsilon.prism.inference", inference.name()))));
      log.info(
          "PRISM-X1N3 snapshot={} inference={} games={} rank={}±{} seconds={} games/s={}",
          snapshot,
          inference,
          result.games(),
          String.format("%.4f", result.meanRank()),
          String.format("%.4f", result.rankError()),
          String.format("%.1f", result.seconds()),
          String.format("%.2f", result.games() / result.seconds()));
      assertEquals(result.games(), games);
      assertEquals(Files.readAllLines(perGame).size(), games);
    } finally {
      Snapshot.deleteTree(root);
    }
  }
}
