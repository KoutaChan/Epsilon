package com.epsilon.prism.policy;

import static org.testng.Assert.assertEquals;

import ai.djl.engine.Engine;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.prism.Main;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.app.Evaluate;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.run.Snapshot;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * GPU のスモークテスト（テストグループ rocm。GPU 1 枚）。evaluate でバージョンを Nano 3 席と -Dprism.games 半荘（既定 2048）打たせ、
 * 平均着順と秒数を出す。バージョンは -Dprism.snapshot で渡し、無ければ標準の形・深さ 2 の初期化したままのバージョンを作る。Nano は
 * -Dprism.nano に classic の checkpoint の絶対パスを渡し、classic の系列で開く。
 *
 * <p>推論のバックエンドは -Dprism.kernels（EAGER か NATIVE、既定 EAGER。設定の inference.kernels）で選び、計算の型は BFLOAT16。
 * PRISM の席の判断/秒は、方策を閉じるときのログ「PRISM policy v&lt;バージョン&gt; (&lt;バックエンド&gt;)」に出る。
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
    Kernels kernels = Kernels.valueOf(System.getProperty("prism.kernels", Kernels.EAGER.name()));
    Path root = Files.createTempDirectory("prism-gpu");
    try {
      String given = System.getProperty("prism.snapshot");
      Path snapshot = given == null ? root.resolve("1") : Path.of(given);
      if (given == null) {
        NetworkPolicyTest.write(snapshot, PrismShape.standard(), false);
      }
      Path out = root.resolve("evaluate");
      List<String> arguments =
          new ArrayList<>(
              List.of(
                  "evaluate", "--model", snapshot.toString(), "--table", "x1n3", "--out", out.toString(),
                  "--nano-device", "gpu0", "--set", "run.variant=TENHOU_4P", "--set", "run.seed=1", "--set",
                  "inference.devices=[\"gpu0\"]", "--set", "inference.precision=BFLOAT16", "--set",
                  "inference.kernels=" + kernels.name()));
      for (String assignment : TestSettings.table("x1n3", "nano:" + nano, games, 1, 1)) {
        arguments.addAll(List.of("--set", assignment));
      }
      assertEquals(Main.run(arguments.toArray(String[]::new)), Exit.OK.code());
      Path job = out.resolve("jobs").resolve(RunDirectory.jobName(Evaluate.MODEL_VERSION, "x1n3"));
      DuelResult result = Jobs.result(job, DuelResult.class);
      DuelResult.SlotValue candidate = result.slots().get(0);
      log.info(
          "PRISM-X1N3 snapshot={} kernels={} games={} rank={}±{} seconds={} games/s={}",
          snapshot,
          kernels,
          result.games(),
          String.format("%.4f", candidate.meanRank()),
          String.format("%.4f", candidate.rankError()),
          String.format("%.1f", result.seconds()),
          String.format("%.2f", result.games() / result.seconds()));
      assertEquals(result.games(), games);
      assertEquals(Files.readAllLines(job.resolve(JobOutputs.GAMES)).size(), games);
    } finally {
      Snapshot.deleteTree(root);
    }
  }
}
