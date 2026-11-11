package com.epsilon.prism.eval;

import ai.djl.Device;
import com.epsilon.agent.checkpoint.CheckpointFiles;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 打ち手の指定の書き方（評価の卓と、系譜の評価の相手）のテスト。系列と名前、デバイスの名前と、計画に書いた打ち手を開くときの照合を
 * 確かめる。
 */
public class EvaluationTableTest {

  @Test
  public void specsNameTheirSeries() throws Exception {
    Path root = Files.createTempDirectory("prism-evaluation-table").toAbsolutePath();
    try {
      Path snapshot = root.resolve("17");
      Assert.assertEquals(EvaluationTable.label(EvaluationTable.HEURISTIC), "heuristic");
      Assert.assertEquals(EvaluationTable.label("nano:" + root.resolve("latest")), "nano:latest");
      Assert.assertEquals(EvaluationTable.label("major:" + root.resolve("macro614")), "major:macro614");
      Assert.assertEquals(EvaluationTable.label(snapshot.toString()), "prism:17");
      Assert.assertEquals(EvaluationTable.label("prism:" + snapshot), "prism:17");
      Assert.assertEquals(EvaluationTable.spec(snapshot.toString()), new Series.Spec(Series.PRISM, snapshot));
      Assert.assertEquals(EvaluationTable.spec(EvaluationTable.HEURISTIC), new Series.Spec(Series.RULE_BOT, null));
      Assert.assertThrows(IllegalArgumentException.class, () -> EvaluationTable.spec("nano:"));
      Assert.assertThrows(IllegalArgumentException.class, () -> EvaluationTable.spec("heuristic:" + root));
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  /**
   * パスで渡す系譜の外のチェックポイントは、計画に今の中身の SHA-256 を書き、開くときに照合して、中身が替わっていれば拒む。系譜の
   * バージョンは計画に SHA-256 を書かず、開くときにも照合しない。
   */
  @Test
  public void anOutsideCheckpointThatChangedIsRefusedWhenItOpens() throws Exception {
    Path root = Files.createTempDirectory("prism-evaluation-verify").toAbsolutePath();
    try {
      Path outside = root.resolve("v760");
      Files.createDirectories(outside);
      Files.writeString(outside.resolve("weights.bin"), "v760", StandardCharsets.UTF_8);
      DuelSetup.Entrant planned = EvaluationTable.entrant("nano:" + outside);
      Assert.assertNull(planned.version());
      Assert.assertEquals(planned.sha256(), CheckpointFiles.sha256(outside));
      EvaluationTable.verify(planned);
      Files.writeString(outside.resolve("weights.bin"), "changed", StandardCharsets.UTF_8);
      IOException changed = Assert.expectThrows(IOException.class, () -> EvaluationTable.verify(planned));
      Assert.assertTrue(
          changed.getMessage().contains("but the duel was planned with " + planned.sha256()), changed.getMessage());

      Path snapshot = root.resolve("00017");
      Files.createDirectories(snapshot);
      Files.writeString(snapshot.resolve("weights.bin"), "17", StandardCharsets.UTF_8);
      DuelSetup.Entrant version = new DuelSetup.Entrant("prism:" + snapshot, "v00017", 17, null);
      Files.writeString(snapshot.resolve("weights.bin"), "17 again", StandardCharsets.UTF_8);
      EvaluationTable.verify(version);
      Assert.assertEquals(
          EvaluationTable.entrant(EvaluationTable.HEURISTIC),
          new DuelSetup.Entrant(EvaluationTable.HEURISTIC, "heuristic", null, null));
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void devicesAreCpuOrGpuN() {
    Assert.assertEquals(NetworkDevices.device("gpu0"), Device.gpu(0));
    Assert.assertEquals(NetworkDevices.device("cpu"), Device.cpu());
    IllegalArgumentException failure =
        Assert.expectThrows(IllegalArgumentException.class, () -> NetworkDevices.device("gpu:0"));
    Assert.assertTrue(failure.getMessage().contains("cpu or gpuN"), failure.getMessage());
  }
}
