package com.epsilon.prism.play;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.core.Action;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.run.RunFiles;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.workflow.WorkflowArguments;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/** play-riichi: 打つ版の選び方、token の確かめ、版の s の貪欲が実際の判断で合法手を返すこと（CPU、通信なし）。 */
public class RiichiPlayWorkflowTest {
  @Test
  public void checkpointDirIsASnapshotOrTheLatestWrittenSnapshotOfARun() throws Exception {
    Path root = Files.createTempDirectory("prism-riichi");
    try {
      Path older = RunFiles.snapshot(root, 9);
      Path newer = RunFiles.snapshot(root, 12);
      for (Path snapshot : List.of(older, newer)) {
        Files.createDirectories(snapshot);
        Files.writeString(snapshot.resolve(Snapshot.POLICY), "{}");
      }
      // 書き込み中の版（policy.json がまだない）は選ばない
      Files.createDirectories(RunFiles.snapshot(root, 13));
      Assert.assertEquals(RiichiPlayWorkflow.snapshot(root), newer);
      Assert.assertEquals(RiichiPlayWorkflow.snapshot(older), older);
      Assert.assertThrows(IOException.class, () -> RiichiPlayWorkflow.snapshot(root.resolve("missing")));
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void playRiichiNeedsAToken() {
    var arguments = new WorkflowArguments("play-riichi", new String[0], EpsilonSettings.defaults());
    var failure =
        Assert.expectThrows(
            IllegalArgumentException.class, () -> new RiichiPlayWorkflow().execute(arguments));
    Assert.assertTrue(failure.getMessage().contains("epsilon.riichi.token"), failure.getMessage());
  }

  /** 版を開いて半荘を貪欲で進め、手が 2 つ以上の判断 12 個で合法手が返る（CPU は FP32 の eager）。 */
  @Test(timeOut = 300_000)
  public void greedyPlayerAnswersRealDecisions() throws Exception {
    Path root = Files.createTempDirectory("prism-riichi");
    try {
      Path snapshot = root.resolve("1");
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        Snapshot.write(snapshot, HipFixtures.network(manager, 2, 17), PolicyParams.of(1));
      }
      try (SnapshotScorer scorer =
          SnapshotScorer.open(
              snapshot, EpsilonSettings.of(Map.of("epsilon.devices.inference", "cpu")))) {
        RiichiPlayWorkflow.warmUp(scorer);
        var player = new GreedyPlayer(scorer);
        var engine = new GameEngine(23);
        var selections = new EngineSelectionBuffer();
        int answered = 0;
        GameStepResult step = engine.stepHanchan();
        while (answered < 12 && !(step instanceof GameStepResult.HanchanEnded)) {
          if (!(step instanceof GameStepResult.AwaitingDecisions awaiting)) {
            step = engine.stepHanchan();
            continue;
          }
          selections.clear();
          for (EngineDecisionPoint point : awaiting.decisions()) {
            List<Action> legal = point.legalActions();
            Action chosen = player.selectAction(engine.getState(), point.player(), legal);
            Assert.assertTrue(legal.contains(chosen), point.toString());
            if (legal.size() > 1) answered++;
            selections.add(point.id(), chosen);
          }
          step = engine.commitDecisions(selections);
        }
        Assert.assertTrue(answered >= 12, "answered " + answered);
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }
}
