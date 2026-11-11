package com.epsilon.prism.app;

import static com.epsilon.mahjong.rules.Variant.TENHOU_3P;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.app.RunFiles;
import com.epsilon.agent.policy.ClientSeat;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.backend.PerturbedNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism3p.input.InputSchema;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 三人打ちの play-riichi のテスト（CPU、通信なし）。run のディレクトリの最新のバージョンが三人打ちの入力の形式なら、play-riichi は
 * 設定なしで三人打ちで打つ。そのルールで開いた方策をウォームアップし、三人打ちのエンジンの半荘の判断に、席が合法手の番号で答える。
 */
public class ThreePlayerPlayRiichiTest {

  @Test(timeOut = 300_000)
  public void aThreePlayerRunPlaysThreePlayer() throws Exception {
    Path root = Files.createTempDirectory("prism-3p-riichi");
    try {
      Path snapshot = RunFiles.snapshot(root, 4);
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        Snapshot.write(
            snapshot, PerturbedNetwork.of(manager, InputSchema.FORMAT, PrismShape.standard(2), 23), PolicyParams.of(4));
      }
      Path chosen = PlayRiichi.snapshot(root);
      Assert.assertEquals(chosen, snapshot);
      Assert.assertEquals(PlayRiichi.variant(chosen), TENHOU_3P);
      int[] answered = new int[1];
      try (Policy policy =
          ModelFamily.of(Series.PRISM).open(chosen, PlayRiichi.variant(chosen), PolicyOptions.defaults())) {
        Assert.assertEquals(policy.variant(), TENHOU_3P);
        PlayRiichi.warmUp(policy);
        var seat = new ClientSeat(policy);
        new TableWalk(TENHOU_3P, TableWalk.Policy.GREEDY)
            .run(
                29,
                1,
                (engine, slot) -> {
                  if (answered[0] >= 12) {
                    return;
                  }
                  Decisions decisions = engine.decisions();
                  int chosenAction = seat.decide(engine.view(decisions.seat(slot)), decisions, slot);
                  Assert.assertTrue(
                      chosenAction >= 0 && chosenAction < decisions.legalCount(slot), "chosen " + chosenAction);
                  if (decisions.legalCount(slot) > 1) {
                    answered[0]++;
                  }
                });
      }
      Assert.assertTrue(answered[0] >= 12, "answered " + answered[0]);
    } finally {
      Snapshot.deleteTree(root);
    }
  }
}
