package com.epsilon.prism.app;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.app.RunFiles;
import com.epsilon.agent.policy.ClientSeat;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.prism.Main;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * play-riichi のテスト（CPU、通信なし）。打つバージョンの選び方、token の受け付け方（設定ファイルからだけ受け付ける）、方策を開く設定の
 * セクションを確かめる。接続ごとの席が、新しいエンジンの実際の判断に合法手の番号で答えることも確かめる。
 */
public class PlayRiichiTest {

  @Test
  public void modelIsASnapshotOrTheLatestWrittenSnapshotOfARun() throws Exception {
    Path root = Files.createTempDirectory("prism-riichi");
    try {
      Path older = RunFiles.snapshot(root, 9);
      Path newer = RunFiles.snapshot(root, 12);
      for (Path snapshot : List.of(older, newer)) {
        Files.createDirectories(snapshot);
        Files.writeString(snapshot.resolve(Snapshot.POLICY), "{}");
      }
      // 書き込み中のバージョン（policy.json がまだない）は選ばない
      Files.createDirectories(RunFiles.snapshot(root, 13));
      Assert.assertEquals(PlayRiichi.snapshot(root), newer);
      Assert.assertEquals(PlayRiichi.snapshot(older), older);
      Assert.assertThrows(IOException.class, () -> PlayRiichi.snapshot(root.resolve("missing")));
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void playRiichiNeedsATokenFromTheSettingsFile() {
    var failure =
        Assert.expectThrows(
            SettingsException.class,
            () -> new PlayRiichi(Path.of("/runs/x"), 0, PolicyOptions.defaults(), null).run(null));
    Assert.assertTrue(failure.getMessage().contains("play-riichi.token"), failure.getMessage());
    // 秘密のキーはコマンドラインから受け付けない
    Assert.assertEquals(
        Main.run(new String[] {"play-riichi", "--model", "/runs/x", "--token", "abc"}), Exit.USAGE.code());
  }

  @Test
  public void policyComesFromTheCommandSection() {
    Layer file =
        Layer.parse(
            "riichi.toml",
            Layer.Source.FILE,
            "[play-riichi]\nmodel = \"/runs/x\"\ntoken = \"abc\"\n\n[play-riichi.policy]\ndevice = \"gpu1\"\n");
    PlayRiichi command = Settings.section(PlayRiichi.class, "play-riichi", List.of(file)).value();
    Assert.assertEquals(command.policy().device(), "gpu1");
    Assert.assertEquals(command.policy().precision(), Precision.FLOAT32);
    Assert.assertEquals(command.token(), "abc");
    Layer plain = Layer.parse("plain.toml", Layer.Source.FILE, "[play-riichi]\nmodel = \"/runs/x\"\n");
    Assert.assertEquals(
        Settings.section(PlayRiichi.class, "play-riichi", List.of(plain)).value().policy(), PolicyOptions.defaults());
  }

  /** バージョンを開いてウォームアップし、新しいエンジンの半荘で合法手が 2 つ以上ある判断 12 個に、席が合法手の番号で答える（CPU は FP32 の eager）。 */
  @Test(timeOut = 300_000)
  public void clientSeatAnswersRealDecisions() throws Exception {
    Path root = Files.createTempDirectory("prism-riichi");
    try {
      Path snapshot = root.resolve("1");
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        Snapshot.write(snapshot, HipFixtures.network(manager, 2, 17), PolicyParams.of(1));
      }
      int[] answered = new int[1];
      try (Policy policy = ModelFamily.of(Series.PRISM).open(snapshot, TENHOU_4P, PolicyOptions.defaults())) {
        PlayRiichi.warmUp(policy);
        var seat = new ClientSeat(policy);
        new TableWalk(TENHOU_4P, TableWalk.Policy.GREEDY)
            .run(
                23,
                1,
                (engine, slot) -> {
                  if (answered[0] >= 12) {
                    return;
                  }
                  Decisions decisions = engine.decisions();
                  int chosen = seat.decide(engine.view(decisions.seat(slot)), decisions, slot);
                  Assert.assertTrue(chosen >= 0 && chosen < decisions.legalCount(slot), "chosen " + chosen);
                  if (decisions.legalCount(slot) > 1) answered[0]++;
                });
      }
      Assert.assertTrue(answered[0] >= 12, "answered " + answered[0]);
    } finally {
      Snapshot.deleteTree(root);
    }
  }
}
