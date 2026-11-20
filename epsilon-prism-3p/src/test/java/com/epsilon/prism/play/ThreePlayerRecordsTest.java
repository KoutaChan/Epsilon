package com.epsilon.prism.play;

import com.epsilon.agent.app.Exit;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.io.RecordFiles;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.ReplayDriver;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.Main;
import com.epsilon.prism.ThreePlayerSettings;
import com.epsilon.prism.run.Snapshot;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 三人打ちの run の records は、三人打ちの半荘を書く。読み直すと記録と食い違わずに再生でき、席の数だけ打ち手の名前が並ぶ。 */
public class ThreePlayerRecordsTest {

  private static final Variant THREE = Variant.TENHOU_3P;

  @Test(timeOut = 300_000)
  public void threePlayerGamesAreWrittenAndReplay() throws Exception {
    Path root = Files.createTempDirectory("prism-3p-records");
    try {
      Path out = root.resolve("records");
      String[] command = {
        "records",
        "--model",
        "heuristic",
        "--games",
        "2",
        "--out",
        out.toString(),
        "--set",
        "run.variant=TENHOU_3P",
        "--set",
        "run.seed=" + ThreePlayerSettings.SEED,
        "--set",
        "table.activeGames=2",
        "--set",
        "table.workers=2",
        "--set",
        "inference.encoders=2"
      };
      Assert.assertEquals(Main.run(command), Exit.OK.code());
      for (long gameId = 0; gameId < 2; gameId++) {
        String name = gameId + SelfPlayRecords.SUFFIX;
        RecordedGame game = RecordFiles.read(Files.readAllBytes(out.resolve(name)), name);
        new ReplayDriver(THREE).replay(game, new RuleWorkspace(THREE));
        Assert.assertFalse(game.rounds().isEmpty(), name);
        Assert.assertEquals(game.meta().names().size(), THREE.seats(), name);
        Assert.assertTrue(
            game.meta().names().stream().allMatch("heuristic"::equals),
            game.meta().names().toString());
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }
}
