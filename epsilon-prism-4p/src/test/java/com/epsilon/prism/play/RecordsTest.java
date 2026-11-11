package com.epsilon.prism.play;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.app.Exit;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.io.RecordFiles;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.ReplayDriver;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.Main;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.run.Snapshot;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 自己対局の牌譜（records）のテスト。乱数で初期化した小さな PRISM を 2 席、ルールベースの打ち手を 2 席に置いて CPU で半荘を打つ。牌譜は
 * 1 半荘ごとに 1 ファイルに書かれる。読み直すと記録と食い違わずに再生でき、プレイヤーの名前は席の打ち手の名前になる。乱数の種が同じなら、
 * 同じバイトになる。
 */
public class RecordsTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;

  /** CPU で回す小さな卓と、テストのルールと乱数の種。 */
  private static final String[] CPU = {
    "--set", "run.variant=TENHOU_4P", "--set", "run.seed=" + TestSettings.SEED, "--set", "table.activeGames=4", "--set",
    "table.workers=2", "--set", "inference.encoders=2", "--set", "table.batchWaitMicros=1000"
  };

  @Test(timeOut = 900_000)
  public void prismAndRuleSeatsWriteReplayableRecords() throws Exception {
    Path root = Files.createTempDirectory("prism-records");
    try {
      Path snapshot = root.resolve("1");
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        Snapshot.write(snapshot, HipFixtures.network(manager, 2, 17), PolicyParams.of(1));
      }
      Path out = root.resolve("records");
      run(
          "--model", snapshot.toString(),
          "--opponent", "heuristic",
          "--seats", "2",
          "--games", "4",
          "--first-game", "10",
          "--out", out.toString());
      long rounds = 0;
      for (long gameId = 10; gameId < 14; gameId++) {
        Path file = out.resolve(gameId + SelfPlayRecords.SUFFIX);
        RecordedGame game = RecordFiles.read(Files.readAllBytes(file), Long.toString(gameId));
        new ReplayDriver(VARIANT).replay(game, new RuleWorkspace(VARIANT));
        rounds += game.rounds().size();
        for (int seat = 0; seat < VARIANT.seats(); seat++) {
          String expected = SelfPlayRecords.candidateSeat(gameId, seat, 2) ? "prism:1" : "heuristic";
          Assert.assertEquals(game.meta().names().get(seat), expected, "game " + gameId + " seat " + seat);
        }
      }
      Assert.assertTrue(rounds >= 4, "rounds " + rounds);
      try (var files = Files.list(out)) {
        Assert.assertEquals(files.count(), 4);
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test(timeOut = 300_000)
  public void theSameSeedWritesTheSameBytes() throws Exception {
    Path root = Files.createTempDirectory("prism-records-rule");
    try {
      List<Path> outs = List.of(root.resolve("a"), root.resolve("b"));
      for (Path out : outs) {
        run("--model", "heuristic", "--games", "3", "--out", out.toString());
      }
      for (long gameId = 0; gameId < 3; gameId++) {
        String name = gameId + SelfPlayRecords.SUFFIX;
        byte[] first = Files.readAllBytes(outs.get(0).resolve(name));
        Assert.assertEquals(Files.readAllBytes(outs.get(1).resolve(name)), first, name);
        RecordedGame game = RecordFiles.read(first, name);
        Assert.assertTrue(game.meta().names().stream().allMatch("heuristic"::equals), game.meta().names().toString());
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test
  public void candidateSeatsRotateWithTheGame() {
    for (long gameId = 0; gameId < 8; gameId++) {
      boolean[] seats = new boolean[4];
      for (int seat = 0; seat < 4; seat++) seats[seat] = SelfPlayRecords.candidateSeat(gameId, seat, 3);
      int empty = (int) Math.floorMod(gameId + 3, 4);
      boolean[] expected = new boolean[4];
      Arrays.fill(expected, true);
      expected[empty] = false;
      Assert.assertEquals(seats, expected, "game " + gameId);
    }
  }

  /** records のコマンドを CPU の卓で実行し、終了コードが 0 であることを確かめる。 */
  private static void run(String... arguments) {
    String[] command = TestSettings.concat(TestSettings.concat(new String[] {"records"}, arguments), CPU);
    Assert.assertEquals(Main.run(command), Exit.OK.code());
  }
}
