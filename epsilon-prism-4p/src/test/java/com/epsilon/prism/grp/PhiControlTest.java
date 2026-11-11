package com.epsilon.prism.grp;

import com.epsilon.agent.app.Exit;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.TrainLock;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.prism.Main;
import com.epsilon.prism.run.Snapshot;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Φ の切り替えと巻き戻し（{@link PhiControl#switchTo} と grp-switch）のテスト。READY の候補は、まだ始まっていない最初の反復から効かせる
 * （学習の JVM が動いていれば、さらに 1 つ先から）。巻き戻しでは、既にある項目の行を足す。既に始まった反復、READY でない候補、知らない id
 * は拒む。既に同じ Φ が効いていれば何も足さず、コマンドは 0 で終わる。自前の GRP でない run では、コマンドを使い方の誤りとして拒む。
 * チェックポイントか q が開けない項目と、sha の変わった run の外のチェックポイントも拒む。
 */
public class PhiControlTest {

  /** PRISM のコマンドを 1 つ実行し、終了コードを返す。 */
  static int prism(String... arguments) {
    return Main.run(arguments);
  }

  @Test
  public void switchesRollsBackAndRefuses() throws Exception {
    Path run = Files.createTempDirectory("prism-switch");
    try {
      var files = new GrpFiles(run);
      GrpFixtures.writeSettings(run, GrpFixtures.scoreStartAssignments());
      GrpFixtures.writeState(run, 8);
      // g000 は学習の JVM が書く。書かれる前は切り替えられない
      Assert.assertThrows(IllegalStateException.class, () -> PhiControl.switchTo(run, "g000", null, "test"));
      Assert.assertEquals(prism("grp-switch", "--run", run.toString(), "--to", "g000"), Exit.FAILED.code());
      PhiHistory.initialize(files, GrpFixtures.scoreStart());
      Path first = GrpFixtures.network(run.resolve("first"), 1);
      Path second = GrpFixtures.network(run.resolve("second"), 2);
      GrpFixtures.ready(files, "g007-a1", 7, first);
      GrpFixtures.ready(files, "g007-a2", 7, second);
      RefitRecord rejected = GrpFixtures.ready(files, "g007-a3", 7, second);
      rejected.with(RefitStatus.REJECTED, null, "gate", 0).append(files);

      Assert.assertEquals(prism("grp-switch", "--run", run.toString(), "--to", "g007-a1"), Exit.OK.code());
      PhiEntry switched = PhiHistory.read(files).getLast();
      Assert.assertEquals(switched.id(), "g007-a1");
      Assert.assertEquals(switched.kind(), PhiEntryKind.NETWORK);
      Assert.assertEquals(switched.effectiveFrom(), 8);
      Assert.assertEquals(switched.parent(), PhiEntry.INITIAL_ID);
      // 同じ Φ が効いていれば何も足さない。することが無くても終了コードは 0
      Assert.assertTrue(PhiControl.switchTo(run, "g007-a1", null, "test").isEmpty());
      Assert.assertEquals(prism("grp-switch", "--run", run.toString(), "--to", "g007-a1"), Exit.OK.code());
      Assert.assertEquals(PhiHistory.read(files).size(), 2);
      // 学習の JVM が動いていれば、1 反復先から効く
      try (TrainLock training = TrainLock.hold(new RunDirectory(run).trainLock())) {
        Assert.assertTrue(PhiControl.switchTo(run, "g007-a2", null, "manual").isPresent());
        Assert.assertEquals(PhiHistory.read(files).getLast().effectiveFrom(), 9);
        Assert.assertThrows(IllegalArgumentException.class, () -> PhiControl.switchTo(run, "g000", 8, "test"));
      }
      Assert.assertThrows(IllegalArgumentException.class, () -> PhiControl.switchTo(run, "g007-a2", 7, "test"));
      // 巻き戻し
      Assert.assertEquals(
          prism("grp-switch", "--run", run.toString(), "--to", "g000", "--from", "12", "--reason", "x1n3"), Exit.OK.code());
      PhiEntry back = PhiHistory.read(files).getLast();
      Assert.assertEquals(back.id(), PhiEntry.INITIAL_ID);
      Assert.assertEquals(back.kind(), PhiEntryKind.SCORE_RANK);
      Assert.assertEquals(back.reason(), "rollback: x1n3");
      Assert.assertTrue(back.rollback());
      Assert.assertEquals(PhiHistory.effectiveAt(PhiHistory.read(files), 11).orElseThrow().id(), "g007-a2");
      Assert.assertEquals(PhiHistory.effectiveAt(PhiHistory.read(files), 12).orElseThrow().id(), PhiEntry.INITIAL_ID);
      // 関門に落ちた候補と知らない id は拒む
      Assert.assertThrows(IllegalArgumentException.class, () -> PhiControl.switchTo(run, "g007-a3", 13, "test"));
      Assert.assertThrows(IllegalArgumentException.class, () -> PhiControl.switchTo(run, "g099-a1", 13, "test"));
      Assert.assertEquals(
          prism("grp-switch", "--run", run.toString(), "--to", "g099-a1", "--from", "13"), Exit.FAILED.code());
      Assert.assertEquals(PhiHistory.read(files).size(), 4);
      // 自前の GRP でない run では切り替えない
      Files.delete(new RunDirectory(run).settings());
      GrpFixtures.writeSettings(run, "phi.source=SCORE_RANK");
      Assert.assertEquals(
          prism("grp-switch", "--run", run.toString(), "--to", "g007-a2", "--from", "13"), Exit.USAGE.code());
      Assert.assertEquals(PhiHistory.read(files).size(), 4);
      // run の記録を読むコマンドは --set を受け付けない
      Assert.assertEquals(
          prism("grp-switch", "--run", run.toString(), "--to", "g007-a2", "--set", "phi.source=SELF_GRP"),
          Exit.USAGE.code());
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  /**
   * 開けない候補には切り替えず、戻しもしない。候補は sha を持たないので、チェックポイントと q を開いてみて確かめる。既に効いたことのある
   * 項目が開けないと、系譜はその反復で止まる。
   */
  @Test
  public void refusesACandidateThatCannotBeOpened() throws Exception {
    Path run = Files.createTempDirectory("prism-switch-open");
    try {
      var files = new GrpFiles(run);
      GrpFixtures.writeSettings(run, GrpFixtures.scoreStartAssignments());
      GrpFixtures.writeState(run, 8);
      PhiHistory.initialize(files, GrpFixtures.scoreStart());
      Path first = GrpFixtures.network(run.resolve("first"), 1);
      Path q = GrpFixtures.outcomes(run.resolve("q"), GrpFixtures.VARIANT, 2);
      GrpFixtures.ready(files, "g007-a1", 7, first);
      GrpFixtures.ready(files, "g007-a2", 7, GrpFixtures.network(run.resolve("second"), 3), q);
      GrpFixtures.ready(files, "g007-a3", 7, run.resolve("missing"));
      PhiControl.switchTo(run, "g007-a1", 8, "test").orElseThrow();
      PhiControl.switchTo(run, "g007-a2", 9, "test").orElseThrow();
      PhiControl.switchTo(run, PhiEntry.INITIAL_ID, 10, "test").orElseThrow();
      // 重みの大きさが形に合わないチェックポイントには戻さない
      Files.writeString(
          first.resolve(GrpCheckpoint.WEIGHTS_FILE), "x", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
      Assert.assertThrows(IllegalStateException.class, () -> PhiControl.switchTo(run, "g007-a1", 13, "test"));
      Assert.assertEquals(
          prism("grp-switch", "--run", run.toString(), "--to", "g007-a1", "--from", "13"), Exit.FAILED.code());
      // q が開けない項目にも戻さない
      Files.writeString(q.resolve(GrpCheckpoint.WEIGHTS_FILE), "x", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
      Assert.assertThrows(IllegalStateException.class, () -> PhiControl.switchTo(run, "g007-a2", 13, "test"));
      // チェックポイントの無い候補には切り替えない
      Assert.assertThrows(IllegalStateException.class, () -> PhiControl.switchTo(run, "g007-a3", 13, "test"));
      Assert.assertEquals(PhiHistory.read(files).size(), 4);
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  /**
   * run の外から来た GRP のチェックポイント（g000）は sha で照合する。大きさが同じまま中身が変わっていれば g000 に戻さず、中身を元に
   * 戻せば戻せる。
   */
  @Test
  public void refusesAChangedOutsideCheckpoint() throws Exception {
    Path run = Files.createTempDirectory("prism-switch-sha");
    try {
      var files = new GrpFiles(run);
      Path leaf = GrpFixtures.network(run.resolve("leaf"), 1);
      GrpFixtures.writeSettings(run, GrpFixtures.grpStartAssignments(leaf));
      GrpFixtures.writeState(run, 4);
      PhiHistory.initialize(files, GrpFixtures.grpStart(leaf));
      GrpFixtures.ready(files, "g003-a1", 3, GrpFixtures.network(run.resolve("b"), 3));
      PhiControl.switchTo(run, "g003-a1", 4, "test").orElseThrow();
      Path weights = leaf.resolve(GrpCheckpoint.WEIGHTS_FILE);
      byte[] original = Files.readAllBytes(weights);
      byte[] changed = original.clone();
      changed[0] ^= 1;
      Files.write(weights, changed);
      IllegalStateException refused =
          Assert.expectThrows(
              IllegalStateException.class, () -> PhiControl.switchTo(run, PhiEntry.INITIAL_ID, 6, "test"));
      Assert.assertTrue(refused.getMessage().startsWith("GRP checkpoint "), refused.getMessage());
      Assert.assertEquals(
          prism("grp-switch", "--run", run.toString(), "--to", PhiEntry.INITIAL_ID, "--from", "6"), Exit.FAILED.code());
      Assert.assertEquals(PhiHistory.read(files).size(), 2);
      Files.write(weights, original);
      Assert.assertEquals(
          prism("grp-switch", "--run", run.toString(), "--to", PhiEntry.INITIAL_ID, "--from", "6"), Exit.OK.code());
      Assert.assertEquals(PhiHistory.read(files).getLast().id(), PhiEntry.INITIAL_ID);
      Assert.assertEquals(PhiHistory.read(files).size(), 3);
    } finally {
      Snapshot.deleteTree(run);
    }
  }
}
