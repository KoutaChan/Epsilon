package com.epsilon.prism.grp;

import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.Phi;
import com.epsilon.grp.PhiTable;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 学習の JVM の側で、Φ の候補を使えないときの扱いを確かめる。まだ効いたことのない候補が、消えた・壊れた・READY でないときは、
 * REJECTED の行と、前の項目に戻す行を書いて続ける（再開しても同じ結果になる）。既に効いたことのある項目が読めなければ止める。有限でない値を
 * 持つ表は開かない。拒否の行は、効く時点を持つ。学習の失敗は refits.jsonl の行にして続ける（{@link GrpRefitterTest}）。
 */
public class FailureTableTest {

  @Test
  public void candidatesThatCannotBeTrustedAreRejectedAndThePreviousPhiStays() throws Exception {
    Path run = Files.createTempDirectory("prism-failure");
    try {
      var files = new GrpFiles(run);
      GrpFixtures.writeState(run, 1);
      Path good = GrpFixtures.network(run.resolve("good"), 1);
      Path gone = GrpFixtures.network(run.resolve("gone"), 2);
      PhiSettings phi = GrpFixtures.scoreStart().phi();
      try (PhiResolver resolver = PhiResolver.open(GrpFixtures.scoreStart(), files)) {
        resolver.resolve(1, phi);
        // 4: checkpoint が消えた
        PhiHistory.append(files, GrpFixtures.ready(files, "g002-a1", 2, gone).entry(GrpFixtures.VARIANT, 4, "test"));
        Snapshot.deleteTree(gone);
        // 5: checkpoint が壊れた
        Path broken = GrpFixtures.network(run.resolve("broken"), 3);
        PhiHistory.append(files, GrpFixtures.ready(files, "g003-a1", 3, broken).entry(GrpFixtures.VARIANT, 5, "test"));
        Path weights = broken.resolve(GrpCheckpoint.WEIGHTS_FILE);
        Files.write(weights, Arrays.copyOf(Files.readAllBytes(weights), Float.BYTES));
        // 6: refits.jsonl に READY の行がない
        PhiHistory.append(
            files,
            new PhiEntry(
                "g004-a1", PhiEntryKind.NETWORK, GrpFixtures.VARIANT, good.toString(), "", "", "", "", "", 6,
                PhiEntry.INITIAL_ID, "test"));
        for (int t = 4; t <= 6; t++) {
          ActivePhi active = resolver.resolve(t, phi);
          Assert.assertEquals(active.entry().id(), PhiEntry.INITIAL_ID, "t=" + t);
          Assert.assertEquals(active.entry().kind(), PhiEntryKind.SCORE_RANK, "t=" + t);
        }
        List<RefitRecord> records = RefitRecord.readAll(files);
        for (String id : List.of("g002-a1", "g003-a1")) {
          RefitRecord rejected = RefitRecord.latest(records, id).orElseThrow();
          Assert.assertEquals(rejected.status(), RefitStatus.REJECTED, id);
          Assert.assertEquals(rejected.slot(), Integer.parseInt(id.substring(1, 4)), id);
        }
        Assert.assertTrue(RefitRecord.latest(records, "g004-a1").isEmpty());
        List<PhiEntry> entries = PhiHistory.read(files);
        for (int t = 4; t <= 6; t++) {
          PhiEntry effective = PhiHistory.effectiveAt(entries, t).orElseThrow();
          Assert.assertEquals(effective.id(), PhiEntry.INITIAL_ID);
          Assert.assertTrue(effective.reason().startsWith("rejected g00" + (t - 2) + "-a1"), effective.reason());
        }
        // 拒んだ候補は grp-switch でも入れ直せない
        Assert.assertThrows(IllegalArgumentException.class, () -> PhiControl.switchTo(run, "g003-a1", null, "test"));
      }
      // 再開しても同じ履歴になる
      try (PhiResolver resolver = PhiResolver.open(GrpFixtures.scoreStart(), files)) {
        Assert.assertEquals(resolver.resolve(5, phi).entry().kind(), PhiEntryKind.SCORE_RANK);
        // 既に効いたことのある項目（時点 7 で効いた候補）は、読めなくなったら止める
        PhiHistory.append(files, GrpFixtures.ready(files, "g005-a1", 5, good).entry(GrpFixtures.VARIANT, 7, "test"));
        Assert.assertEquals(resolver.resolve(7, phi).entry().id(), "g005-a1");
      }
      Snapshot.deleteTree(good);
      try (PhiResolver resolver = PhiResolver.open(GrpFixtures.scoreStart(), files)) {
        Assert.assertThrows(IOException.class, () -> resolver.resolve(8, phi));
      }
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  @Test
  public void anUnreadableG000StopsTheLineage() throws Exception {
    Path run = Files.createTempDirectory("prism-failure");
    try {
      Path checkpoint = GrpFixtures.network(run.resolve("model"), 3);
      var files = new GrpFiles(run);
      PhiSetup setup = GrpFixtures.grpStart(checkpoint);
      try (PhiResolver resolver = PhiResolver.open(setup, files)) {
        Assert.assertEquals(resolver.resolve(1, setup.phi()).entry().kind(), PhiEntryKind.NETWORK);
      }
      Snapshot.deleteTree(checkpoint);
      try (PhiResolver resolver = PhiResolver.open(setup, files)) {
        Assert.assertThrows(IOException.class, () -> resolver.resolve(2, setup.phi()));
      }
      Assert.assertTrue(RefitRecord.readAll(files).isEmpty(), "g000 is never rejected");
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  /** 有限でない値を持つ補正の表は開かない（ラベルに届く前に止める）。補正の表は GruSinkhorn にだけ重ねられる。 */
  @Test
  public void nonFiniteTablesAndTablesOnNetworksAreRefused() throws Exception {
    Path run = Files.createTempDirectory("prism-failure");
    try {
      Path gru = GrpFixtures.gru(run.resolve("gru"), 5);
      Path table = run.resolve("nan.json");
      Files.writeString(
          table,
          "{\"variant\": \"" + GrpFixtures.VARIANT + "\", \"temperatures\": [1.0], \"low\": -1, \"high\": 0.667,"
              + " \"corrections\": [[NaN, 0.0]]}");
      var files = new GrpFiles(run);
      PhiSetup nan = GrpFixtures.grpStart(gru, "phi.start.table=" + table);
      try (PhiResolver resolver = PhiResolver.open(nan, files)) {
        Assert.assertThrows(IOException.class, () -> resolver.resolve(1, nan.phi()));
      }
      Path good = run.resolve("good.json");
      PhiTable.of(GrpFixtures.VARIANT, new double[] {1.0}, -1, 2.0 / 3, new double[][] {{0.0, 0.01}})
          .write(good);
      PhiEntry gruEntry = HandEndPhi.checkpointEntry("x", GrpFixtures.VARIANT, gru, good.toString(), null, 1, "", "test");
      Assert.assertEquals(HandEndPhi.open(gruEntry, GrpFixtures.UTILITY).kind(), Phi.Kind.GRU);
      Path network = GrpFixtures.network(run.resolve("network"), 7);
      PhiEntry overlaid =
          HandEndPhi.checkpointEntry("y", GrpFixtures.VARIANT, network, good.toString(), null, 1, "", "test");
      Assert.assertThrows(IllegalStateException.class, () -> HandEndPhi.open(overlaid, GrpFixtures.UTILITY));
      // ルールの違う項目は開かない
      PhiEntry three =
          new PhiEntry(
              "z", PhiEntryKind.NETWORK, Variant.TENHOU_3P, network.toString(),
              GrpCheckpoint.sha256(network), "", "", "", "", 1, "", "test");
      Assert.assertThrows(IllegalStateException.class, () -> HandEndPhi.open(three, GrpFixtures.UTILITY));
      // 点数だけの項目は checkpoint を読まない
      Assert.assertEquals(
          HandEndPhi.open(PhiEntry.scoreOnly(PhiEntryKind.SCORE_SMOOTH, GrpFixtures.VARIANT), GrpFixtures.UTILITY).kind(),
          Phi.Kind.SCORE_SMOOTH);
    } finally {
      Snapshot.deleteTree(run);
    }
  }
}
