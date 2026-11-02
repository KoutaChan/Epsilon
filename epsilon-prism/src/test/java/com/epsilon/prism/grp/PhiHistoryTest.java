package com.epsilon.prism.grp;

import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * phi.jsonl: g000 は 1 回だけ書き再開では読むだけ（設定と食い違えば拒む）、反復 t の項目は効き始めが最も遅い行、追記は始まった反復以前と別の Φ
 * を持つ同じ id を拒み、学習の JVM が動いていれば 1 反復先から。書きかけの最後の行は読まず次の追記で切り捨てる。並んだ追記は錠で 1 行ずつ。
 * phiStart の検証も確かめる。
 */
public class PhiHistoryTest {

  private static PhiEntry score(String id, int from) {
    return new PhiEntry(id, PhiEntryKind.SCORE_RANK, "", "", "", "", from, "", "test");
  }

  private static PhiEntry self(String id, int from, String sha) {
    return new PhiEntry(id, PhiEntryKind.SELF_CHECKPOINT, "/grp/" + id, sha, "", "", from, PhiEntry.INITIAL_ID, "test");
  }

  @Test
  public void g000IsWrittenOnceAndARestartOnlyReadsIt() throws Exception {
    Path run = Files.createTempDirectory("prism-phi");
    try {
      var files = new GrpFiles(run);
      PhiEntry first = PhiHistory.initialize(files, GrpFixtures.scoreStart());
      Assert.assertEquals(first.id(), PhiEntry.INITIAL_ID);
      Assert.assertEquals(first.kind(), PhiEntryKind.SCORE_RANK);
      Assert.assertEquals(first.effectiveFrom(), 1);
      Assert.assertEquals(PhiHistory.initialize(files, GrpFixtures.scoreStart()), first);
      Assert.assertEquals(PhiHistory.read(files), List.of(first));
      // 再開で phiStart を変えた設定は拒む（黙って別の Φ で続けない）
      Path decision = run.resolve("decision");
      GrpFixtures.checkpoint(decision);
      Assert.assertThrows(IOException.class, () -> PhiHistory.initialize(files, GrpFixtures.grpStart(decision)));
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  @Test
  public void g000OfAGrpStartNamesTheExactCheckpointAndTable() throws Exception {
    Path run = Files.createTempDirectory("prism-phi");
    try {
      Path decision = run.resolve("decision");
      Path exact = GrpFixtures.checkpoint(decision);
      Path table = run.resolve("table.json");
      PrismConfig config =
          GrpFixtures.config(
              "phiSource", "SELF_GRP", "phiStart", "GRP_CHECKPOINT", "phiMonitor", "RECORD", "phiDrift", "OFF",
              "grpCheckpoint", decision.toString(), "phiTemperatures", "1.0,1.2", "phiCalibration", table.toString());
      var files = new GrpFiles(run);
      PhiEntry first = PhiHistory.initialize(files, config);
      Assert.assertEquals(first.kind(), PhiEntryKind.FIXED_CHECKPOINT);
      Assert.assertEquals(first.path(), exact.toString());
      Assert.assertEquals(first.sha256(), EpsilonGrpCheckpointManager.checkpointSha256(exact));
      Assert.assertEquals(first.temperatures(), "1.0,1.2");
      Assert.assertEquals(first.calibration(), table.toString());
      Assert.assertEquals(PhiHistory.initialize(files, config), first);
      // 表を変えた設定は拒む
      PrismConfig other =
          GrpFixtures.config(
              "phiSource", "SELF_GRP", "phiStart", "GRP_CHECKPOINT", "phiMonitor", "RECORD", "phiDrift", "OFF",
              "grpCheckpoint", decision.toString(), "phiTemperatures", "1.0,1.2");
      Assert.assertThrows(IOException.class, () -> PhiHistory.initialize(files, other));
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  @Test
  public void theLatestStartingLineIsInEffect() {
    List<PhiEntry> entries =
        List.of(score("g000", 1), self("g010-a1", 12, "a"), self("g020-a1", 30, "b"), self("g015-a1", 20, "c"), score("g000", 20));
    Assert.assertTrue(PhiHistory.effectiveAt(entries, 0).isEmpty());
    Assert.assertEquals(PhiHistory.effectiveAt(entries, 11).orElseThrow().id(), "g000");
    Assert.assertEquals(PhiHistory.effectiveAt(entries, 12).orElseThrow().id(), "g010-a1");
    // 後から足した 20 の行は 30 の行より前の反復だけに効き、同じ 20 では後の行が勝つ
    Assert.assertEquals(PhiHistory.effectiveAt(entries, 25).orElseThrow(), score("g000", 20));
    Assert.assertEquals(PhiHistory.effectiveAt(entries, 30).orElseThrow().id(), "g020-a1");
  }

  @Test
  public void appendRejectsStartedIterationsAndWaitsOneMoreWhileTraining() throws Exception {
    Path run = Files.createTempDirectory("prism-phi");
    try {
      var files = new GrpFiles(run);
      // g000 の前は何も足せない
      Assert.assertThrows(IllegalArgumentException.class, () -> PhiHistory.append(files, self("g004-a1", 9, "a")));
      PhiHistory.initialize(files, GrpFixtures.scoreStart());
      GrpFixtures.writeState(run, 5);
      Assert.assertEquals(PhiHistory.firstOpenIteration(files), 5);
      Assert.assertThrows(IllegalArgumentException.class, () -> PhiHistory.append(files, self("g004-a1", 4, "a")));
      PhiHistory.append(files, self("g004-a1", 5, "a"));
      try (PhiHistory.Locked training = PhiHistory.holdTraining(files)) {
        Assert.assertEquals(PhiHistory.firstOpenIteration(files), 6);
        Assert.assertThrows(IllegalArgumentException.class, () -> PhiHistory.append(files, self("g004-a2", 5, "b")));
        PhiHistory.append(files, self("g004-a2", 6, "b"));
        // 学習の JVM は 1 つだけ
        Assert.assertThrows(IOException.class, () -> PhiHistory.holdTraining(files));
      }
      Assert.assertEquals(PhiHistory.firstOpenIteration(files), 5);
      Assert.assertEquals(PhiHistory.read(files).size(), 3);
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  /** 裏の作業や grp-switch の探りが印の錠を一瞬持っていても、学習の JVM は待って取る（取れずに落ちると系譜が止まる）。 */
  @Test
  public void theTrainingLockWaitsOutABriefProbe() throws Exception {
    Path run = Files.createTempDirectory("prism-phi");
    try {
      var files = new GrpFiles(run);
      Path path = files.lock(PhiHistory.TRAINING_LOCK);
      Files.createDirectories(path.getParent());
      var probing = new java.util.concurrent.CountDownLatch(1);
      Thread probe =
          new Thread(
              () -> {
                try (var channel = java.nio.channels.FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
                    var lock = channel.lock()) {
                  probing.countDown();
                  Thread.sleep(500);
                } catch (IOException | InterruptedException failure) {
                  throw new IllegalStateException(failure);
                }
              });
      probe.start();
      probing.await();
      long started = System.nanoTime();
      try (PhiHistory.Locked training = PhiHistory.holdTraining(files)) {
        Assert.assertTrue(System.nanoTime() - started >= 300_000_000L, "waited for the probe");
        Assert.assertTrue(PhiHistory.trainingRunning(files));
      }
      probe.join();
      Assert.assertFalse(PhiHistory.trainingRunning(files));
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  /** 次のコホートが封印済み（当てはめの途中で落ちた反復。再開は前の Φ で集めた行を使い回す）なら、その反復は始まっている。 */
  @Test
  public void aSealedNextCohortCountsAsStarted() throws Exception {
    Path run = Files.createTempDirectory("prism-phi");
    try {
      var files = new GrpFiles(run);
      PhiHistory.initialize(files, GrpFixtures.scoreStart());
      GrpFixtures.writeState(run, 7);
      Assert.assertEquals(PhiHistory.firstOpenIteration(files), 7);
      Files.createDirectories(files.cohort(7));
      Files.writeString(files.cohort(7).resolve(com.epsilon.prism.rows.CohortWriter.MANIFEST), "{}");
      Assert.assertEquals(PhiHistory.firstOpenIteration(files), 8);
      Assert.assertThrows(IllegalArgumentException.class, () -> PhiHistory.append(files, self("g006-a1", 7, "a")));
      PhiHistory.append(files, self("g006-a1", 8, "a"));
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  @Test
  public void anIdNamesOnePhiButMayComeBack() throws Exception {
    Path run = Files.createTempDirectory("prism-phi");
    try {
      var files = new GrpFiles(run);
      PhiHistory.initialize(files, GrpFixtures.scoreStart());
      PhiHistory.append(files, self("g008-a1", 3, "a"));
      Assert.assertThrows(IllegalArgumentException.class, () -> PhiHistory.append(files, self("g008-a1", 4, "other")));
      // 巻き戻しは同じ id・同じ Φ の行
      PhiHistory.append(files, score("g000", 4));
      PhiHistory.append(files, self("g008-a1", 6, "a"));
      Assert.assertEquals(PhiHistory.effectiveAt(PhiHistory.read(files), 5).orElseThrow().kind(), PhiEntryKind.SCORE_RANK);
      Assert.assertEquals(PhiHistory.effectiveAt(PhiHistory.read(files), 6).orElseThrow().id(), "g008-a1");
      // 種類ごとの形を守らない項目は作れない
      Assert.assertThrows(
          IllegalArgumentException.class,
          () -> new PhiEntry("x", PhiEntryKind.SELF_CHECKPOINT, "/grp", "sha", "1.1", "", 2, "", ""));
      Assert.assertThrows(
          IllegalArgumentException.class, () -> new PhiEntry("x", PhiEntryKind.SCORE_RANK, "/grp", "", "", "", 2, "", ""));
      Assert.assertThrows(
          IllegalArgumentException.class, () -> new PhiEntry("x", PhiEntryKind.FIXED_CHECKPOINT, "/grp", "", "", "", 2, "", ""));
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  @Test
  public void aTornLastLineIsIgnoredAndCutByTheNextAppend() throws Exception {
    Path run = Files.createTempDirectory("prism-phi");
    try {
      var files = new GrpFiles(run);
      PhiHistory.initialize(files, GrpFixtures.scoreStart());
      Files.writeString(files.phiHistory(), "{\"id\":\"g009-a1\",\"kind\":\"SEL", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
      Assert.assertEquals(PhiHistory.read(files).size(), 1);
      PhiHistory.append(files, self("g009-a2", 2, "b"));
      List<PhiEntry> entries = PhiHistory.read(files);
      Assert.assertEquals(entries.size(), 2);
      Assert.assertEquals(entries.get(1).id(), "g009-a2");
      Assert.assertTrue(Files.readString(files.phiHistory()).endsWith("\n"));
      // 改行で終わる壊れた行は読めない（効いている項目が分からないので止める）
      Files.writeString(files.phiHistory(), "not json\n", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
      Assert.assertThrows(IOException.class, () -> PhiHistory.read(files));
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  @Test
  public void parallelAppendsLandOneLineEach() throws Exception {
    Path run = Files.createTempDirectory("prism-phi");
    try {
      var files = new GrpFiles(run);
      PhiHistory.initialize(files, GrpFixtures.scoreStart());
      List<Thread> threads = new ArrayList<>();
      List<Throwable> failures = new ArrayList<>();
      for (int w = 0; w < 8; w++) {
        int writer = w;
        threads.add(
            Thread.ofPlatform()
                .start(
                    () -> {
                      try {
                        for (int i = 0; i < 10; i++) {
                          PhiHistory.append(files, self("g" + (100 + writer) + "-a" + (i + 1), 2 + i, "s" + writer + i));
                          new RefitRecord("g" + writer + "-a" + i, RefitStatus.FAILED, RefitCause.MANUAL, null, GrpInit.WARM, "", "", "", "", null, "x", 0)
                              .append(files);
                        }
                      } catch (Throwable failure) {
                        synchronized (failures) {
                          failures.add(failure);
                        }
                      }
                    }));
      }
      for (Thread thread : threads) thread.join();
      Assert.assertEquals(failures, List.of());
      List<PhiEntry> entries = PhiHistory.read(files);
      Assert.assertEquals(entries.size(), 81);
      Set<String> ids = new HashSet<>();
      for (PhiEntry entry : entries) ids.add(entry.id());
      Assert.assertEquals(ids.size(), 81);
      Assert.assertEquals(RefitRecord.readAll(files).size(), 80);
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  /** phiStart: GRP_CHECKPOINT は grpCheckpoint が要り、点数の順位は GRP と表を拒み、SELF_GRP は記録なしでは学べない。 */
  @Test
  public void phiStartIsValidated() {
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> GrpFixtures.config("phiSource", "SELF_GRP", "phiMonitor", "RECORD", "phiDrift", "OFF").validateTraining());
    Assert.assertThrows(
        IllegalArgumentException.class,
        () ->
            GrpFixtures.config(
                    "phiSource", "SELF_GRP", "phiStart", "SCORE_RANK", "phiMonitor", "RECORD", "phiDrift", "OFF",
                    "grpCheckpoint", "/grp")
                .validate());
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> GrpFixtures.config("phiSource", "SELF_GRP", "phiStart", "SCORE_RANK", "phiMonitor", "OFF", "phiDrift", "OFF").validate());
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> GrpFixtures.config("phiSource", "FIXED_GRP", "phiStart", "SCORE_RANK", "grpCheckpoint", "/grp").validate());
    GrpFixtures.scoreStart().validateTraining();
    GrpFixtures.grpStart(Path.of("/grp")).validateTraining();
  }
}
