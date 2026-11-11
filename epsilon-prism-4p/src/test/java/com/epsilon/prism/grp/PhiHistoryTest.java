package com.epsilon.prism.grp;

import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.TrainLock;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.Phi;
import com.epsilon.grp.PhiTable;
import com.epsilon.grp.train.GrpInit;
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
 * phi.jsonl のテスト。g000 は 1 回だけ書き、再開では読むだけにする（設定と食い違えば拒む）。反復 t に効く項目は、効き始めが t 以下の行の
 * うち、効き始めが最も遅い行。追記では、既に始まった反復から効く行と、同じ id で別の Φ を持つ行を拒む。学習の JVM が動いていれば、1 反復先
 * から効かせる。書きかけの最後の行は読まず、次の追記で切り捨てる。同時に追記しても、ロックで 1 行ずつ書く。Φ の始め方の設定
 * （{@code phi.start}）の検証も確かめる。
 */
public class PhiHistoryTest {

  private static PhiEntry score(String id, int from) {
    return GrpFixtures.scoreEntry(id, from);
  }

  private static PhiEntry self(String id, int from, String sha) {
    return GrpFixtures.selfEntry(id, from, sha);
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
      // 再開のときに Φ の始め方を変えた設定は拒む。黙って別の Φ で続けることはしない
      Path checkpoint = GrpFixtures.network(run.resolve("model"), 1);
      Assert.assertThrows(IOException.class, () -> PhiHistory.initialize(files, GrpFixtures.grpStart(checkpoint)));
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  @Test
  public void g000OfAGrpStartNamesTheExactCheckpointAndTable() throws Exception {
    Path run = Files.createTempDirectory("prism-phi");
    try {
      Path exact = GrpFixtures.gru(run.resolve("model"), 3);
      Path table = run.resolve("table.json");
      PhiTable.of(GrpFixtures.VARIANT, new double[] {1.0, 1.2}, -1, 2.0 / 3, new double[][] {{0.01}}).write(table);
      PhiSetup setup = GrpFixtures.grpStart(run.resolve("model/../model"), "phi.start.table=" + table);
      var files = new GrpFiles(run);
      PhiEntry first = PhiHistory.initialize(files, setup);
      Assert.assertEquals(first.kind(), PhiEntryKind.NETWORK);
      Assert.assertEquals(first.variant(), GrpFixtures.VARIANT);
      Assert.assertEquals(first.path(), exact.toString());
      Assert.assertEquals(first.sha256(), GrpCheckpoint.sha256(exact));
      Assert.assertEquals(first.table(), table.toString());
      Assert.assertEquals(first.tableSha256(), PhiTable.sha256(table));
      Assert.assertEquals(PhiHistory.initialize(files, setup), first);
      Assert.assertEquals(HandEndPhi.open(first, GrpFixtures.UTILITY).kind(), Phi.Kind.GRU);
      // 表を変えた設定は拒む
      Assert.assertThrows(IOException.class, () -> PhiHistory.initialize(files, GrpFixtures.grpStart(exact)));
      // 表の中身が変われば、項目を開かない
      PhiTable.of(GrpFixtures.VARIANT, new double[] {1.3}, -1, 2.0 / 3, new double[][] {{0.01}}).write(table);
      Assert.assertThrows(IOException.class, () -> HandEndPhi.open(first, GrpFixtures.UTILITY));
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
    // 後から足した効き始め 20 の行は、効き始め 30 の行より前の反復にだけ効く。効き始めが同じ 20 の行どうしなら、後の行が効く
    Assert.assertEquals(PhiHistory.effectiveAt(entries, 25).orElseThrow(), score("g000", 20));
    Assert.assertEquals(PhiHistory.effectiveAt(entries, 30).orElseThrow().id(), "g020-a1");
  }

  @Test
  public void appendRejectsStartedIterationsAndWaitsOneMoreWhileTraining() throws Exception {
    Path run = Files.createTempDirectory("prism-phi");
    try {
      var files = new GrpFiles(run);
      // g000 を書く前は何も足せない
      Assert.assertThrows(IllegalArgumentException.class, () -> PhiHistory.append(files, self("g004-a1", 9, "a")));
      PhiHistory.initialize(files, GrpFixtures.scoreStart());
      GrpFixtures.writeState(run, 5);
      Assert.assertEquals(PhiHistory.firstOpenIteration(files), 5);
      Assert.assertThrows(IllegalArgumentException.class, () -> PhiHistory.append(files, self("g004-a1", 4, "a")));
      PhiHistory.append(files, self("g004-a1", 5, "a"));
      try (TrainLock training = TrainLock.hold(new RunDirectory(run).trainLock())) {
        Assert.assertEquals(PhiHistory.firstOpenIteration(files), 6);
        Assert.assertThrows(IllegalArgumentException.class, () -> PhiHistory.append(files, self("g004-a2", 5, "b")));
        PhiHistory.append(files, self("g004-a2", 6, "b"));
        // 学習の JVM は 1 つしか動かせない
        Assert.assertThrows(IOException.class, () -> TrainLock.hold(new RunDirectory(run).trainLock()));
      }
      Assert.assertEquals(PhiHistory.firstOpenIteration(files), 5);
      Assert.assertEquals(PhiHistory.read(files).size(), 3);
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  /**
   * 裏の処理や grp-switch が、学習中かを調べるためにロックを一瞬だけ取っていても、学習の JVM は待ってからロックを取る。取れずに落ちると、
   * 系譜が止まってしまうため。
   */
  @Test
  public void theTrainingLockWaitsOutABriefProbe() throws Exception {
    Path run = Files.createTempDirectory("prism-phi");
    try {
      var files = new GrpFiles(run);
      Path path = new RunDirectory(run).trainLock();
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
      try (TrainLock training = TrainLock.hold(path)) {
        Assert.assertTrue(System.nanoTime() - started >= 300_000_000L, "waited for the probe");
        Assert.assertTrue(PhiHistory.trainingRunning(files));
      }
      probe.join();
      Assert.assertFalse(PhiHistory.trainingRunning(files));
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  /**
   * 次のコホートが確定済みなら、その反復は始まっているとみなす。これは学習の途中で落ちた反復で、再開すると前の Φ で集めた行を使い回す。
   */
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
      // 巻き戻しは、同じ id で同じ Φ を持つ行として足せる
      PhiHistory.append(files, score("g000", 4));
      PhiHistory.append(files, self("g008-a1", 6, "a"));
      Assert.assertEquals(PhiHistory.effectiveAt(PhiHistory.read(files), 5).orElseThrow().kind(), PhiEntryKind.SCORE_RANK);
      Assert.assertEquals(PhiHistory.effectiveAt(PhiHistory.read(files), 6).orElseThrow().id(), "g008-a1");
      // 種類ごとに決まったフィールドの組を守らない項目は作れない
      Assert.assertThrows(
          IllegalArgumentException.class,
          () -> new PhiEntry("x", PhiEntryKind.NETWORK, GrpFixtures.VARIANT, "/grp", "", "", "", "", "", 2, "", ""));
      Assert.assertThrows(
          IllegalArgumentException.class,
          () -> new PhiEntry("x", PhiEntryKind.SCORE_RANK, GrpFixtures.VARIANT, "/grp", "", "", "", "", "", 2, "", ""));
      Assert.assertThrows(
          IllegalArgumentException.class,
          () ->
              new PhiEntry(
                  "x", PhiEntryKind.NETWORK, GrpFixtures.VARIANT, "/grp", "sha", "/t.json", "", "", "", 2, "", ""));
      Assert.assertThrows(
          IllegalArgumentException.class,
          () -> new PhiEntry("x", PhiEntryKind.SCORE_RANK, GrpFixtures.VARIANT, "", "", "", "", "", "", 0, "", ""));
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
      // 改行で終わる壊れた行があれば、読まずに止める。どの項目が効いているか分からないため
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
                          new RefitRecord("g" + writer + "-a" + i, writer, RefitStatus.FAILED, RefitCause.MANUAL, null, GrpInit.WARM, "", "", "", "", null, null, "x", 0)
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

  /**
   * g000 の Φ の検証。元の GRP から始めるには、チェックポイントが要る。点数の順位から始めるなら、GRP と表の指定を拒む。自前の GRP は、
   * 記録しなければ学習できない。固定の GRP は g000 を持たない。どれも、設定を読み込むときに拒む。
   */
  @Test
  public void phiStartIsValidated() {
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> GrpFixtures.setup("phi.source=SELF_GRP", "phi.start.kind=GRP_CHECKPOINT", "phi.grp.monitor=RECORD"));
    Assert.assertThrows(IllegalArgumentException.class, () -> GrpFixtures.scoreStart("phi.start.grp=/grp"));
    Assert.assertThrows(IllegalArgumentException.class, () -> GrpFixtures.scoreStart("phi.start.table=/table.json"));
    Assert.assertThrows(IllegalArgumentException.class, () -> GrpFixtures.scoreStart("phi.grp.monitor=OFF"));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> GrpFixtures.setup("phi.source=FIXED_GRP", "phi.start.kind=SCORE_RANK", "phi.grp=/grp"));
    Assert.assertTrue(GrpFixtures.scoreStart().selfPlay());
    Assert.assertTrue(GrpFixtures.grpStart(Path.of("/grp")).selfPlay());
  }

  /** 前に使っていた別の Φ を探すときは、今と同じ Φ の行を飛ばす。 */
  @Test
  public void previousAtSkipsTheSameValue() {
    List<PhiEntry> entries =
        List.of(score("g000", 1), self("g003-a1", 4, "a"), self("g003-a1", 6, "a"), self("g007-a1", 9, "b"));
    Assert.assertTrue(PhiHistory.previousAt(entries, 3).isEmpty());
    Assert.assertEquals(PhiHistory.previousAt(entries, 7).orElseThrow().id(), "g000");
    Assert.assertEquals(PhiHistory.previousAt(entries, 9).orElseThrow().id(), "g003-a1");
    Assert.assertTrue(PhiHistory.previousAt(entries, 0).isEmpty());
  }
}
