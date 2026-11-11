package com.epsilon.prism.grp;

import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpShape;
import com.epsilon.grp.train.CountFile;
import com.epsilon.grp.train.GrpInit;
import com.epsilon.grp.train.RecencyWindow;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.run.ControlRequests;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.run.Request;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.OptionalInt;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 時点 n での GRP の再学習 1 回（{@link GrpRefitter}）のテスト。n までの範囲の記録で学習し、候補を grp/candidates の下に試行の名前で書いて、
 * TRAINED の行を書く。試行の番号は、refits.jsonl と候補のディレクトリ（書きかけも含む）の続きから振る。使用中の Φ が点数だけなら、範囲の
 * 半荘が信頼する数（{@code grpTrustGames}）に届くまでは学習せず、届いたらゼロから学習する。WARM で前のネットワークから続けて学ぶのは、
 * 同じ形の自己対局の候補からだけ。停止のフラグで止めた試行は何も残さない。半減期 τ は、構造を替えた直後の 8 反復は 1 で、監視で範囲を
 * 広げている間は 2 倍になる。grp-refit は run の control に要求を置くだけで（置いた数だけ番号の順に待つ）、自前の GRP でない run では
 * 使い方の誤りとして拒む。
 */
public class GrpRefitterTest {
  private Path run;
  private GrpFiles files;

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("prism-refit");
    files = new GrpFiles(run);
    GrpFixtures.writeState(run, 6);
    PhiHistory.write(files.phiHistory(), PhiHistory.LINES.toJson(GrpFixtures.scoreEntry(PhiEntry.INITIAL_ID, 1)));
    for (int t = 1; t <= 5; t++)
      GrpFixtures.record(files, t, GrpFixtures.scoreEntry(PhiEntry.INITIAL_ID, 1), GrpFixtures.games(1000L * t, 30, t));
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(run);
  }

  /** 点数の順位から始める自前の GRP の run の設定。信頼する半荘は 10、スレッドは 1 で、引数の上書きを後に重ねる。 */
  private static String[] assignments(String... more) {
    return GrpFixtures.concat(
        GrpFixtures.concat(GrpFixtures.scoreStartAssignments(), "phi.grp.trustGames=10", "phi.grp.threads=1"), more);
  }

  private static PhiSetup setup(String... more) {
    return GrpFixtures.setup(assignments(more));
  }

  @Test(groups = "native")
  public void aScoreOnlyStartTrainsFromScratchAndWritesATrainedCandidate() throws Exception {
    RefitRecord record = GrpRefitter.refit(files, setup(), 5, RefitCause.SCHEDULE, () -> false).orElseThrow();
    Assert.assertEquals(record.id(), "g005-a1");
    Assert.assertEquals(record.slot(), 5);
    Assert.assertEquals(record.status(), RefitStatus.TRAINED, record.detail());
    Assert.assertEquals(record.cause(), RefitCause.SCHEDULE);
    Assert.assertEquals(record.init(), GrpInit.SCRATCH);
    Assert.assertEquals(record.parent(), PhiEntry.INITIAL_ID);
    Assert.assertEquals(record.window(), new RecencyWindow(5, 2.0, 64));
    Assert.assertNull(record.gate());
    Assert.assertTrue(record.detail().contains("games=90"), record.detail());
    Path candidate = files.candidate("g005-a1");
    Assert.assertEquals(record.path(), candidate.toAbsolutePath().normalize().toString());
    Assert.assertEquals(((GrpNetwork) GrpCheckpoint.read(candidate)).shape(), GrpRefitter.shape(GrpFixtures.VARIANT));
    Assert.assertEquals(RefitRecord.readAll(files), List.of(record));
    PhiEntry entry = record.entry(GrpFixtures.VARIANT, 7, "test");
    Assert.assertEquals(entry.kind(), PhiEntryKind.NETWORK);
    Assert.assertTrue(entry.candidate());
    Assert.assertEquals(entry.table(), "");
    // run が書いた候補とその q は sha を持たない
    Assert.assertEquals(entry.sha256(), "");
    Assert.assertEquals(entry.outcomesSha256(), "");
    // 同じ時点の試行が既にあれば学習しない
    Assert.assertTrue(GrpRefitter.refit(files, setup(), 5, RefitCause.MANUAL, () -> false).isEmpty());
  }

  @Test
  public void nothingIsTrainedWithoutEnoughGamesOrCohorts() throws Exception {
    // 点数だけの Φ からは、範囲の半荘が信頼する数に届くまで学習しない
    Assert.assertTrue(GrpRefitter.refit(files, setup("phi.grp.trustGames=91"), 5, RefitCause.SCHEDULE, () -> false).isEmpty());
    // 初めの 2 コホートは範囲に入らない
    Assert.assertTrue(GrpRefitter.refit(files, setup(), 2, RefitCause.SCHEDULE, () -> false).isEmpty());
    // まだ確定していない時点
    Assert.assertTrue(GrpRefitter.refit(files, setup(), 6, RefitCause.SCHEDULE, () -> false).isEmpty());
    Assert.assertTrue(RefitRecord.readAll(files).isEmpty());
    Assert.assertFalse(Files.exists(files.candidate("x").getParent()));
  }

  /** 残す数より古い行のコホートを消しても、grp/records と grp/counts に残る記録は確定済みのまま使える。 */
  @Test
  public void recordsOutliveTheirRemovedRows() throws Exception {
    CountFile.write(files.counts(4), GrpFixtures.played(4, 4000L, 30, 4).counted());
    for (int t = 1; t <= 5; t++) DurableFiles.delete(files.cohort(t));
    Assert.assertTrue(files.recorded(4));
    Assert.assertTrue(files.counted(4));
    Assert.assertFalse(files.counted(5), "cohort 5 has no count file");
    Assert.assertFalse(files.recorded(6), "cohort 6 was never collected");
    GrpOperations operations = setup().selfGrp().grp();
    Assert.assertEquals(PhiControl.requestRefit(run, operations, null).request(), new Request.GrpRefit(5));
    Assert.assertEquals(PhiControl.requestRefit(run, operations, 3).request(), new Request.GrpRefit(3));
    Assert.assertEquals(TrustTrial.latestCohort(files), OptionalInt.of(5));
  }

  /** 半荘の記録の先頭の項目の id が、そのコホートに効いた項目と違えば、grp-trust の測り（{@link TrustTrial}）は拒む。 */
  @Test
  public void theTrustTrialRefusesGamesRecordedUnderAnotherEntry() throws Exception {
    GrpFixtures.record(files, 4, GrpFixtures.scoreEntry("other", 1), GrpFixtures.games(4000L, 30, 4));
    IllegalArgumentException refused =
        Assert.expectThrows(IllegalArgumentException.class, () -> TrustTrial.run(files, setup(), 3, 4, 2));
    Assert.assertTrue(refused.getMessage().startsWith("Cohort 4 was recorded with other, not g000"), refused.getMessage());
  }

  /** 行を消したコホートの半荘も、学び直しの範囲に入る。 */
  @Test(groups = "native")
  public void cohortsWithRemovedRowsStayInTheWindow() throws Exception {
    for (int t = 1; t <= 4; t++) DurableFiles.delete(files.cohort(t));
    RefitRecord record = GrpRefitter.refit(files, setup(), 5, RefitCause.SCHEDULE, () -> false).orElseThrow();
    Assert.assertEquals(record.status(), RefitStatus.TRAINED, record.detail());
    Assert.assertTrue(record.detail().contains("games=90"), record.detail());
  }

  /** 手動の学び直しの候補は、次のコホートの行を消した後でも、その半荘の記録で関門に通す。 */
  @Test
  public void aLateGateUsesTheRecordsOfACohortWithRemovedRows() throws Exception {
    Path candidate = GrpFixtures.network(files.candidate("g003-a1"), 1, 0);
    new RefitRecord(
            "g003-a1", 3, RefitStatus.TRAINED, RefitCause.MANUAL, new RecencyWindow(3, 2.0, 64), GrpInit.SCRATCH,
            PhiEntry.INITIAL_ID, candidate.toString(), "", "", null, null, "", 1)
        .append(files);
    new SwitchWatchRecord(4, PhiEntry.INITIAL_ID, "", null, null, false, false, "").append(files);
    DurableFiles.delete(files.cohort(4));
    new GrpLifecycle(files, setup()).lateGate(3);
    RefitRecord gated = RefitRecord.latest(RefitRecord.readAll(files), "g003-a1").orElseThrow();
    Assert.assertNotNull(gated.gate(), gated.status() + " " + gated.detail());
  }

  @Test(groups = "native")
  public void aStopSignalDiscardsTheAttempt() throws Exception {
    Assert.assertTrue(GrpRefitter.refit(files, setup(), 5, RefitCause.SCHEDULE, () -> true).isEmpty());
    Assert.assertTrue(RefitRecord.readAll(files).isEmpty());
    Assert.assertFalse(Files.exists(files.candidate("g005-a1")));
  }

  /** 試行の番号は、候補のディレクトリと書きかけの一時ディレクトリにある番号の続きから振る。 */
  @Test(groups = "native")
  public void attemptIdsContinueAfterLeftoverDirectories() throws Exception {
    Files.createDirectories(files.candidate("g004-a1"));
    Files.createDirectories(files.candidate(".g004-a2.writing"));
    RefitRecord record = GrpRefitter.refit(files, setup(), 4, RefitCause.MANUAL, () -> false).orElseThrow();
    Assert.assertEquals(record.id(), "g004-a3");
    Assert.assertEquals(GrpRefitter.attemptId(9, 1), "g009-a1");
    Assert.assertEquals(GrpRefitter.attemptId(1024, 12), "g1024-a12");
  }

  /** WARM でも、n ＋ 1 に使う項目が同じ形の自己対局の候補のときだけ、そのネットワークから続けて学ぶ。ほかはゼロから学習する。 */
  @Test(groups = "native")
  public void warmStartsOnlyFromASelfTrainedNetworkOfTheSameShape() throws Exception {
    Path same = GrpFixtures.network(run.resolve("same"), 1);
    Path other =
        GrpFixtures.network(run.resolve("other"), new GrpShape(GrpFixtures.VARIANT, GrpRefitter.shape(GrpFixtures.VARIANT).fields(), 16, false), 2);
    PhiHistory.write(files.phiHistory(), PhiHistory.LINES.toJson(GrpFixtures.ready(files, "g001-a1", 1, same).entry(GrpFixtures.VARIANT, 4, "test")));
    PhiHistory.write(files.phiHistory(), PhiHistory.LINES.toJson(GrpFixtures.ready(files, "g001-a2", 1, other).entry(GrpFixtures.VARIANT, 5, "test")));
    RefitRecord warm = GrpRefitter.refit(files, setup("phi.grp.init=WARM"), 3, RefitCause.MANUAL, () -> false).orElseThrow();
    Assert.assertEquals(warm.init(), GrpInit.WARM);
    Assert.assertEquals(warm.parent(), "g001-a1");
    RefitRecord scratch = GrpRefitter.refit(files, setup("phi.grp.init=WARM"), 4, RefitCause.MANUAL, () -> false).orElseThrow();
    Assert.assertEquals(scratch.init(), GrpInit.SCRATCH, "another shape trains from scratch");
    Assert.assertEquals(scratch.parent(), "g001-a2");
    RefitRecord chosen = GrpRefitter.refit(files, setup("phi.grp.init=SCRATCH"), 5, RefitCause.MANUAL, () -> false).orElseThrow();
    Assert.assertEquals(chosen.init(), GrpInit.SCRATCH);
  }

  /** 半減期 τ はふだんは設定の値で、構造を替えた直後の 8 反復は 1 になる。監視で範囲を広げている間は 2 倍。 */
  @Test
  public void halfLifeFollowsSwitchesAndTheSwitchWatch() {
    GrpOperations config = setup("phi.grp.halfLife=3.0").selfGrp().grp();
    PhiEntry start = GrpFixtures.scoreEntry(PhiEntry.INITIAL_ID, 1);
    PhiEntry switched = GrpFixtures.selfEntry("g010-a1", 12, "a");
    PhiEntry refit =
        new PhiEntry(
            "g020-a1", PhiEntryKind.NETWORK, GrpFixtures.VARIANT, "/grp/b", "b", "", "", "", "", 22, "g010-a1", "AUTO");
    List<PhiEntry> history = List.of(start, switched, refit);
    var quiet = new SwitchWatchRecord.State(0, List.of(), false, false);
    var widened = new SwitchWatchRecord.State(0, List.of(), true, false);
    Assert.assertEquals(GrpRefitter.halfLife(config, history, quiet, 11), 3.0);
    Assert.assertEquals(GrpRefitter.halfLife(config, history, quiet, 12), 1.0);
    Assert.assertEquals(GrpRefitter.halfLife(config, history, quiet, 19), 1.0);
    Assert.assertEquals(GrpRefitter.halfLife(config, history, quiet, 20), 3.0);
    Assert.assertEquals(GrpRefitter.halfLife(config, history, quiet, 25), 3.0, "a refit is not a structure switch");
    Assert.assertEquals(GrpRefitter.halfLife(config, history, widened, 15), 2.0);
    Assert.assertEquals(GrpRefitter.halfLife(config, history, widened, 25), 6.0);
    // 巻き戻しの行は、構造を替えた時点に数えない
    List<PhiEntry> rolledBack = List.of(start, switched.from(40, PhiEntry.ROLLBACK + " test"));
    Assert.assertEquals(GrpRefitter.halfLife(config, rolledBack, quiet, 41), 3.0);
  }

  @Test
  public void grpRefitOnlyPlacesTheRequest() throws Exception {
    GrpFixtures.writeSettings(run, assignments());
    Assert.assertEquals(PhiControlTest.prism("grp-refit", "--run", run.toString()), Exit.OK.code());
    Assert.assertEquals(slots(), List.of(5));
    Assert.assertEquals(PhiControlTest.prism("grp-refit", "--run", run.toString(), "--slot", "4"), Exit.OK.code());
    Assert.assertEquals(slots(), List.of(5, 4));
    GrpOperations operations = setup().selfGrp().grp();
    Assert.assertThrows(IllegalArgumentException.class, () -> PhiControl.requestRefit(run, operations, 7));
    Assert.assertThrows(IllegalArgumentException.class, () -> PhiControl.requestRefit(run, operations, 2));
    Assert.assertEquals(PhiControlTest.prism("grp-refit", "--run", run.toString(), "--slot", "7"), Exit.FAILED.code());
    Assert.assertEquals(slots(), List.of(5, 4));
    Assert.assertTrue(RefitRecord.readAll(files).isEmpty(), "grp-refit does not train");
    Files.delete(new RunDirectory(run).settings());
    GrpFixtures.writeSettings(run, "phi.source=SCORE_RANK");
    Assert.assertEquals(PhiControlTest.prism("grp-refit", "--run", run.toString()), Exit.USAGE.code());
  }

  /** run の control で待っている学び直しの頼みの時点（番号の順）。 */
  private List<Integer> slots() throws IOException {
    return new ControlRequests(new RunDirectory(run))
        .pending().stream().map(request -> ((Request.GrpRefit) request.request()).slot()).toList();
  }
}
