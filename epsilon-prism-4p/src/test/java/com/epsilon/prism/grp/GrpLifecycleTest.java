package com.epsilon.prism.grp;

import com.epsilon.agent.run.RunDirectory;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.train.GateFailure;
import com.epsilon.grp.train.GateKind;
import com.epsilon.grp.train.GrpInit;
import com.epsilon.grp.train.RecencyWindow;
import com.epsilon.grp.train.SwitchWatchVerdict;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 反復の区切りで行う手続き（{@link GrpLifecycle}）を、学習しない小さな checkpoint で確かめる。重みが 0 のネットワーク（全ての順列が同じ
 * 確率）は、重みの大きな乱数のネットワークより MSE がはっきり良い。
 *
 * <p>切り替えの後の監視: 区切りごとに 1 行を書く（前に使った別の項目と記録があれば、監視の行も書く）。8 行で判定し、ROLLBACK なら最も良い
 * 項目へ h ＋ 1 から戻して、τ を 2 倍にする。τ を広げている間のアラートは HALT になる。関門: 使用中の項目が自己対局の候補でなければ置き換えの
 * 関門、そうでなければ毎回の関門を通す。戻した後は、アンカーとの比べも足す。AUTO では合格した候補を h ＋ 1 から効かせ、既に始まった反復の
 * 区切りでは phi.jsonl に何も足さない。
 */
public class GrpLifecycleTest {
  private static final int GAMES = 60;

  private Path run;
  private GrpFiles files;
  private Path good;
  private Path bad;

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("prism-lifecycle");
    files = new GrpFiles(run);
    good = GrpFixtures.network(run.resolve("good"), 1, 0);
    bad = GrpFixtures.network(run.resolve("bad"), 2, 1.0);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(run);
  }

  /** {@code checkpoint} の GRP から始める自前の GRP の run（AUTO）を作る。g000 を書き、state.json の次のコホートを {@code next} にする。 */
  private PhiSetup start(Path checkpoint, int next) throws IOException {
    PhiSetup setup = GrpFixtures.grpStart(checkpoint, "phi.grp.switching=AUTO", "phi.grp.refit=SCHEDULE");
    GrpFixtures.writeState(run, next);
    PhiHistory.initialize(files, setup);
    return setup;
  }

  /** コホート {@code t} を、その反復に効く項目で集めた記録にする。 */
  private void record(int t) throws IOException {
    PhiEntry used = PhiHistory.effectiveAt(PhiHistory.read(files), t).orElseThrow();
    GrpFixtures.record(files, t, used, GrpFixtures.games(10_000L * t, GAMES, t));
  }

  /** 時点 {@code slot} の TRAINED の候補の行を書く（{@code checkpoint} を候補のディレクトリに写す）。 */
  private RefitRecord trained(int slot, Path checkpoint) throws IOException {
    String id = GrpRefitter.attemptId(slot, 1);
    Path candidate = files.candidate(id);
    Files.createDirectories(candidate);
    for (String name : List.of(GrpCheckpoint.JSON_FILE, GrpCheckpoint.WEIGHTS_FILE))
      Files.copy(checkpoint.resolve(name), candidate.resolve(name));
    var record =
        new RefitRecord(
            id, slot, RefitStatus.TRAINED, RefitCause.SCHEDULE, new RecencyWindow(slot, 2, 64), GrpInit.SCRATCH,
            PhiEntry.INITIAL_ID, candidate.toAbsolutePath().normalize().toString(), "", "", null, null, "", 1);
    record.append(files);
    return record;
  }

  private RefitRecord latest(String id) throws IOException {
    return RefitRecord.latest(RefitRecord.readAll(files), id).orElseThrow();
  }

  @Test
  public void aReplacementGateSwitchesFromTheNextIteration() throws Exception {
    PhiSetup setup = start(bad, 3);
    var lifecycle = new GrpLifecycle(files, setup);
    record(1);
    record(2);
    lifecycle.boundary(1);
    trained(1, good);
    lifecycle.boundary(2);
    RefitRecord gated = latest("g001-a1");
    Assert.assertEquals(gated.status(), RefitStatus.READY, gated.detail());
    Assert.assertEquals(gated.gate().kind(), GateKind.REPLACEMENT);
    Assert.assertTrue(gated.gate().passed());
    PhiEntry switched = PhiHistory.read(files).getLast();
    Assert.assertEquals(switched.id(), "g001-a1");
    Assert.assertEquals(switched.effectiveFrom(), 3);
    Assert.assertTrue(switched.reason().startsWith("AUTO"), switched.reason());
    // 区切りごとに 1 行。前に使った別の項目がないので、監視の行はない
    List<SwitchWatchRecord> watch = SwitchWatchRecord.readAll(files);
    Assert.assertEquals(watch.stream().map(SwitchWatchRecord::cohort).toList(), List.of(1, 2));
    Assert.assertNull(watch.get(1).line());
    Assert.assertEquals(lifecycle.lastBoundary(), 2);

    // 毎回の関門: 使用中の候補と同じ値の候補は通る
    GrpFixtures.writeState(run, 4);
    record(3);
    trained(2, good);
    lifecycle.boundary(3);
    RefitRecord refit = latest("g002-a1");
    Assert.assertEquals(refit.status(), RefitStatus.READY, refit.detail());
    Assert.assertEquals(refit.gate().kind(), GateKind.REFIT);
    SwitchWatchRecord line = SwitchWatchRecord.readAll(files).getLast();
    Assert.assertEquals(line.usedId(), "g001-a1");
    Assert.assertEquals(line.previousId(), PhiEntry.INITIAL_ID);
    Assert.assertTrue(line.line().mseDifference() < 0, line.toString());
    Assert.assertEquals(PhiHistory.read(files).getLast().id(), "g002-a1");
    Assert.assertEquals(PhiHistory.read(files).getLast().effectiveFrom(), 4);
  }

  /** 置き換えの関門は、使用中と同じ値の候補を通さない（−2 SE より良いことを求める）。 */
  @Test
  public void aRejectedReplacementAndALateBoundaryWriteNoEntry() throws Exception {
    PhiSetup setup = start(bad, 5);
    var lifecycle = new GrpLifecycle(files, setup);
    record(1);
    record(2);
    record(3);
    lifecycle.boundary(1);
    trained(1, bad);
    lifecycle.boundary(2);
    RefitRecord rejected = latest("g001-a1");
    Assert.assertEquals(rejected.status(), RefitStatus.REJECTED);
    Assert.assertTrue(rejected.gate().failures().contains(GateFailure.NOT_BETTER), rejected.gate().toString());
    // 反復 4 が始まった後の区切り 2・3 では、合格しても phi.jsonl に足さない
    trained(2, good);
    lifecycle.boundary(3);
    Assert.assertEquals(latest("g002-a1").status(), RefitStatus.READY);
    Assert.assertEquals(PhiHistory.read(files).size(), 1);
    Assert.assertEquals(SwitchWatchRecord.readAll(files).size(), 3);
  }

  /** 次の反復が Φ を決めた後（Φ の段の完了マーカーがある）に処理した区切りは、合格しても phi.jsonl に足さない。 */
  @Test
  public void aBoundaryAfterTheNextIterationDecidedItsPhiWritesNoEntry() throws Exception {
    PhiSetup setup = start(bad, 3);
    var lifecycle = new GrpLifecycle(files, setup);
    record(1);
    record(2);
    lifecycle.boundary(1);
    trained(1, good);
    Path decided = new RunDirectory(run).work(3).resolve("PHI.json");
    Files.createDirectories(decided.getParent());
    Files.writeString(decided, "{}");
    lifecycle.boundary(2);
    Assert.assertEquals(latest("g001-a1").status(), RefitStatus.READY);
    Assert.assertEquals(PhiHistory.read(files).size(), 1);
  }

  /** 記録の先頭の項目の id が、そのコホートに効いた項目と違えば記録を使わず、関門に通す候補は REJECTED にする。 */
  @Test
  public void gamesRecordedUnderAnotherEntryAreRefused() throws Exception {
    PhiSetup setup = start(bad, 3);
    var lifecycle = new GrpLifecycle(files, setup);
    record(1);
    GrpFixtures.record(files, 2, GrpFixtures.scoreEntry("other", 1), GrpFixtures.games(20_000L, GAMES, 2));
    lifecycle.boundary(1);
    trained(1, good);
    lifecycle.boundary(2);
    RefitRecord refused = latest("g001-a1");
    Assert.assertEquals(refused.status(), RefitStatus.REJECTED);
    Assert.assertTrue(refused.detail().contains("has no usable game records"), refused.detail());
    Assert.assertEquals(PhiHistory.read(files).size(), 1);
  }

  /** 悪い候補を 8 反復使うと監視が鳴り、最も良かった g000 へ h ＋ 1 から戻して、τ を 2 倍にする。戻した区切りでは関門を通さない。 */
  @Test
  public void theSwitchWatchRollsBackToTheBestRecentEntry() throws Exception {
    PhiSetup setup = start(good, 10);
    PhiHistory.appendUnstarted(files, new RefitRecord(
        "g000-a1", 0, RefitStatus.READY, RefitCause.MANUAL, null, GrpInit.SCRATCH, PhiEntry.INITIAL_ID, bad.toString(),
        "", "", null, null, "", 0).entry(GrpFixtures.VARIANT, 2, "test"));
    var lifecycle = new GrpLifecycle(files, setup);
    for (int t = 1; t <= 9; t++) record(t);
    for (int h = 1; h <= 8; h++) lifecycle.boundary(h);
    trained(8, bad);
    lifecycle.boundary(9);
    List<SwitchWatchRecord> watch = SwitchWatchRecord.readAll(files);
    Assert.assertEquals(watch.size(), 9);
    SwitchWatchRecord last = watch.getLast();
    Assert.assertEquals(last.verdict(), SwitchWatchVerdict.ROLLBACK);
    Assert.assertTrue(last.widened());
    Assert.assertFalse(last.halted());
    Assert.assertEquals(last.rollbackTo(), PhiEntry.INITIAL_ID);
    for (SwitchWatchRecord earlier : watch.subList(0, 8)) Assert.assertNull(earlier.verdict());
    PhiEntry back = PhiHistory.read(files).getLast();
    Assert.assertEquals(back.id(), PhiEntry.INITIAL_ID);
    Assert.assertEquals(back.effectiveFrom(), 10);
    Assert.assertTrue(back.rollback(), back.reason());
    RefitRecord refused = latest("g008-a1");
    Assert.assertEquals(refused.status(), RefitStatus.REJECTED);
    Assert.assertTrue(refused.detail().contains("rolled back"), refused.detail());
    // 広げている間は τ が 2 倍
    SwitchWatchRecord.State state = SwitchWatchRecord.state(watch);
    Assert.assertTrue(state.widened());
    GrpOperations operations = setup.selfGrp().grp();
    Assert.assertEquals(GrpRefitter.halfLife(operations, PhiHistory.read(files), state, 10), 2 * operations.halfLife());
    Assert.assertFalse(lifecycle.halted());
  }

  /** τ を広げている間にまた鳴れば、HALT で学び直しを止め、関門も通さない。 */
  @Test
  public void anAlarmWhileWidenedHalts() throws Exception {
    PhiSetup setup = start(good, 10);
    PhiHistory.appendUnstarted(files, new RefitRecord(
        "g000-a1", 0, RefitStatus.READY, RefitCause.MANUAL, null, GrpInit.SCRATCH, PhiEntry.INITIAL_ID, bad.toString(),
        "", "", null, null, "", 0).entry(GrpFixtures.VARIANT, 2, "test"));
    new SwitchWatchRecord(1, PhiEntry.INITIAL_ID, "", null, SwitchWatchVerdict.ROLLBACK, true, false, "").append(files);
    var lifecycle = new GrpLifecycle(files, setup);
    for (int t = 2; t <= 9; t++) record(t);
    for (int h = 2; h <= 8; h++) lifecycle.boundary(h);
    trained(8, good);
    lifecycle.boundary(9);
    SwitchWatchRecord last = SwitchWatchRecord.readAll(files).getLast();
    Assert.assertEquals(last.verdict(), SwitchWatchVerdict.HALT);
    Assert.assertTrue(last.halted());
    Assert.assertEquals(last.rollbackTo(), "");
    Assert.assertTrue(lifecycle.halted());
    Assert.assertEquals(PhiHistory.read(files).getLast().id(), "g000-a1", "a halt does not roll back");
    RefitRecord refused = latest("g008-a1");
    Assert.assertEquals(refused.status(), RefitStatus.REJECTED);
    Assert.assertTrue(refused.detail().contains("halted"), refused.detail());
  }

  /** 戻した後の 8 反復は、毎回の関門に、戻した項目との比べを足す。 */
  @Test
  public void refitsAfterARollbackAreAnchored() throws Exception {
    PhiSetup setup = start(bad, 6);
    PhiHistory.appendUnstarted(files, readyEntry("g001-a1", 1, good, 2));
    PhiHistory.appendUnstarted(files, readyEntry("g001-a1", 1, good, 4).from(4, PhiEntry.ROLLBACK + " test"));
    PhiHistory.appendUnstarted(files, readyEntry("g003-a1", 3, bad, 5));
    var lifecycle = new GrpLifecycle(files, setup);
    for (int t = 1; t <= 5; t++) record(t);
    for (int h = 1; h <= 4; h++) lifecycle.boundary(h);
    trained(4, bad);
    lifecycle.boundary(5);
    RefitRecord anchored = latest("g004-a1");
    Assert.assertEquals(anchored.status(), RefitStatus.REJECTED, anchored.detail());
    Assert.assertEquals(anchored.gate().kind(), GateKind.REFIT);
    Assert.assertEquals(anchored.gate().failures(), List.of(GateFailure.WORSE_THAN_ANCHOR));
  }

  private PhiEntry readyEntry(String id, int slot, Path checkpoint, int from) throws IOException {
    return GrpFixtures.ready(files, id, slot, checkpoint).entry(GrpFixtures.VARIANT, from, "test");
  }
}
