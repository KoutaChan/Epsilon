package com.epsilon.prism.grp;

import com.epsilon.agent.app.Exit;
import com.epsilon.agent.run.ControlRequest;
import com.epsilon.agent.run.ControlRequests;
import com.epsilon.agent.run.Request;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.grp.train.GrpInit;
import com.epsilon.grp.train.GrpTrainer;
import com.epsilon.grp.train.RecencyWindow;
import com.epsilon.prism.run.Snapshot;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Stream;
import java.util.zip.GZIPOutputStream;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 学習の JVM で GRP を裏で回す処理（{@link GrpBackground}）のテスト。自前の GRP の run でなければ、要求されても何もしない。1 回の処理では、
 * まだ採点していないコホートを 1 度だけ採点する。済んでいないコホートの区切りを古い順に済ませ（再起動の後にまとめて追いつく場合を含む）、
 * 待つ側は区切りが済むまで待つ。run の control にある手動の学び直しの頼みには 1 回だけ従って control/done に移し、学習した候補をその次の
 * コホートの関門に通す。従うまでは、起動し直しても頼みが残る。関門を通った候補を次の反復から使うのは、その反復がまだ Φ を決めていない
 * ときだけ。
 */
public class GrpBackgroundTest {
  private Path run;
  private GrpFiles files;
  private PhiSetup setup;

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("prism-background");
    files = new GrpFiles(run);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(run);
  }

  private static final PhiEntry SCORE_RANK = PhiEntry.scoreOnly(PhiEntryKind.SCORE_RANK, GrpFixtures.VARIANT);

  /** 記録と採点のキーは自前の GRP の設定の中にしかないので、Φ の出どころがほかの run では何もしない。 */
  @Test
  public void runsWithoutTheSelfPlayGrpDoNothing() throws Exception {
    PhiSetup setup = GrpFixtures.setup("phi.source=SCORE_RANK");
    GrpFixtures.writeState(run, 4);
    GrpFixtures.record(files, 3, SCORE_RANK, GrpFixtures.games(0, 20, 3));
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.request();
      background.request();
      Assert.assertTrue(background.awaitBoundary(3, Duration.ZERO), "only SELF_GRP waits for boundaries");
    }
    Assert.assertFalse(Files.exists(files.monitor()), "nothing is scored");
    Assert.assertFalse(Files.exists(files.switchWatch()));
  }

  @Test
  public void aPassScoresEachRecordedCohortOnce() throws Exception {
    PhiSetup setup = GrpFixtures.scoreStart("phi.grp.monitor=SCORE");
    GrpFixtures.writeState(run, 4);
    PhiEntry initial = PhiHistory.initialize(files, setup);
    GrpFixtures.record(files, 3, initial, GrpFixtures.games(0, 20, 3));
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.pass();
      background.pass();
    }
    List<PhiMonitorLine> lines = PhiMonitorLine.readAll(files);
    Assert.assertEquals(lines.stream().map(PhiMonitorLine::cohort).toList(), List.of(3));
    Assert.assertEquals(lines.getFirst().phiId(), PhiEntry.INITIAL_ID);
  }

  /** 再起動の後は、済んでいない区切りを古い順に済ませる。既に始まった反復の区切りには、監視の記録だけを書く。 */
  @Test(timeOut = 120_000)
  public void restartsCatchUpTheBoundariesInOrder() throws Exception {
    PhiSetup setup = GrpFixtures.scoreStart();
    GrpFixtures.writeState(run, 4);
    PhiEntry initial = PhiHistory.initialize(files, setup);
    for (int t = 1; t <= 3; t++) GrpFixtures.record(files, t, initial, GrpFixtures.games(100L * t, 20, t));
    // 確定したが半荘の記録の無いコホートは、区切りを済ませても監視の行を持たない
    GrpFixtures.seal(files, 4);
    GrpFixtures.writeState(run, 4);
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.request();
      Assert.assertTrue(background.awaitBoundary(4, Duration.ofSeconds(100)));
    }
    List<SwitchWatchRecord> watch = SwitchWatchRecord.readAll(files);
    Assert.assertEquals(watch.stream().map(SwitchWatchRecord::cohort).toList(), List.of(1, 2, 3, 4));
    for (SwitchWatchRecord record : watch) Assert.assertNull(record.line());
    Assert.assertEquals(PhiHistory.read(files), List.of(initial));
    // もう一度回しても、済んだ区切りは済ませ直さない
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.pass();
    }
    Assert.assertEquals(SwitchWatchRecord.readAll(files).size(), 4);
  }

  /**
   * 収集の段が確定した後、半荘の記録をコホートから grp/records へ移す前に落ちたコホートは、移し終えるまで確定したとみなさない（記録の無い
   * コホートと取り違えて区切りを済ませないため）。
   */
  @Test(timeOut = 120_000)
  public void aCohortWhoseRecordsAreStillInsideIsNotSealedUntilTheyMove() throws Exception {
    PhiSetup setup = GrpFixtures.scoreStart();
    GrpFixtures.writeState(run, 3);
    PhiEntry initial = PhiHistory.initialize(files, setup);
    for (int t = 1; t <= 3; t++) GrpFixtures.record(files, t, initial, GrpFixtures.games(100L * t, 20, t));
    Files.move(files.records(3), files.unmovedRecords(3));
    Assert.assertFalse(files.sealed(3));
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.request();
      Assert.assertTrue(background.awaitBoundary(2, Duration.ofSeconds(100)));
      Assert.assertFalse(background.awaitBoundary(3, Duration.ofSeconds(1)), "the cohort with unmoved records waits");
    }
    Assert.assertEquals(SwitchWatchRecord.readAll(files).stream().map(SwitchWatchRecord::cohort).toList(), List.of(1, 2));
    Files.move(files.unmovedRecords(3), files.records(3));
    Assert.assertTrue(files.sealed(3));
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.request();
      Assert.assertTrue(background.awaitBoundary(3, Duration.ofSeconds(100)));
    }
    Assert.assertEquals(
        SwitchWatchRecord.readAll(files).stream().map(SwitchWatchRecord::cohort).toList(), List.of(1, 2, 3));
  }

  /** 止めた run に grp-refit で置いた頼みには、次の起動の裏の作業が 1 回だけ従い、control/done に移す。 */
  @Test(timeOut = 120_000)
  public void aRequestPlacedOnAStoppedRunIsObeyedOnceAtTheNextStart() throws Exception {
    PhiSetup setup = recordedRun(4);
    ControlRequests requests = new ControlRequests(new RunDirectory(run));
    requests.markStopped("Stopped at ITERATION by a signal");
    Assert.assertEquals(PhiControlTest.prism("grp-refit", "--run", run.toString(), "--slot", "3"), Exit.OK.code());
    ControlRequest placed = requests.pending().getFirst();
    Assert.assertEquals(placed.request(), new Request.GrpRefit(3));
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.pass();
      Assert.assertEquals(requests.pending(), List.of());
      Assert.assertEquals(retiredRequests(), List.of(placed));
      background.pass();
    }
    Assert.assertEquals(retiredRequests(), List.of(placed));
    Assert.assertTrue(RefitRecord.readAll(files).isEmpty(), "the score-only Φ waits for trustGames games");
  }

  /** 頼みは従うまで control に残る。時点のコホートが確定するまで待ち、閉じ始めた起動で従えなかった頼みには次の起動で従う。 */
  @Test(timeOut = 120_000)
  public void aRequestStaysUntilObeyedAcrossRestarts() throws Exception {
    PhiSetup setup = recordedRun(3);
    ControlRequests requests = new ControlRequests(new RunDirectory(run));
    ControlRequest placed = requests.submit(new Request.GrpRefit(4));
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.pass();
    }
    Assert.assertEquals(requests.pending(), List.of(placed), "cohort 4 is not sealed yet");
    GrpFixtures.record(files, 4, PhiHistory.read(files).getFirst(), GrpFixtures.games(400, 20, 4));
    GrpFixtures.writeState(run, 5);
    var closing = new AtomicReference<GrpBackground>();
    try (GrpBackground background = GrpBackground.start(files, setup, () -> closing.get().close())) {
      closing.set(background);
      background.pass();
    }
    Assert.assertEquals(requests.pending(), List.of(placed), "a closing JVM leaves the request");
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.pass();
    }
    Assert.assertEquals(requests.pending(), List.of());
    Assert.assertEquals(retiredRequests(), List.of(placed));
  }

  /** control cancel が指した頼みには従わず、消すのは学習の JVM の安全点の確かめに任せる。 */
  @Test(timeOut = 120_000)
  public void aCancelledRequestIsNotObeyed() throws Exception {
    PhiSetup setup = recordedRun(4);
    String directory = run.toString();
    Assert.assertEquals(PhiControlTest.prism("grp-refit", "--run", directory, "--slot", "3"), Exit.OK.code());
    Assert.assertEquals(PhiControlTest.prism("control", "cancel", "--run", directory, "--request", "1"), Exit.OK.code());
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.pass();
    }
    ControlRequests requests = new ControlRequests(new RunDirectory(run));
    Assert.assertEquals(
        requests.pending().stream().map(ControlRequest::request).toList(),
        List.of(new Request.GrpRefit(3), new Request.Cancel(1)));
    Assert.assertEquals(retiredRequests(), List.of());
  }

  /** 手動の頼みに従って学び、学習した候補を次のコホートの関門に通す（区切りは既に済んでいる）。 */
  @Test(groups = "native", timeOut = 300_000)
  public void aManualRequestTrainsAndGatesTheSlot() throws Exception {
    PhiSetup setup = GrpFixtures.scoreStart("phi.grp.trustGames=10", "phi.grp.threads=1");
    GrpFixtures.writeState(run, 5);
    PhiEntry initial = PhiHistory.initialize(files, setup);
    for (int t = 1; t <= 4; t++) GrpFixtures.record(files, t, initial, GrpFixtures.games(100L * t, 40, t));
    ControlRequests requests = new ControlRequests(new RunDirectory(run));
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.pass();
      Assert.assertTrue(RefitRecord.readAll(files).isEmpty(), "MANUAL does not train on schedule");
      ControlRequest placed = requests.submit(new Request.GrpRefit(3));
      background.pass();
      Assert.assertEquals(retiredRequests(), List.of(placed));
    }
    Assert.assertEquals(requests.pending(), List.of(), "the request is obeyed");
    List<RefitRecord> records = RefitRecord.readAll(files);
    Assert.assertEquals(records.stream().map(RefitRecord::id).distinct().toList(), List.of("g003-a1"));
    Assert.assertEquals(records.getFirst().status(), RefitStatus.TRAINED);
    Assert.assertEquals(records.getFirst().cause(), RefitCause.MANUAL);
    RefitRecord gated = records.getLast();
    Assert.assertTrue(
        gated.status() == RefitStatus.READY || gated.status() == RefitStatus.REJECTED, gated.status().toString());
    Assert.assertNotNull(gated.gate());
    Assert.assertEquals(PhiHistory.read(files), List.of(initial), "PROPOSE does not switch");
  }

  /**
   * 外れた GRP で始める AUTO の run。コホート 1 と 2 を記録し、次のコホートを 3 にする。
   *
   * @return g000
   */
  private PhiEntry startAuto() throws IOException {
    Path bad = GrpFixtures.network(run.resolve("bad"), 2, 1.0);
    setup = GrpFixtures.grpStart(bad, "phi.grp.switching=AUTO");
    GrpFixtures.writeState(run, 3);
    PhiEntry initial = PhiHistory.initialize(files, setup);
    for (int t = 1; t <= 2; t++) GrpFixtures.record(files, t, initial, GrpFixtures.games(10_000L * t, 60, t));
    return initial;
  }

  /** 時点 1 の TRAINED の候補（置き換えの関門を通る GRP）を置く。 */
  private void trainCandidate() throws IOException {
    String id = GrpRefitter.attemptId(1, 1);
    Path candidate = GrpFixtures.network(files.candidate(id), 1, 0);
    new RefitRecord(
            id, 1, RefitStatus.TRAINED, RefitCause.MANUAL, new RecencyWindow(1, 2, 64), GrpInit.SCRATCH,
            PhiEntry.INITIAL_ID, candidate.toString(), "", "", null, null, "", 1)
        .append(files);
  }

  /** 時点 1 の手動の学び直しを頼む。候補はもうあるので、次の処理は学習せずに、候補をコホート 2 の関門に通す。 */
  private void requestSlotOne() throws IOException {
    new ControlRequests(new RunDirectory(run)).submit(new Request.GrpRefit(1));
  }

  private RefitStatus candidateStatus() throws IOException {
    return RefitRecord.latest(RefitRecord.readAll(files), GrpRefitter.attemptId(1, 1)).orElseThrow().status();
  }

  /** 区切りの後にできた候補は、次の反復がまだ Φ を決めていなければ、その反復から使う。 */
  @Test(timeOut = 120_000)
  public void aLateGateSwitchesWhileTheNextIterationHasNotDecidedItsPhi() throws Exception {
    startAuto();
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.pass();
      trainCandidate();
      requestSlotOne();
      background.pass();
      Assert.assertTrue(background.awaitBoundary(2, Duration.ZERO));
    }
    Assert.assertEquals(candidateStatus(), RefitStatus.READY);
    PhiEntry switched = PhiHistory.read(files).getLast();
    Assert.assertEquals(switched.id(), "g001-a1");
    Assert.assertEquals(switched.effectiveFrom(), 3);
  }

  /** 次の反復の Φ の段が区切りを通った後は、後からの関門が通っても phi.jsonl に足さない（その反復は前の Φ で集める）。 */
  @Test(timeOut = 120_000)
  public void aLateGateAppendsNothingOnceTheNextPhiStepPassed() throws Exception {
    PhiEntry initial = startAuto();
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.pass();
      Assert.assertTrue(background.awaitBoundary(2, Duration.ZERO), "iteration 3 decides its Φ");
      trainCandidate();
      requestSlotOne();
      background.pass();
    }
    Assert.assertEquals(candidateStatus(), RefitStatus.READY);
    Assert.assertEquals(PhiHistory.read(files), List.of(initial), "iteration 3 collects with g000");
  }

  /** 再起動の前に次の反復が Φ を決めていれば（Φ の段の完了マーカーがある）、再起動の後の関門も phi.jsonl に足さない。 */
  @Test(timeOut = 120_000)
  public void aLateGateAfterARestartAppendsNothingToAnIterationWithADecidedPhi() throws Exception {
    PhiEntry initial = startAuto();
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.pass();
    }
    Path decided = new RunDirectory(run).work(3).resolve("PHI.json");
    Files.createDirectories(decided.getParent());
    Files.writeString(decided, "{}");
    trainCandidate();
    requestSlotOne();
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.pass();
    }
    Assert.assertEquals(candidateStatus(), RefitStatus.READY);
    Assert.assertEquals(PhiHistory.read(files), List.of(initial), "iteration 3 collects with g000");
  }

  /**
   * 処理に失敗した区切りは済ませたとみなすので、次の反復の Φ の段が通る。その後で区切りを済ませ直しても、phi.jsonl には足さない。
   */
  @Test(timeOut = 120_000)
  public void aBoundaryRedoneAfterTheNextPhiStepPassedAppendsNothing() throws Exception {
    PhiEntry initial = startAuto();
    trainCandidate();
    Path kept = run.resolve("c00002.kept");
    Files.move(files.records(2), kept);
    Files.writeString(files.records(2), "not a game record file");
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.pass();
      Assert.assertEquals(
          SwitchWatchRecord.readAll(files).stream().map(SwitchWatchRecord::cohort).toList(), List.of(1));
      Assert.assertTrue(background.awaitBoundary(2, Duration.ZERO), "iteration 3 decides its Φ");
      Files.move(kept, files.records(2), StandardCopyOption.REPLACE_EXISTING);
      background.pass();
    }
    Assert.assertEquals(
        SwitchWatchRecord.readAll(files).stream().map(SwitchWatchRecord::cohort).toList(), List.of(1, 2));
    Assert.assertEquals(candidateStatus(), RefitStatus.READY);
    Assert.assertEquals(PhiHistory.read(files), List.of(initial), "iteration 3 collects with g000");
  }

  /**
   * 監視の行を書いた後、候補の関門が済む前に止まった区切りは、次の起動が関門を済ませる。次の反復がまだ Φ を決めていなければ、通った候補を
   * その反復から使う。
   */
  @Test(timeOut = 120_000)
  public void aBoundaryThatStoppedAfterItsWatchLineIsGatedAtTheNextStart() throws Exception {
    startAuto();
    trainCandidate();
    Path candidate = files.candidate(GrpRefitter.attemptId(1, 1));
    Path kept = run.resolve("candidate.kept");
    Files.move(candidate, kept);
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.pass();
    }
    Assert.assertEquals(
        SwitchWatchRecord.readAll(files).stream().map(SwitchWatchRecord::cohort).toList(), List.of(1, 2));
    Assert.assertEquals(candidateStatus(), RefitStatus.TRAINED, "the gate of boundary 2 failed after its watch line");
    Files.move(kept, candidate);
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.pass();
      Assert.assertTrue(background.awaitBoundary(2, Duration.ZERO));
    }
    Assert.assertEquals(candidateStatus(), RefitStatus.READY);
    PhiEntry switched = PhiHistory.read(files).getLast();
    Assert.assertEquals(switched.id(), "g001-a1");
    Assert.assertEquals(switched.effectiveFrom(), 3);
    Assert.assertEquals(SwitchWatchRecord.readAll(files).size(), 2);
  }

  /** 学習の途中で control cancel が指した頼みは、学習を止めて試行を捨てる。頼みを消すのは安全点の確かめに任せる。 */
  @Test(groups = "native", timeOut = 300_000)
  public void aCancelDuringTheTrainingStopsIt() throws Exception {
    // エポックの上限と打ち切りを届かない数にして、取り消すまで学習を続けさせる
    PhiSetup setup =
        GrpFixtures.scoreStart(
            "phi.grp.trustGames=10",
            "phi.grp.threads=1",
            "phi.grp.training.scratchEpochs=1000000000",
            "phi.grp.training.stopPatience=1000000000");
    GrpFixtures.writeState(run, 5);
    PhiEntry initial = PhiHistory.initialize(files, setup);
    for (int t = 1; t <= 4; t++) GrpFixtures.record(files, t, initial, GrpFixtures.games(100L * t, 40, t));
    ControlRequests requests = new ControlRequests(new RunDirectory(run));
    ControlRequest placed = requests.submit(new Request.GrpRefit(3));
    GrpBackground background = GrpBackground.start(files, setup, () -> {});
    Thread worker = new Thread(background::pass, "grp-background-pass");
    worker.setDaemon(true);
    ControlRequest cancel;
    try {
      worker.start();
      while (!training(worker)) {
        Assert.assertTrue(worker.isAlive(), "the pass ended before the training began");
        Thread.sleep(1);
      }
      cancel = requests.submit(new Request.Cancel(placed.number()));
      worker.join(30_000);
      Assert.assertFalse(worker.isAlive(), "the cancel stops the training");
    } finally {
      background.close();
      worker.join();
    }
    Assert.assertTrue(RefitRecord.readAll(files).isEmpty(), "the stopped attempt writes nothing");
    Assert.assertEquals(requests.pending(), List.of(placed, cancel));
    Assert.assertEquals(retiredRequests(), List.of());
  }

  /** スレッドが GRP のネットワークの学習の中にいれば true。 */
  private static boolean training(Thread thread) {
    for (StackTraceElement frame : thread.getStackTrace())
      if (frame.getClassName().equals(GrpTrainer.class.getName())) return true;
    return false;
  }

  /** 記録が読めずに落ちた頼みも、従ったものとして control/done に移す（残すと、裏の作業が動くたびに同じ学び直しをやり直す）。 */
  @Test(timeOut = 120_000)
  public void aRequestWhoseRecordsCannotBeReadIsRetired() throws Exception {
    PhiSetup setup = recordedRun(4);
    ControlRequests requests = new ControlRequests(new RunDirectory(run));
    try (GrpBackground background = GrpBackground.start(files, setup, () -> {})) {
      background.pass();
      // 半荘の数が int の最大の記録: 読むと、数の分の半荘が無いままファイルが終わる
      try (DataOutputStream out = new DataOutputStream(new GZIPOutputStream(Files.newOutputStream(files.records(3))))) {
        out.writeUTF(GrpFixtures.VARIANT.name());
        out.writeInt(3);
        out.writeUTF("");
        out.writeUTF("");
        out.writeInt(Integer.MAX_VALUE);
      }
      Assert.expectThrows(IOException.class, () -> GameRecordCodec.read(files.records(3)));
      ControlRequest placed = requests.submit(new Request.GrpRefit(3));
      background.pass();
      Assert.assertEquals(requests.pending(), List.of());
      Assert.assertEquals(retiredRequests(), List.of(placed));
    }
  }

  /**
   * 点数の順位で始め、学び直しは手動の run の設定の行を書き、コホート 1〜last を 20 半荘ずつ記録する。範囲の半荘が trustGames に届かない
   * ので、学び直しの頼みに従っても学ばない。
   */
  private PhiSetup recordedRun(int last) throws IOException {
    GrpFixtures.writeSettings(run, GrpFixtures.scoreStartAssignments());
    PhiSetup setup = GrpFixtures.scoreStart();
    GrpFixtures.writeState(run, last + 1);
    PhiEntry initial = PhiHistory.initialize(files, setup);
    for (int t = 1; t <= last; t++) GrpFixtures.record(files, t, initial, GrpFixtures.games(100L * t, 20, t));
    return setup;
  }

  /** control/done に移した要求（番号の順）。 */
  private List<ControlRequest> retiredRequests() throws IOException {
    Path done = new RunDirectory(run).control().resolve("done");
    if (!Files.isDirectory(done)) return List.of();
    List<ControlRequest> requests = new ArrayList<>();
    try (Stream<Path> listed = Files.list(done)) {
      for (Path file : listed.filter(path -> path.getFileName().toString().endsWith(".json")).toList())
        requests.add(RecordFile.read(ControlRequest.class, file).value());
    }
    requests.sort(Comparator.comparingInt(ControlRequest::number));
    return requests;
  }
}
