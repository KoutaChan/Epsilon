package com.epsilon.prism.grp;

import com.epsilon.grp.FinalRound;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.Phi;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.grp.train.GrpMetrics;
import com.epsilon.grp.train.GrpReport;
import com.epsilon.grp.train.HandEndValues;
import com.epsilon.grp.train.HandEnds;
import com.epsilon.grp.train.LateReport;
import com.epsilon.grp.train.NextHands;
import com.epsilon.grp.train.OutcomeCalibration;
import com.epsilon.grp.train.PhiStamp;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 裏の採点（{@link PhiScorer}）の LATE のモードのテスト。コホートは学習の JVM と同じ手順で集める。{@link PhiResolver} が反復の値を開き、
 * その Phi で局末の Φ を求めて、数のファイルのヘッダーに値の stamp を書く。
 *
 * <p>採点の行は、使ったモードの採点を持つ。この採点は、収集の Phi で作った値の列で測った結果とビットで同じ。項目が q を持てば、もう一方の
 * モードの採点、数え上げる局末の報告、q の較正も持つ。GRP のモードのコホートでは、LATE の打ち切りの値に、その反復までに効いた最後の LATE の
 * 行の値を使う。そうした行が無ければ設定の値を使う。
 *
 * <p>採点する値のモードと打ち切りの値は、数のファイルのヘッダーから取る。数のファイルが無いか読めなければ、mode.jsonl から取る。影の
 * 項目は使ったモードで採点し、LATE のモードで q を持たない影は飛ばす。もう一方のモードで失敗しても、空にするのはそのフィールドだけ。
 * 行の中身は採点のスレッドの数に依らない。
 */
public class LateMonitorTest {
  /** 設定に書く LATE の打ち切りの値。 */
  private static final double SETTINGS_CUTOFF = 5e-3;

  /** mode.jsonl の行に書く LATE の打ち切りの値。 */
  private static final double LINE_CUTOFF = 1e-3;

  /** テストの側で開く Phi が、数え上げた卓の Φ を覚えるスロットの数。値には効かない。 */
  private static final int LATE_SLOTS = 1024;

  private Path run;
  private GrpFiles files;
  private Path leaf;
  private Path q;
  private PhiSetup setup;
  private PhiEntry initial;

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("prism-late-monitor");
    files = new GrpFiles(run);
    leaf = GrpFixtures.network(run.resolve("leaf"), 11);
    q = GrpFixtures.outcomes(run.resolve("q"), GrpFixtures.VARIANT, 12);
    setup = grpStart("phi.grp.threads=2");
    initial = PhiHistory.initialize(files, setup);
    Assert.assertTrue(initial.hasOutcomes());
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(run);
  }

  /**
   * 集めたコホート。
   *
   * @param active 収集に使った値と Phi
   * @param played 記録した半荘と、数のファイルの中身
   * @param ends 記録にある局末
   * @param values 収集の Phi で求めた、局末の値の列
   */
  private record Collected(ActivePhi active, GrpFixtures.Played played, HandEnds ends, HandEndValues values) {}

  @Test(timeOut = 300_000)
  public void aLateCohortCarriesBothModesTheLateReportAndTheQCalibration() throws Exception {
    PhiHistory.appendMode(files, ModeLine.late(LINE_CUTOFF, 3, "test"));
    Collected collected;
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      collected = collect(files, resolver, setup, 3, 120);
    }
    Assert.assertEquals(collected.active().value().mode(), PhiMode.LATE);
    Assert.assertEquals(collected.active().phi().kind(), Phi.Kind.LATE);
    HandEnds ends = collected.ends();
    int covered = covered(ends);
    Assert.assertTrue(covered > 0 && covered < ends.handEnds(), covered + " / " + ends.handEnds());

    PhiMonitorLine line = PhiScorer.score(files, setup, 3).orElseThrow();
    Assert.assertEquals(line.phiId(), PhiEntry.INITIAL_ID);
    Assert.assertEquals(line.mode(), PhiMode.LATE);
    Assert.assertEquals(line.cutoff(), Double.valueOf(LINE_CUTOFF));
    // 採点に使う値は、収集の値とビットで同じ
    Assert.assertEquals(line.active(), GrpMetrics.measure(ends, collected.values()));
    HandEndValues grp = values(PhiValue.grp(initial), ends);
    Assert.assertEquals(line.other(), GrpMetrics.measure(ends, grp));
    Assert.assertEquals(line.late(), LateReport.measure(ends, collected.values(), grp));
    Assert.assertEquals(line.late().handEnds(), covered);
    Assert.assertTrue(line.late().enumeration().settled() > 0, line.late().enumeration().toString());
    OutcomeCalibration.Check calibration = calibration(collected);
    Assert.assertEquals(line.outcomeCalibration(), calibration);
    Assert.assertEquals(calibration.handEnds() + calibration.skipped(), covered);
    Assert.assertTrue(line.summary().contains(" mode=LATE late="), line.summary());
    Assert.assertFalse(line.summary().contains("late=- "), line.summary());
    Assert.assertEquals(PhiMonitorLine.readAll(files), List.of(line), "the line survives the round trip");
  }

  @Test(timeOut = 300_000)
  public void aGrpCohortMeasuresLateAtTheLastLateLineInEffectOrTheSettings() throws Exception {
    PhiHistory.appendMode(files, ModeLine.late(LINE_CUTOFF, 3, "test"));
    PhiHistory.appendMode(files, ModeLine.grp(5, "test"));
    // 反復 9 からの行は、反復 6 にはまだ効かない
    PhiHistory.appendMode(files, ModeLine.late(2e-3, 9, "test"));
    Collected second;
    Collected sixth;
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      second = collect(files, resolver, setup, 2, 100);
      sixth = collect(files, resolver, setup, 6, 100);
    }
    Assert.assertEquals(second.active().value().mode(), PhiMode.GRP);
    Assert.assertEquals(sixth.active().value().mode(), PhiMode.GRP);

    // 反復 2 ではまだ LATE の行が効いていないので、設定の打ち切りの値で測る
    PhiMonitorLine line2 = PhiScorer.score(files, setup, 2).orElseThrow();
    HandEnds ends2 = second.ends();
    Assert.assertEquals(line2.mode(), PhiMode.GRP);
    Assert.assertNull(line2.cutoff());
    Assert.assertEquals(line2.active(), GrpMetrics.measure(ends2, second.values()));
    HandEndValues late2 = values(PhiValue.of(initial, PhiMode.LATE, SETTINGS_CUTOFF), ends2);
    Assert.assertEquals(line2.other(), GrpMetrics.measure(ends2, late2));
    Assert.assertEquals(line2.late(), LateReport.measure(ends2, late2, second.values()));
    Assert.assertEquals(line2.outcomeCalibration(), calibration(second));
    Assert.assertTrue(line2.summary().contains(" mode=GRP late="), line2.summary());
    Assert.assertFalse(line2.summary().contains("late=- "), line2.summary());

    // 反復 6 では、最後に効いた LATE の行（反復 3 から効く）の打ち切りの値で測る
    PhiMonitorLine line6 = PhiScorer.score(files, setup, 6).orElseThrow();
    HandEnds ends6 = sixth.ends();
    HandEndValues late6 = values(PhiValue.of(initial, PhiMode.LATE, LINE_CUTOFF), ends6);
    Assert.assertEquals(line6.mode(), PhiMode.GRP);
    Assert.assertEquals(line6.active(), GrpMetrics.measure(ends6, sixth.values()));
    Assert.assertEquals(line6.other(), GrpMetrics.measure(ends6, late6));
    Assert.assertNotEquals(
        line6.other(), GrpMetrics.measure(ends6, values(PhiValue.of(initial, PhiMode.LATE, SETTINGS_CUTOFF), ends6)));
    Assert.assertEquals(line6.late(), LateReport.measure(ends6, late6, sixth.values()));
    Assert.assertEquals(line6.outcomeCalibration(), calibration(sixth));
  }

  @Test(timeOut = 300_000)
  public void theModeFollowsTheCountFileHeader() throws Exception {
    PhiHistory.appendMode(files, ModeLine.late(LINE_CUTOFF, 2, "test"));
    PhiValue late = PhiValue.of(initial, PhiMode.LATE, LINE_CUTOFF);
    PhiValue grp = PhiValue.grp(initial);

    // ヘッダーのモードと打ち切りの値で値を作り直す。mode.jsonl の打ち切りの値と違っても、ヘッダーを使う
    GrpFixtures.Played header3 = GrpFixtures.played(3, 3000, 60, 3, new PhiStamp("LATE", 4e-3));
    GrpFixtures.recordCounted(files, 3, initial, header3);
    PhiMonitorLine rebuilt = PhiScorer.score(files, setup, 3).orElseThrow();
    Assert.assertEquals(rebuilt.mode(), PhiMode.LATE);
    Assert.assertEquals(rebuilt.cutoff(), Double.valueOf(4e-3));
    Assert.assertEquals(rebuilt.active(), report(PhiValue.of(initial, PhiMode.LATE, 4e-3), ends(3)));
    Assert.assertNotNull(rebuilt.outcomeCalibration());

    // mode.jsonl よりヘッダーのモードを優先する
    GrpFixtures.recordCounted(files, 4, initial, GrpFixtures.played(4, 4000, 60, 4, grp.stamp()));
    PhiMonitorLine header = PhiScorer.score(files, setup, 4).orElseThrow();
    Assert.assertEquals(header.mode(), PhiMode.GRP);
    Assert.assertEquals(header.active(), report(grp, ends(4)));
    Assert.assertEquals(header.other(), report(late, ends(4)));

    // 数のファイルが無ければ、モードは mode.jsonl から取る。q の較正は無い
    GrpFixtures.record(files, 5, initial, GrpFixtures.games(5000, 60, 5));
    PhiMonitorLine missing = PhiScorer.score(files, setup, 5).orElseThrow();
    assertFromModeLines(missing, late, 5);
    Assert.assertNotNull(missing.other());
    Assert.assertNotNull(missing.late());

    // 別のコホートの数のファイル、壊れた数のファイル、知らないモードを書いたヘッダーも、読めないものとして扱う
    GrpFixtures.record(files, 6, initial, GrpFixtures.games(6000, 60, 6));
    Files.copy(files.counts(4), files.counts(6));
    assertFromModeLines(PhiScorer.score(files, setup, 6).orElseThrow(), late, 6);
    GrpFixtures.record(files, 7, initial, GrpFixtures.games(7000, 60, 7));
    Files.writeString(files.counts(7), "not a count file");
    assertFromModeLines(PhiScorer.score(files, setup, 7).orElseThrow(), late, 7);
    GrpFixtures.Played rules = GrpFixtures.played(8, 8000, 60, 8, new PhiStamp("RULES", Double.NaN));
    GrpFixtures.recordCounted(files, 8, initial, rules);
    assertFromModeLines(PhiScorer.score(files, setup, 8).orElseThrow(), late, 8);

    // 効く行が無い反復は GRP のモード
    GrpFixtures.record(files, 1, initial, GrpFixtures.games(1000, 40, 1));
    assertFromModeLines(PhiScorer.score(files, setup, 1).orElseThrow(), grp, 1);
    Assert.assertEquals(
        PhiMonitorLine.readAll(files).stream().map(PhiMonitorLine::cohort).toList(), List.of(3, 4, 5, 6, 7, 8, 1));
  }

  @Test(timeOut = 300_000)
  public void anEntryWithoutQIsScoredInTheGrpModeOnly() throws Exception {
    GrpFiles scoreFiles = new GrpFiles(run.resolve("score"));
    PhiSetup scoreSetup = GrpFixtures.scoreStart("phi.grp.monitor=SCORE", "phi.late.from=1");
    Collected scoreOnly;
    Collected network;
    try (PhiResolver resolver = PhiResolver.open(scoreSetup, scoreFiles)) {
      Assert.assertEquals(ModeLine.effectiveAt(PhiHistory.readModes(scoreFiles), 2).orElseThrow().mode(), PhiMode.LATE);
      scoreOnly = collect(scoreFiles, resolver, scoreSetup, 2, 40);
      PhiHistory.append(
          scoreFiles, GrpFixtures.ready(scoreFiles, "g001-a1", 1, leaf).entry(GrpFixtures.VARIANT, 3, "test"));
      network = collect(scoreFiles, resolver, scoreSetup, 3, 60);
    }
    for (Collected collected : List.of(scoreOnly, network)) {
      Assert.assertEquals(collected.active().value().requested(), PhiMode.LATE);
      Assert.assertEquals(collected.active().value().mode(), PhiMode.GRP);
    }
    PhiMonitorLine first = PhiScorer.score(scoreFiles, scoreSetup, 2).orElseThrow();
    PhiMonitorLine second = PhiScorer.score(scoreFiles, scoreSetup, 3).orElseThrow();
    Assert.assertEquals(second.phiId(), "g001-a1");
    for (PhiMonitorLine line : List.of(first, second)) {
      Assert.assertEquals(line.mode(), PhiMode.GRP);
      Assert.assertNull(line.cutoff());
      Assert.assertNull(line.other());
      Assert.assertNull(line.late());
      Assert.assertNull(line.outcomeCalibration());
      Assert.assertTrue(line.summary().contains(" mode=GRP late=- "), line.summary());
    }
    Assert.assertEquals(first.active(), GrpMetrics.measure(scoreOnly.ends(), scoreOnly.values()));
    Assert.assertEquals(second.active(), GrpMetrics.measure(network.ends(), network.values()));
  }

  @Test(timeOut = 300_000)
  public void shadowsAreScoredInTheUsedMode() throws Exception {
    PhiSetup shadowSetup = grpStart("phi.grp.threads=2", "phi.grp.shadow=WITH_ORIGINAL");
    Path a = GrpFixtures.network(run.resolve("a"), 21);
    Path aq = GrpFixtures.outcomes(run.resolve("aq"), GrpFixtures.VARIANT, 22);
    PhiEntry active = GrpFixtures.ready(files, "g001-a1", 1, a, aq).entry(GrpFixtures.VARIANT, 2, "test");
    PhiHistory.append(files, active);
    Path b = GrpFixtures.network(run.resolve("b"), 23);
    Path bq = GrpFixtures.outcomes(run.resolve("bq"), GrpFixtures.VARIANT, 24);
    PhiEntry withQ = GrpFixtures.ready(files, "g002-a1", 2, b, bq).entry(GrpFixtures.VARIANT, 2, "x");
    PhiEntry withoutQ =
        GrpFixtures.ready(files, "g003-a1", 3, GrpFixtures.network(run.resolve("c"), 25))
            .entry(GrpFixtures.VARIANT, 2, "x");
    PhiEntry original = PhiScorer.original(shadowSetup);
    Assert.assertTrue(original.hasOutcomes());
    PhiHistory.appendMode(files, ModeLine.late(LINE_CUTOFF, 2, "test"));
    PhiHistory.appendMode(files, ModeLine.grp(4, "test"));
    Collected third;
    Collected fourth;
    try (PhiResolver resolver = PhiResolver.open(shadowSetup, files)) {
      third = collect(files, resolver, shadowSetup, 3, 60);
      fourth = collect(files, resolver, shadowSetup, 4, 60);
    }
    Assert.assertEquals(third.active().entry().id(), "g001-a1");
    Assert.assertEquals(third.active().value().mode(), PhiMode.LATE);

    // LATE のモードでは、影も同じ打ち切りの値で求める。q を持たない影は飛ばす
    PhiMonitorLine line3 = PhiScorer.score(files, shadowSetup, 3).orElseThrow();
    Assert.assertEquals(line3.phiId(), "g001-a1");
    Assert.assertEquals(line3.active(), GrpMetrics.measure(third.ends(), third.values()));
    Assert.assertEquals(line3.shadows().keySet(), Set.of("g002-a1", PhiEntry.FIXED_ID));
    for (PhiEntry shadow : List.of(withQ, original))
      Assert.assertEquals(
          line3.shadows().get(shadow.id()),
          report(PhiValue.of(shadow, PhiMode.LATE, LINE_CUTOFF), third.ends()),
          shadow.id());

    // GRP のモードでは、影も全て GRP のモードで求める
    PhiMonitorLine line4 = PhiScorer.score(files, shadowSetup, 4).orElseThrow();
    Assert.assertEquals(line4.mode(), PhiMode.GRP);
    Assert.assertEquals(line4.shadows().keySet(), Set.of("g002-a1", "g003-a1", PhiEntry.FIXED_ID));
    for (PhiEntry shadow : List.of(withQ, withoutQ, original))
      Assert.assertEquals(line4.shadows().get(shadow.id()), report(PhiValue.grp(shadow), fourth.ends()), shadow.id());
  }

  @Test(timeOut = 300_000)
  public void anUnreadableQEmptiesOnlyItsFields() throws Exception {
    PhiHistory.appendMode(files, ModeLine.late(LINE_CUTOFF, 3, "test"));
    Collected second;
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      second = collect(files, resolver, setup, 2, 60);
      collect(files, resolver, setup, 3, 60);
    }
    Snapshot.deleteTree(q);
    // GRP のモードのコホートは、使ったモードだけを採点する
    PhiMonitorLine line = PhiScorer.score(files, setup, 2).orElseThrow();
    Assert.assertEquals(line.active(), GrpMetrics.measure(second.ends(), second.values()));
    Assert.assertNull(line.other());
    Assert.assertNull(line.late());
    Assert.assertNull(line.outcomeCalibration());
    // LATE のモードのコホートは、使ったモードの値を開けないので採点しない
    Assert.assertEquals(PhiScorer.score(files, setup, 3), Optional.empty());
    Assert.assertEquals(PhiMonitorLine.readAll(files), List.of(line));
  }

  @Test(timeOut = 300_000)
  public void theLineDoesNotDependOnTheScoringThreads() throws Exception {
    PhiHistory.appendMode(files, ModeLine.late(LINE_CUTOFF, 2, "test"));
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      collect(files, resolver, setup, 2, 100);
    }
    PhiMonitorLine one = PhiScorer.score(files, grpStart("phi.grp.threads=1"), 2).orElseThrow();
    Files.delete(files.monitor());
    PhiMonitorLine four = PhiScorer.score(files, grpStart("phi.grp.threads=4"), 2).orElseThrow();
    Assert.assertNotNull(one.late());
    Assert.assertNotNull(one.outcomeCalibration());
    Assert.assertEquals(withoutSeconds(four), withoutSeconds(one));
  }

  @Test(timeOut = 300_000)
  public void aLineWithoutTheNewFieldsReadsWithNulls() throws Exception {
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      collect(files, resolver, setup, 2, 30);
    }
    PhiMonitorLine line = PhiScorer.score(files, setup, 2).orElseThrow();
    // 前の jar が書いた行。モード、もう一方のモード、終盤の報告、q の較正のフィールドが無い
    var old = new PhiMonitorLine(3, line.phiId(), null, null, line.active(), null, null, null, Map.of(), 1.0);
    old.append(files);
    String text = Files.readAllLines(files.monitor(), StandardCharsets.UTF_8).getLast();
    for (String field : List.of("mode", "cutoff", "other", "late", "outcomeCalibration"))
      Assert.assertFalse(text.contains("\"" + field + "\""), field + " in " + text);
    List<PhiMonitorLine> lines = PhiMonitorLine.readAll(files);
    Assert.assertEquals(lines, List.of(line, old));
    PhiMonitorLine read = lines.getLast();
    Assert.assertNull(read.mode());
    Assert.assertNull(read.late());
    Assert.assertTrue(read.summary().startsWith("phi[3] g000 mse="), read.summary());
    Assert.assertTrue(read.summary().contains(" late=- "), read.summary());
    // 前の jar の行があるコホートは、採点し直さない
    GrpFixtures.record(files, 3, PhiHistory.read(files).getFirst(), GrpFixtures.games(3000, 2, 3));
    Assert.assertEquals(PhiScorer.pending(files), List.of());
  }

  /** 元の GRP（leaf と q）から始めて裏の採点をする、自前の GRP の run の設定。引数の上書きを後に重ねる。 */
  private PhiSetup grpStart(String... assignments) {
    return GrpFixtures.grpStart(
        leaf,
        GrpFixtures.concat(
            new String[] {"phi.start.outcomes=" + q, "phi.grp.monitor=SCORE", "phi.late.cutoff=" + SETTINGS_CUTOFF},
            assignments));
  }

  /**
   * 学習の JVM と同じ手順でコホート t を集める。反復 t の値を開き、その値の stamp をヘッダーに書いた数のファイルと、半荘の記録を書く。
   * 最後に、開いた Phi で局末の値の列を求める。
   */
  private static Collected collect(GrpFiles files, PhiResolver resolver, PhiSetup setup, int t, int games)
      throws Exception {
    ActivePhi active = resolver.resolve(t, setup.phi());
    GrpFixtures.Played played = GrpFixtures.played(t, 1000L * t, games, t, active.value().stamp());
    GrpFixtures.recordCounted(files, t, active.entry(), played);
    HandEnds ends =
        HandEnds.of(GrpFixtures.VARIANT, GrpFixtures.UTILITY, GameRecordCodec.read(files.records(t)).games());
    return new Collected(active, played, ends, collected(active.phi(), ends));
  }

  /**
   * 収集で使う値を返す。開いた Phi で、局末ごとの Φ を 3 つのスレッドから 2 回ずつ求める（1 つのスレッドは逆順に回り、2 回目はメモに
   * 当たる）。どの呼び出しの Φ も、その Phi で作った局末の値の列とビットで同じことを確かめてから、その列を返す。
   */
  private static HandEndValues collected(Phi phi, HandEnds ends) throws Exception {
    int threads = 3;
    int seats = ends.seats();
    long[][][] bits = new long[threads * 2][ends.handEnds()][seats];
    try (ExecutorService pool = Executors.newFixedThreadPool(threads)) {
      List<Future<?>> done = new ArrayList<>();
      for (int thread = 0; thread < threads; thread++) {
        int which = thread;
        done.add(
            pool.submit(
                () -> {
                  double[] out = new double[seats];
                  for (int pass = 0; pass < 2; pass++)
                    for (int i = 0; i < ends.handEnds(); i++) {
                      int h = which == 1 ? ends.handEnds() - 1 - i : i;
                      phi.values(ends.state(h), out);
                      for (int seat = 0; seat < seats; seat++)
                        bits[which * 2 + pass][h][seat] = Double.doubleToLongBits(out[seat]);
                    }
                }));
      }
      for (Future<?> future : done) future.get();
    }
    HandEndValues values = HandEndValues.of(ends, phi);
    for (int h = 0; h < ends.handEnds(); h++)
      for (int seat = 0; seat < seats; seat++)
        for (long[][] call : bits)
          Assert.assertEquals(call[h][seat], Double.doubleToLongBits(values.phi(h, seat)), "hand end " + h);
    return values;
  }

  /** 記録した数のファイルにある局のうち、数え上げる局で測った元の GRP の q の較正。q はチェックポイントから直に読む。 */
  private OutcomeCalibration.Check calibration(Collected collected) throws IOException {
    HandEnds ends = collected.ends();
    return OutcomeCalibration.check(
        NextHands.of(ends, collected.played().counted()),
        h -> FinalRound.covers(ends.state(h)),
        GrpCheckpoint.readOutcomes(q));
  }

  /** 数のファイルを読めず、モードと打ち切りの値を mode.jsonl から取った行であること。 */
  private void assertFromModeLines(PhiMonitorLine line, PhiValue value, int t) throws IOException {
    Assert.assertEquals(line.mode(), value.mode(), "c" + t);
    Assert.assertEquals(line.cutoff(), value.cutoff(), "c" + t);
    Assert.assertNull(line.outcomeCalibration(), "c" + t);
    Assert.assertEquals(line.active(), report(value, ends(t)), "c" + t);
  }

  private static int covered(HandEnds ends) {
    int covered = 0;
    for (int h = 0; h < ends.handEnds(); h++) if (FinalRound.covers(ends.state(h))) covered++;
    return covered;
  }

  private HandEnds ends(int t) throws IOException {
    return HandEnds.of(GrpFixtures.VARIANT, GrpFixtures.UTILITY, GameRecordCodec.read(files.records(t)).games());
  }

  private static HandEndValues values(PhiValue value, HandEnds ends) throws IOException {
    return HandEndValues.of(ends, HandEndPhi.open(value, GrpFixtures.UTILITY, LATE_SLOTS));
  }

  private static GrpReport report(PhiValue value, HandEnds ends) throws IOException {
    return GrpMetrics.measure(ends, values(value, ends));
  }

  private static PhiMonitorLine withoutSeconds(PhiMonitorLine line) {
    return new PhiMonitorLine(
        line.cohort(), line.phiId(), line.mode(), line.cutoff(), line.active(), line.other(), line.late(),
        line.outcomeCalibration(), line.shadows(), 0);
  }
}
