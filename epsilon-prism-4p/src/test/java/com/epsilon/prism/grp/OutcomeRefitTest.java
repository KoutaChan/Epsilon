package com.epsilon.prism.grp;

import com.epsilon.agent.learn.GrpTraining;
import com.epsilon.grp.FinalRound;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.train.CountFile;
import com.epsilon.grp.train.GateCohort;
import com.epsilon.grp.train.GateKind;
import com.epsilon.grp.train.GrpInit;
import com.epsilon.grp.train.GrpTrainingRule;
import com.epsilon.grp.train.OutcomeGate;
import com.epsilon.grp.train.OutcomeRows;
import com.epsilon.grp.train.OutcomeTrainer;
import com.epsilon.grp.train.Paired;
import com.epsilon.grp.train.RecencyWindow;
import com.epsilon.mahjong.util.SeedMixer;
import com.epsilon.prism.run.Snapshot;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;
import java.util.stream.Stream;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 再学習の候補と q のテスト。候補は GRP のネットワークと q（局の結果のネットワーク）の組で、どの q を使うかは、関門で GRP の判定を
 * 済ませた後に決まる。
 *
 * <p>再学習（{@link GrpRefitter}）では、範囲の数のファイルにある局のうち、数え上げる局から q を学習する。範囲、半荘の分け方、学習の
 * しかたはネットワークと同じで、数のファイルの無いコホートは外す。GRP のネットワークで失敗すれば試行ごと FAILED になり、q だけで失敗すれば q を持たない TRAINED
 * になる。q の学習の途中で停止のフラグが立っても、何も残さない。q の有無は、GRP の候補の重みにも関門の結果にも効かない。run が書いた
 * 候補の q は sha を持たず、run の外から引き継いだ q だけが sha を持つ。
 *
 * <p>関門（{@link GrpLifecycle} の区切りの関門と、手動の再学習の関門）では、READY かどうかは GRP の関門だけで決まる。q の関門に落ちたとき、
 * 候補が q を持たないとき、数のファイルが無いか読めないときは、使用中の項目の q を引き継ぐ。AUTO と grp-switch で足す行は、こうして決まった
 * q を持つ。
 */
public class OutcomeRefitTest {
  private static final int GAMES = 60;

  /**
   * 再学習の run での、コホートごとの半荘の数。合成の半荘は飛びで早く終わることが多いので、早期終了用の検証の半荘（範囲のごく一部）にも
   * 数え上げる局が入るだけの数にする。
   */
  private static final int REFIT_GAMES = 120;

  /** 再学習が q の乱数の種に混ぜるソルト。 */
  private static final long OUTCOME_SALT = 0x71726566L;

  private static final OutcomeShape SHAPE = GrpRefitter.outcomeShape(GrpFixtures.VARIANT);

  private Path run;
  private Path good;
  private Path bad;
  private Path uniform;
  private Path wild;

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("prism-outcome-refit");
    good = GrpFixtures.network(run.resolve("good"), 1, 0);
    bad = GrpFixtures.network(run.resolve("bad"), 2, 1.0);
    uniform = GrpFixtures.outcomes(run.resolve("uniform"), SHAPE, 3, 0);
    wild = GrpFixtures.outcomes(run.resolve("wild"), SHAPE, 4, 1.0);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(run);
  }

  // 関門

  /** 使用中の q より悪い q は関門に落ち、候補は READY のまま使用中の q を引き継ぐ。AUTO で足す行も、引き継いだ q を持つ。 */
  @Test
  public void aFailedQGateKeepsTheUsedQAndTheCandidateStaysReady() throws Exception {
    GrpFiles files = files("kept");
    PhiSetup setup = start(files, bad, uniform, "phi.grp.switching=AUTO");
    PhiEntry initial = PhiHistory.read(files).getFirst();
    var lifecycle = new GrpLifecycle(files, setup);
    recordCounted(files, 1);
    recordCounted(files, 2);
    lifecycle.boundary(1);
    trained(files, 1, good, wild);
    lifecycle.boundary(2);
    RefitRecord gated = latest(files, "g001-a1");
    Assert.assertEquals(gated.status(), RefitStatus.READY, gated.detail());
    Assert.assertEquals(gated.gate().kind(), GateKind.REPLACEMENT);
    OutcomeGate.Report q = gated.outcomeGate();
    Assert.assertNotNull(q, gated.detail());
    Assert.assertEquals(q.kind(), OutcomeGate.Kind.REFIT);
    Assert.assertFalse(q.passed());
    Assert.assertTrue(q.versus().mean() > 0, q.toString());
    Assert.assertTrue(q.handEnds() > 0, q.toString());
    Assert.assertEquals(gated.outcomes(), initial.outcomes());
    Assert.assertEquals(gated.outcomesSha256(), initial.outcomesSha256());
    Assert.assertTrue(gated.detail().startsWith("REPLACEMENT gate at c2; q kept from g000: failed q REFIT gate at c2: mean +"),
        gated.detail());
    PhiEntry switched = PhiHistory.read(files).getLast();
    Assert.assertEquals(switched.id(), "g001-a1");
    Assert.assertEquals(switched.outcomes(), initial.outcomes());
    Assert.assertEquals(switched.outcomesSha256(), initial.outcomesSha256());

    // 使用中の q と同じ値の q は、毎回の関門（+2 SE 以内）を通り、候補は自分の q を持つ
    GrpFixtures.writeState(files.run(), 4);
    recordCounted(files, 3);
    Path same = GrpFixtures.outcomes(files.run().resolve("same"), SHAPE, 5, 0);
    trained(files, 2, good, same);
    lifecycle.boundary(3);
    RefitRecord refit = latest(files, "g002-a1");
    Assert.assertEquals(refit.status(), RefitStatus.READY, refit.detail());
    Assert.assertEquals(refit.gate().kind(), GateKind.REFIT);
    Assert.assertEquals(refit.outcomeGate().kind(), OutcomeGate.Kind.REFIT);
    Assert.assertTrue(refit.outcomeGate().passed(), refit.outcomeGate().toString());
    Path own = files.candidateOutcomes("g002-a1").toAbsolutePath().normalize();
    Assert.assertEquals(refit.outcomes(), own.toString());
    Assert.assertEquals(refit.outcomesSha256(), "", "the run's own q carries no sha");
    Assert.assertTrue(refit.detail().startsWith("REFIT gate at c3; q REFIT gate at c3: mean +0.000000 SE 0.000000 over "),
        refit.detail());
    PhiEntry next = PhiHistory.read(files).getLast();
    Assert.assertEquals(next.id(), "g002-a1");
    Assert.assertEquals(next.outcomes(), own.toString());
  }

  /** 使用中の項目が q を持たなければ、一様な q と比べる。それより良くない q は落ち、候補は q を持たないまま READY になる。 */
  @Test
  public void withoutAUsedQTheFirstGateComparesWithTheUniformQ() throws Exception {
    GrpFiles files = files("first");
    RefitRecord gated = gated(files, good, uniform, null);
    Assert.assertEquals(gated.status(), RefitStatus.READY, gated.detail());
    Assert.assertEquals(gated.outcomeGate().kind(), OutcomeGate.Kind.FIRST);
    Assert.assertFalse(gated.outcomeGate().passed(), "the uniform q is not better than itself");
    Assert.assertFalse(gated.hasOutcomes());
    Assert.assertEquals(gated.outcomesSha256(), "");
    Assert.assertTrue(gated.detail().contains("; q kept from g000 (none): failed q FIRST gate at c2: mean "), gated.detail());
    Assert.assertFalse(PhiHistory.read(files).getLast().hasOutcomes());
  }

  /** 候補が q を持たないとき、数のファイルが無いとき、数のファイルが半荘の記録と合わないときも、候補は READY のまま使用中の q を引き継ぐ。 */
  @Test
  public void aMissingQOrCountFileKeepsTheUsedQ() throws Exception {
    RefitRecord none = gated(files("none"), good, null, uniform);
    Assert.assertEquals(none.status(), RefitStatus.READY, none.detail());
    Assert.assertNull(none.outcomeGate());
    Assert.assertTrue(none.detail().endsWith("; q kept from g000: the refit learned no outcome network"), none.detail());
    Assert.assertEquals(none.outcomes(), uniform.toString());

    GrpFiles uncounted = files("uncounted");
    var lifecycle = new GrpLifecycle(uncounted, start(uncounted, bad, uniform));
    recordCounted(uncounted, 1);
    GrpFixtures.record(uncounted, 2, used(uncounted, 2), GrpFixtures.played(2, 20_000L, GAMES, 2).games());
    lifecycle.boundary(1);
    trained(uncounted, 1, good, wild);
    lifecycle.boundary(2);
    RefitRecord missing = latest(uncounted, "g001-a1");
    Assert.assertEquals(missing.status(), RefitStatus.READY, missing.detail());
    Assert.assertNull(missing.outcomeGate());
    Assert.assertTrue(missing.detail().endsWith("; q kept from g000: cohort 2 has no count file"), missing.detail());
    Assert.assertEquals(missing.outcomes(), uniform.toString());

    GrpFiles mismatched = files("mismatched");
    lifecycle = new GrpLifecycle(mismatched, start(mismatched, bad, uniform));
    recordCounted(mismatched, 1);
    GrpFixtures.record(mismatched, 2, used(mismatched, 2), GrpFixtures.played(2, 20_000L, GAMES, 2).games());
    CountFile.write(mismatched.counts(2), GrpFixtures.played(2, 20_000L, GAMES, 99).counted());
    lifecycle.boundary(1);
    trained(mismatched, 1, good, uniform);
    lifecycle.boundary(2);
    RefitRecord unreadable = latest(mismatched, "g001-a1");
    Assert.assertEquals(unreadable.status(), RefitStatus.READY, unreadable.detail());
    Assert.assertNull(unreadable.outcomeGate());
    Assert.assertTrue(unreadable.detail().contains("; q kept from g000: java.lang.IllegalArgumentException: "),
        unreadable.detail());
    Assert.assertEquals(unreadable.outcomes(), uniform.toString());
  }

  /** 使用中の q が開けないときは、関門を測らずにその q を引き継ぐ。q のせいで GRP の更新を止めることはしない。 */
  @Test
  public void anUnreadableUsedQIsKeptAsItIs() throws Exception {
    GrpFiles files = files("tampered");
    Path used = GrpFixtures.outcomes(files.run().resolve("used-q"), SHAPE, 6, 0);
    var lifecycle = new GrpLifecycle(files, start(files, bad, used));
    recordCounted(files, 1);
    recordCounted(files, 2);
    lifecycle.boundary(1);
    trained(files, 1, good, uniform);
    byte[] weights = Files.readAllBytes(used.resolve(GrpCheckpoint.WEIGHTS_FILE));
    weights[0] ^= 1;
    Files.write(used.resolve(GrpCheckpoint.WEIGHTS_FILE), weights);
    lifecycle.boundary(2);
    RefitRecord gated = latest(files, "g001-a1");
    Assert.assertEquals(gated.status(), RefitStatus.READY, gated.detail());
    Assert.assertNull(gated.outcomeGate());
    Assert.assertEquals(gated.outcomes(), used.toString());
    Assert.assertTrue(gated.detail().contains("; q kept from g000: java.io.IOException: Outcome network "), gated.detail());
  }

  /** READY か REJECTED か、そして GRP の関門の判定は、候補の q の有無にも良し悪しにも依らない。 */
  @Test
  public void theGrpGateAloneDecidesReadiness() throws Exception {
    RefitRecord without = gated(files("good-none"), good, null, uniform);
    for (Path q : List.of(uniform, wild)) {
      RefitRecord with = gated(files("good-" + q.getFileName()), good, q, uniform);
      Assert.assertEquals(with.status(), RefitStatus.READY, with.detail());
      Assert.assertEquals(with.gate(), without.gate());
      Assert.assertEquals(head(with.detail()), head(without.detail()));
    }
    RefitRecord rejected = gated(files("bad-none"), bad, null, uniform);
    Assert.assertEquals(rejected.status(), RefitStatus.REJECTED, rejected.detail());
    for (Path q : List.of(uniform, wild)) {
      RefitRecord with = gated(files("bad-" + q.getFileName()), bad, q, uniform);
      Assert.assertEquals(with.status(), RefitStatus.REJECTED, with.detail());
      Assert.assertEquals(with.gate(), rejected.gate());
      Assert.assertEquals(with.detail(), rejected.detail());
      Assert.assertNull(with.outcomeGate(), "a rejected candidate is not q gated");
    }
  }

  /** 手動の再学習の関門でも、q の関門に落ちれば使用中の q を引き継いで READY にする。grp-switch で足す行は、その q を持つ。 */
  @Test
  public void theLateGateKeepsTheUsedQAndGrpSwitchCarriesIt() throws Exception {
    GrpFiles files = files("late");
    var lifecycle = new GrpLifecycle(files, start(files, bad, uniform));
    recordCounted(files, 1);
    recordCounted(files, 2);
    lifecycle.boundary(1);
    lifecycle.boundary(2);
    Assert.assertTrue(RefitRecord.readAll(files).isEmpty());
    trained(files, 1, good, wild);
    lifecycle.lateGate(1);
    RefitRecord gated = latest(files, "g001-a1");
    Assert.assertEquals(gated.status(), RefitStatus.READY, gated.detail());
    Assert.assertFalse(gated.outcomeGate().passed());
    Assert.assertEquals(gated.outcomes(), uniform.toString());
    Assert.assertTrue(gated.detail().contains("; q kept from g000: failed q REFIT gate at c2"), gated.detail());
    Assert.assertEquals(PhiHistory.read(files).size(), 1, "PROPOSE appends nothing");
    PhiEntry switched = PhiControl.switchTo(files.run(), "g001-a1", null, "test").orElseThrow();
    Assert.assertEquals(switched.outcomes(), uniform.toString());
    Assert.assertEquals(switched.outcomesSha256(), GrpCheckpoint.sha256(uniform));
  }

  // 記録

  /**
   * q の sha は q を持つときだけ持てる（run の外から引き継いだ q）。{@code with} は q を残し、{@code withOutcomes} は q だけを替え、
   * {@code entry} は q を項目に入れる。
   */
  @Test
  public void refitRecordsCarryTheirQ() throws Exception {
    GrpFiles files = files("record");
    String sha = GrpCheckpoint.sha256(uniform);
    Assert.assertEquals(record("g001-a1", uniform.toString(), "", null).outcomesSha256(), "");
    Assert.assertThrows(IllegalArgumentException.class, () -> record("g001-a1", "", sha, null));
    var report = new OutcomeGate.Report(OutcomeGate.Kind.FIRST, false, new Paired(Double.NaN, Double.NaN), 0, 3);
    RefitRecord record = record("g001-a1", uniform.toString(), sha, report);
    RefitRecord ready = record.with(RefitStatus.READY, null, "test", 2);
    Assert.assertEquals(ready.outcomes(), uniform.toString());
    Assert.assertEquals(ready.outcomesSha256(), sha);
    Assert.assertEquals(ready.outcomeGate(), report);
    RefitRecord cleared = ready.withOutcomes("", "", null);
    Assert.assertFalse(cleared.hasOutcomes());
    Assert.assertNull(cleared.outcomeGate());
    Assert.assertEquals(cleared.status(), RefitStatus.READY);
    Assert.assertEquals(cleared.detail(), "test");
    PhiEntry entry = ready.entry(GrpFixtures.VARIANT, 5, "test");
    Assert.assertEquals(entry.outcomes(), uniform.toString());
    Assert.assertEquals(entry.outcomesSha256(), sha);
    Assert.assertFalse(cleared.entry(GrpFixtures.VARIANT, 5, "test").hasOutcomes());

    // NaN を含む判定も読み書きできる。q のフィールドが無い古い行は、q を持たない行として読む
    ready.append(files);
    JsonObject old = JsonParser.parseString(PhiHistory.LINES.toJson(cleared)).getAsJsonObject();
    old.remove("outcomes");
    old.remove("outcomesSha256");
    old.remove("outcomeGate");
    PhiHistory.write(files.refits(), old.toString());
    List<RefitRecord> read = RefitRecord.readAll(files);
    Assert.assertEquals(read, List.of(ready, cleared));
  }

  // 再学習（torch で学習する）

  /**
   * q は、範囲の数のファイルにある局のうち、数え上げる局から学習する。範囲、分け方、学習のしかたはネットワークと同じ。数のファイルの無い
   * コホートは外す。
   */
  @Test(groups = "native")
  public void aRefitLearnsQFromTheCountedHandsOfItsWindow() throws Exception {
    GrpFiles files = refitRun("window", true);
    Files.delete(files.counts(4));
    PhiSetup setup = refitSetup();
    RefitRecord record = GrpRefitter.refit(files, setup, 5, RefitCause.SCHEDULE, () -> false).orElseThrow();
    Assert.assertEquals(record.status(), RefitStatus.TRAINED, record.detail());
    Assert.assertEquals(record.window(), new RecencyWindow(5, 2.0, 64));
    Assert.assertTrue(record.detail().startsWith("games=" + 3 * REFIT_GAMES + " "), record.detail());
    Path q = files.candidateOutcomes("g005-a1").toAbsolutePath().normalize();
    Assert.assertEquals(record.outcomes(), q.toString());
    Assert.assertEquals(record.outcomesSha256(), "", "the run's own q carries no sha");
    Assert.assertEquals(GrpCheckpoint.readOutcomes(q).shape(), SHAPE);
    Assert.assertNull(record.outcomeGate());
    PhiEntry entry = record.entry(GrpFixtures.VARIANT, 7, "test");
    Assert.assertEquals(entry.outcomes(), q.toString());
    Assert.assertEquals(entry.outcomesSha256(), record.outcomesSha256());

    // 範囲 3..5 のうち、数のファイルがある 3 と 5 だけから作った行
    OutcomeRows.Builder builder =
        OutcomeRows.builder(GrpFixtures.VARIANT, record.window(), GateCohort.NEXT, FinalRound::covers);
    for (int t : new int[] {3, 5}) builder.add(t, CountFile.read(files.counts(t)));
    OutcomeRows.Split rows = builder.build();
    Assert.assertTrue(
        record.detail().contains(
            "; q init=SCRATCH games=" + 2 * REFIT_GAMES + " rows=" + rows.train().size() + " stopRows=" + rows.stop().size() + " excluded=0 unexplained=0 "),
        record.detail());
    // 同じ行を、GRP と同じ学習のしかたと q の乱数の種で学習した q と、重みが同じ
    OutcomeNetwork again =
        OutcomeTrainer.train(
                SHAPE,
                null,
                rows.train(),
                rows.stop(),
                rule(setup, GrpInit.SCRATCH, SeedMixer.derive(setup.seed(), OUTCOME_SALT, 5, 1)),
                1,
                () -> false)
            .network();
    Assert.assertEquals(GrpCheckpoint.write(again, files.run().resolve("again")), GrpCheckpoint.sha256(q));
  }

  /** 数のファイルが 1 つも無ければ、q を持たない候補になる。それでも GRP の候補の重みと関門の判定は、q を学習したときと同じ。 */
  @Test(groups = "native")
  public void theGrpCandidateAndItsGateDoNotDependOnQ() throws Exception {
    RefitRecord[] gated = new RefitRecord[2];
    RefitRecord[] trained = new RefitRecord[2];
    for (int arm = 0; arm < 2; arm++) {
      boolean counted = arm == 0;
      GrpFiles files = refitRun(counted ? "counted" : "uncounted", counted);
      PhiSetup setup = refitSetup();
      trained[arm] = GrpRefitter.refit(files, setup, 5, RefitCause.SCHEDULE, () -> false).orElseThrow();
      Assert.assertEquals(trained[arm].status(), RefitStatus.TRAINED, trained[arm].detail());
      GrpFixtures.writeState(files.run(), 7);
      GrpFixtures.Played sixth = GrpFixtures.played(6, 6000L, REFIT_GAMES, 6);
      if (counted) GrpFixtures.recordCounted(files, 6, used(files, 6), sixth);
      else GrpFixtures.record(files, 6, used(files, 6), sixth.games());
      new GrpLifecycle(files, setup).boundary(6);
      gated[arm] = latest(files, "g005-a1");
    }
    Assert.assertTrue(trained[0].hasOutcomes(), trained[0].detail());
    Assert.assertFalse(trained[1].hasOutcomes());
    Assert.assertTrue(trained[1].detail().endsWith("; q none: no TRAIN or STOP rows in 0 counted games of " + trained[1].window()),
        trained[1].detail());
    Assert.assertEquals(
        GrpCheckpoint.sha256(Path.of(trained[0].path())), GrpCheckpoint.sha256(Path.of(trained[1].path())));
    Assert.assertEquals(head(trained[0].detail()), head(trained[1].detail()));
    Assert.assertEquals(gated[0].status(), gated[1].status());
    Assert.assertEquals(gated[0].gate(), gated[1].gate());
    Assert.assertEquals(head(gated[0].detail()), head(gated[1].detail()));
    if (gated[1].status() == RefitStatus.READY)
      Assert.assertTrue(gated[1].detail().endsWith("; q kept from g000 (none): the refit learned no outcome network"),
          gated[1].detail());
  }

  /** GRP のネットワークを書けなければ、q を学習できていても試行ごと FAILED になる。q だけが失敗すれば、q を持たない TRAINED になる。 */
  @Test(groups = "native")
  public void onlyTheGrpNetworkFailsTheAttempt() throws Exception {
    GrpFiles files = refitRun("failures", true);
    Path candidates = files.candidate("x").getParent();
    Files.createDirectories(candidates.getParent());
    Files.writeString(candidates, "not a directory");
    RefitRecord failed = GrpRefitter.refit(files, refitSetup(), 4, RefitCause.SCHEDULE, () -> false).orElseThrow();
    Assert.assertEquals(failed.status(), RefitStatus.FAILED, failed.detail());
    Assert.assertEquals(failed.path(), "");
    Assert.assertFalse(failed.hasOutcomes());
    Files.delete(candidates);

    Files.write(files.counts(5), "not a count file".getBytes(StandardCharsets.UTF_8));
    RefitRecord trained = GrpRefitter.refit(files, refitSetup(), 5, RefitCause.SCHEDULE, () -> false).orElseThrow();
    Assert.assertEquals(trained.status(), RefitStatus.TRAINED, trained.detail());
    Assert.assertEquals(trained.path(), files.candidate("g005-a1").toAbsolutePath().normalize().toString());
    Assert.assertEquals(GrpCheckpoint.read(files.candidate("g005-a1")).variant(), GrpFixtures.VARIANT);
    Assert.assertFalse(trained.hasOutcomes());
    Assert.assertFalse(Files.exists(files.candidateOutcomes("g005-a1")));
    Assert.assertTrue(trained.detail().contains("; q none: java.util.zip.ZipException"), trained.detail());
    Assert.assertEquals(RefitRecord.readAll(files), List.of(failed, trained));
  }

  /** q の学習の途中で停止のフラグが立てば、GRP の候補も q も TRAINED の行も残さない。次の再学習は同じ試行の番号でやり直す。 */
  @Test(groups = "native")
  public void aStopDuringTheQTrainingLeavesNothing() throws Exception {
    GrpFiles files = refitRun("stop", true);
    AtomicInteger asked = new AtomicInteger();
    BooleanSupplier stop =
        () -> {
          boolean inQ =
              StackWalker.getInstance()
                  .walk(frames -> frames.anyMatch(frame -> frame.getClassName().equals(OutcomeTrainer.class.getName())));
          if (inQ) asked.incrementAndGet();
          return inQ;
        };
    Assert.assertTrue(GrpRefitter.refit(files, refitSetup(), 5, RefitCause.SCHEDULE, stop).isEmpty());
    Assert.assertTrue(asked.get() > 0, "the q training saw the stop signal");
    Assert.assertTrue(RefitRecord.readAll(files).isEmpty());
    Path candidates = files.candidate("x").getParent();
    if (Files.exists(candidates)) {
      try (Stream<Path> left = Files.list(candidates)) {
        Assert.assertEquals(left.toList(), List.of());
      }
    }
    RefitRecord record = GrpRefitter.refit(files, refitSetup(), 5, RefitCause.SCHEDULE, () -> false).orElseThrow();
    Assert.assertEquals(record.id(), "g005-a1");
    Assert.assertTrue(record.hasOutcomes(), record.detail());
  }

  /** q をウォームスタートするのは、使う項目が同じ形の q を持つ自己対局の候補のときだけ。使う q が開けなければ、q を持たない候補にする。 */
  @Test(groups = "native")
  public void qWarmStartsOnlyFromASelfTrainedQOfTheSameShape() throws Exception {
    GrpFiles files = refitRun("warm", true);
    Path network = GrpFixtures.network(files.run().resolve("same"), 1);
    Path same = GrpFixtures.outcomes(files.run().resolve("q-same"), SHAPE, 7, 0.1);
    Path other = GrpFixtures.outcomes(files.run().resolve("q-other"), GrpFixtures.VARIANT, 8);
    Path broken = GrpFixtures.outcomes(files.run().resolve("q-broken"), SHAPE, 9, 0.1);
    // 既に始まった反復から効く行なので、追記の確かめを通さずに書く
    for (PhiEntry entry :
        List.of(
            GrpFixtures.ready(files, "g001-a1", 1, network, same).entry(GrpFixtures.VARIANT, 4, "test"),
            GrpFixtures.ready(files, "g001-a2", 1, network, other).entry(GrpFixtures.VARIANT, 5, "test"),
            GrpFixtures.ready(files, "g001-a3", 1, network, broken).entry(GrpFixtures.VARIANT, 6, "test")))
      PhiHistory.write(files.phiHistory(), PhiHistory.LINES.toJson(entry));
    Files.writeString(broken.resolve(GrpCheckpoint.JSON_FILE), "{}");
    PhiSetup setup = refitSetup("phi.grp.init=WARM");
    RefitRecord warm = GrpRefitter.refit(files, setup, 3, RefitCause.MANUAL, () -> false).orElseThrow();
    Assert.assertEquals(warm.init(), GrpInit.WARM);
    Assert.assertTrue(warm.detail().contains("; q init=WARM "), warm.detail());
    RefitRecord scratch = GrpRefitter.refit(files, setup, 4, RefitCause.MANUAL, () -> false).orElseThrow();
    Assert.assertEquals(scratch.init(), GrpInit.WARM);
    Assert.assertTrue(scratch.detail().contains("; q init=SCRATCH "), "another q shape: " + scratch.detail());
    RefitRecord unreadable = GrpRefitter.refit(files, setup, 5, RefitCause.MANUAL, () -> false).orElseThrow();
    Assert.assertEquals(unreadable.status(), RefitStatus.TRAINED, unreadable.detail());
    Assert.assertFalse(unreadable.hasOutcomes());
    Assert.assertTrue(
        unreadable.detail().contains("; q none: java.io.IOException: " + GrpCheckpoint.JSON_FILE + " in "),
        unreadable.detail());
  }

  /** g000 の q からは、ネットワークと同じく、続けて学習しない。 */
  @Test(groups = "native")
  public void qDoesNotWarmStartFromTheStartEntry() throws Exception {
    GrpFiles files = refitRun("start", true);
    Path network = GrpFixtures.network(files.run().resolve("same"), 1);
    Path same = GrpFixtures.outcomes(files.run().resolve("q-same"), SHAPE, 7, 0.1);
    Files.delete(files.phiHistory());
    PhiHistory.write(
        files.phiHistory(),
        PhiHistory.LINES.toJson(
            HandEndPhi.checkpointEntry(PhiEntry.INITIAL_ID, GrpFixtures.VARIANT, network, "", same, 1, "", "test")));
    RefitRecord record = GrpRefitter.refit(files, refitSetup("phi.grp.init=WARM"), 5, RefitCause.MANUAL, () -> false).orElseThrow();
    Assert.assertEquals(record.init(), GrpInit.SCRATCH);
    Assert.assertTrue(record.detail().contains("; q init=SCRATCH "), record.detail());
  }

  /** 学習した q は一様な q より良いので、使用中の項目が q を持たなくても関門を通り、AUTO で足す行に入る。 */
  @Test(groups = "native")
  public void aLearnedQPassesTheFirstGateAndTravelsWithTheCandidate() throws Exception {
    GrpFiles files = files("learned");
    PhiSetup setup = start(files, bad, null, "phi.grp.switching=AUTO", "phi.grp.threads=1");
    OutcomeRows.Builder builder =
        OutcomeRows.builder(GrpFixtures.VARIANT, new RecencyWindow(5, 2.0, 64), GateCohort.NEXT, FinalRound::covers);
    builder.add(5, GrpFixtures.played(5, 500_000L, 600, 77).counted());
    OutcomeRows.Split rows = builder.build();
    OutcomeNetwork network =
        OutcomeTrainer.train(SHAPE, null, rows.train(), rows.stop(), rule(setup, GrpInit.SCRATCH, 11), 1, () -> false)
            .network();
    Path learned = files.run().resolve("learned-q");
    GrpCheckpoint.write(network, learned);
    var lifecycle = new GrpLifecycle(files, setup);
    recordCounted(files, 1);
    recordCounted(files, 2);
    lifecycle.boundary(1);
    trained(files, 1, good, learned);
    lifecycle.boundary(2);
    RefitRecord gated = latest(files, "g001-a1");
    Assert.assertEquals(gated.status(), RefitStatus.READY, gated.detail());
    Assert.assertEquals(gated.outcomeGate().kind(), OutcomeGate.Kind.FIRST);
    Assert.assertTrue(gated.outcomeGate().passed(), gated.outcomeGate().toString());
    Path own = files.candidateOutcomes("g001-a1").toAbsolutePath().normalize();
    Assert.assertEquals(gated.outcomes(), own.toString());
    Assert.assertTrue(gated.detail().contains("; q FIRST gate at c2: mean -"), gated.detail());
    PhiEntry switched = PhiHistory.read(files).getLast();
    Assert.assertEquals(switched.id(), "g001-a1");
    Assert.assertEquals(switched.outcomes(), own.toString());
  }

  // テスト用のヘルパー

  private GrpFiles files(String name) throws IOException {
    Path directory = run.resolve(name);
    Files.createDirectories(directory);
    return new GrpFiles(directory);
  }

  /** チェックポイントと q（無ければ null）から始める自前の GRP の run を作る。g000 を書き、state.json の次のコホートを 3 にする。 */
  private static PhiSetup start(GrpFiles files, Path checkpoint, Path q, String... more) throws IOException {
    String[] assignments = q == null ? more : GrpFixtures.concat(new String[] {"phi.start.outcomes=" + q}, more);
    PhiSetup setup = GrpFixtures.grpStart(checkpoint, assignments);
    GrpFixtures.writeState(files.run(), 3);
    PhiHistory.initialize(files, setup);
    return setup;
  }

  /**
   * 使用中の q（無ければ null）を持つ run で、M(1) の候補（ネットワークと q）を区切り 2 の関門に通し、その候補の最後の行を返す。
   */
  private static RefitRecord gated(GrpFiles files, Path checkpoint, Path q, Path usedQ) throws IOException {
    PhiSetup setup = start(files, usedNetwork(files), usedQ, "phi.grp.switching=AUTO");
    var lifecycle = new GrpLifecycle(files, setup);
    recordCounted(files, 1);
    recordCounted(files, 2);
    lifecycle.boundary(1);
    trained(files, 1, checkpoint, q);
    lifecycle.boundary(2);
    return latest(files, "g001-a1");
  }

  /** 使用中の、悪いネットワーク。どの run でも重みは同じ。 */
  private static Path usedNetwork(GrpFiles files) throws IOException {
    return GrpFixtures.network(files.run().resolve("used"), 2, 1.0);
  }

  private static PhiEntry used(GrpFiles files, int t) throws IOException {
    return PhiHistory.effectiveAt(PhiHistory.read(files), t).orElseThrow();
  }

  /** その反復に効く項目で集めたものとして、コホート t の記録と数のファイルを書く。 */
  private static void recordCounted(GrpFiles files, int t) throws IOException {
    GrpFixtures.recordCounted(files, t, used(files, t), GrpFixtures.played(t, 10_000L * t, GAMES, t));
  }

  /** その時点の TRAINED の候補を書く。ネットワークと q を候補のディレクトリに写し、q が null なら q を持たない候補にする。 */
  private static RefitRecord trained(GrpFiles files, int slot, Path checkpoint, Path q) throws IOException {
    String id = GrpRefitter.attemptId(slot, 1);
    Path candidate = copy(checkpoint, files.candidate(id));
    Path outcomes = q == null ? null : copy(q, files.candidateOutcomes(id));
    var record =
        new RefitRecord(
            id, slot, RefitStatus.TRAINED, RefitCause.SCHEDULE, new RecencyWindow(slot, 2, 64), GrpInit.SCRATCH,
            PhiEntry.INITIAL_ID, candidate.toString(), outcomes == null ? "" : outcomes.toString(), "", null, null, "",
            1);
    record.append(files);
    return record;
  }

  private static Path copy(Path from, Path to) throws IOException {
    Files.createDirectories(to);
    for (String name : List.of(GrpCheckpoint.JSON_FILE, GrpCheckpoint.WEIGHTS_FILE))
      Files.copy(from.resolve(name), to.resolve(name));
    return to.toAbsolutePath().normalize();
  }

  private static RefitRecord latest(GrpFiles files, String id) throws IOException {
    return RefitRecord.latest(RefitRecord.readAll(files), id).orElseThrow();
  }

  /** 理由の文のうち、q の部分より前（GRP の学習か関門について書いた部分）。 */
  private static String head(String detail) {
    int q = detail.indexOf("; q ");
    return q < 0 ? detail : detail.substring(0, q);
  }

  private static RefitRecord record(String id, String outcomes, String sha, OutcomeGate.Report report) {
    return new RefitRecord(
        id, 1, RefitStatus.TRAINED, RefitCause.MANUAL, new RecencyWindow(1, 2, 64), GrpInit.SCRATCH, PhiEntry.INITIAL_ID,
        "/grp/" + id, outcomes, sha, null, report, "", 1);
  }

  /**
   * 点数の順位から始める再学習の run を作る。コホート 1..5 を確定して半荘の記録を書き、{@code counted} なら数のファイルも書く。
   * state.json の次のコホートは 6。
   */
  private GrpFiles refitRun(String name, boolean counted) throws IOException {
    GrpFiles files = files(name);
    GrpFixtures.writeState(files.run(), 6);
    PhiEntry initial = GrpFixtures.scoreEntry(PhiEntry.INITIAL_ID, 1);
    PhiHistory.write(files.phiHistory(), PhiHistory.LINES.toJson(initial));
    for (int t = 1; t <= 5; t++) {
      GrpFixtures.Played played = GrpFixtures.played(t, 1000L * t, REFIT_GAMES, t);
      if (counted) GrpFixtures.recordCounted(files, t, initial, played);
      else GrpFixtures.record(files, t, initial, played.games());
    }
    return files;
  }

  /** 点数の順位から始める自前の GRP の設定。信頼する半荘は 10、スレッドは 1 で、引数の上書きを後に重ねる。 */
  private static PhiSetup refitSetup(String... more) {
    return GrpFixtures.setup(
        GrpFixtures.concat(
            GrpFixtures.concat(GrpFixtures.scoreStartAssignments(), "phi.grp.trustGames=10", "phi.grp.threads=1"),
            more));
  }

  /** 再学習と同じ学習のしかた（重み減衰なし）。 */
  private static GrpTrainingRule rule(PhiSetup setup, GrpInit init, long seed) {
    GrpTraining training = setup.selfGrp().grp().training();
    boolean warm = init == GrpInit.WARM;
    return new GrpTrainingRule(
        init,
        warm ? training.warmRate() : training.scratchRate(),
        0,
        warm ? training.warmEpochs() : training.scratchEpochs(),
        training.batchRows(),
        training.clipNorm(),
        training.halvingPatience(),
        training.stopPatience(),
        seed);
  }
}
