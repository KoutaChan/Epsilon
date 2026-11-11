package com.epsilon.prism.grp;

import com.epsilon.agent.run.StepReport;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.train.PhiStamp;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.run.PhiReport;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 反復の Φ の値（{@link PhiValue}）と、q を持つ項目（{@link PhiEntry}）のテスト。LATE の行が効くのは q を持つ項目だけで、q の無い項目は、
 * 求めたモードを記録したうえで GRP のモードで値を求める。値は項目の id と中身・使ったモード・打ち切りの値で見分け、固定の GRP はファイルの
 * sha でも見分ける。巻き戻しの行と g000 の照合では q も持ち越して比べ、q のファイルが変わった項目には戻さない。Φ の段の報告は、どのモードの
 * 値も書いたとおりに読み戻せる。数のファイルの印（{@link PhiValue#stamp}）は、使ったモードと打ち切りの組。
 */
public class PhiValueTest {

  private static final PhiEntry PLAIN = network("g003-a1", "/grp/a", "", "");
  private static final PhiEntry WITH_Q = network("g003-a1", "/grp/a", "/grp/q", "");

  /** run が書いた候補の項目（チェックポイントの sha を持たない）。 */
  private static PhiEntry network(String id, String path, String outcomes, String outcomesSha) {
    return new PhiEntry(
        id, PhiEntryKind.NETWORK, GrpFixtures.VARIANT, path, "", "", "", outcomes, outcomesSha, 4, PhiEntry.INITIAL_ID,
        "test");
  }

  /** run の外から来た固定の GRP の項目。 */
  private static PhiEntry fixed(String sha) {
    return new PhiEntry(
        PhiEntry.FIXED_ID, PhiEntryKind.NETWORK, GrpFixtures.VARIANT, "/grp/f", sha, "", "", "", "", 1, "", "test");
  }

  @Test
  public void lateAppliesOnlyToEntriesWithAnOutcomeNetwork() {
    ModeLine late = ModeLine.late(1e-5, 4, "switch: late");
    PhiValue enumerated = PhiValue.of(WITH_Q, late);
    Assert.assertEquals(enumerated.requested(), PhiMode.LATE);
    Assert.assertEquals(enumerated.mode(), PhiMode.LATE);
    Assert.assertEquals((double) enumerated.cutoff(), 1e-5);
    Assert.assertSame(enumerated.entry(), WITH_Q);

    PhiValue kept = PhiValue.of(PLAIN, late);
    Assert.assertEquals(kept.requested(), PhiMode.LATE);
    Assert.assertEquals(kept.mode(), PhiMode.GRP);
    Assert.assertNull(kept.cutoff());
    Assert.assertTrue(kept.sameValue(PhiValue.grp(PLAIN)), "the requested mode is not the value");

    PhiValue grp = PhiValue.of(WITH_Q, ModeLine.grp(4, "switch: grp"));
    Assert.assertEquals(grp.requested(), PhiMode.GRP);
    Assert.assertEquals(grp.mode(), PhiMode.GRP);
    Assert.assertNull(grp.cutoff());
    Assert.assertEquals(grp, PhiValue.grp(WITH_Q));
    Assert.assertFalse(enumerated.sameValue(grp));

    // 点数だけの項目も、GRP のモードのまま
    PhiValue score = PhiValue.of(PhiEntry.scoreOnly(PhiEntryKind.SCORE_RANK, GrpFixtures.VARIANT), late);
    Assert.assertEquals(score.mode(), PhiMode.GRP);
    Assert.assertEquals(score.requested(), PhiMode.LATE);
  }

  @Test
  public void theValueFollowsWhatDecidesIt() {
    ModeLine late = ModeLine.late(1e-5, 4, "x");
    PhiValue value = PhiValue.of(WITH_Q, late);
    List<PhiValue> others =
        List.of(
            PhiValue.of(WITH_Q, ModeLine.late(1e-6, 4, "x")),
            PhiValue.grp(WITH_Q),
            PhiValue.of(network("g004-a1", "/grp/a", "/grp/q", ""), late),
            PhiValue.of(network("g003-a1", "/grp/b", "/grp/q", ""), late),
            PhiValue.of(network("g003-a1", "/grp/a", "/grp/r", ""), late),
            PhiValue.of(network("g003-a1", "/grp/a", "/grp/q", "qq"), late),
            PhiValue.of(
                new PhiEntry(
                    "g003-a1", PhiEntryKind.NETWORK, Variant.TENHOU_3P, "/grp/a", "", "", "", "/grp/q", "", 4, "", "x"),
                late),
            PhiValue.grp(PhiEntry.scoreOnly(PhiEntryKind.SCORE_RANK, GrpFixtures.VARIANT)));
    for (PhiValue other : others) {
      Assert.assertFalse(value.sameValue(other), "cutoff, mode, id, checkpoint, q and rule each change the value: " + other);
      Assert.assertFalse(other.sameValue(value), other.toString());
    }
    Assert.assertFalse(
        PhiValue.grp(PhiEntry.scoreOnly(PhiEntryKind.SCORE_RANK, GrpFixtures.VARIANT))
            .sameValue(PhiValue.grp(PhiEntry.scoreOnly(PhiEntryKind.SCORE_SMOOTH, GrpFixtures.VARIANT))));

    // 効き始め、親、理由は、値を変えない
    PhiEntry back =
        new PhiEntry(
            "g003-a1", PhiEntryKind.NETWORK, GrpFixtures.VARIANT, "/grp/a", "", "", "", "/grp/q", "", 9, "g001-a1",
            "rollback: test");
    Assert.assertTrue(PhiValue.of(back, late).sameValue(value));
    Assert.assertTrue(PhiValue.grp(back.from(12, "again")).sameValue(PhiValue.grp(WITH_Q)));

    // 固定の GRP は id がいつも同じなので、ファイルを替えたことは sha で分かる。表の sha も値を変える
    Assert.assertTrue(PhiValue.grp(fixed("aa")).sameValue(PhiValue.grp(fixed("aa"))));
    Assert.assertFalse(PhiValue.grp(fixed("aa")).sameValue(PhiValue.grp(fixed("ab"))));
    PhiEntry overlaid =
        new PhiEntry(
            PhiEntry.FIXED_ID, PhiEntryKind.NETWORK, GrpFixtures.VARIANT, "/grp/f", "aa", "/grp/t.json", "tt", "", "",
            1, "", "test");
    Assert.assertFalse(PhiValue.grp(fixed("aa")).sameValue(PhiValue.grp(overlaid)));
  }

  /** 数のファイルの印は、使ったモードと打ち切り（GRP のモードでは NaN）の組。 */
  @Test
  public void theStampCarriesTheModeAndTheCutoff() {
    PhiValue score = PhiValue.grp(PhiEntry.scoreOnly(PhiEntryKind.SCORE_RANK, GrpFixtures.VARIANT));
    PhiValue grp = PhiValue.grp(PLAIN);
    PhiValue late = PhiValue.of(WITH_Q, ModeLine.late(1e-5, 4, "x"));
    Assert.assertEquals(score.stamp(), new PhiStamp("GRP", Double.NaN));
    Assert.assertEquals(grp.stamp(), new PhiStamp("GRP", Double.NaN));
    Assert.assertEquals(late.stamp(), new PhiStamp("LATE", 1e-5));
    // q の無い項目に LATE を求めた値でも、印は GRP のモードになる
    Assert.assertEquals(PhiValue.of(PLAIN, ModeLine.late(1e-5, 4, "x")).stamp(), grp.stamp());
  }

  @Test
  public void inconsistentValuesAreRefused() {
    Assert.assertEquals(
        new PhiValue(WITH_Q, PhiMode.LATE, PhiMode.LATE, 1e-5), PhiValue.of(WITH_Q, ModeLine.late(1e-5, 1, "x")));
    Assert.assertThrows(IllegalArgumentException.class, () -> new PhiValue(PLAIN, PhiMode.LATE, PhiMode.LATE, 1e-5));
    Assert.assertThrows(IllegalArgumentException.class, () -> new PhiValue(WITH_Q, PhiMode.GRP, PhiMode.LATE, 1e-5));
    Assert.assertThrows(IllegalArgumentException.class, () -> new PhiValue(WITH_Q, PhiMode.LATE, PhiMode.LATE, null));
    Assert.assertThrows(IllegalArgumentException.class, () -> new PhiValue(WITH_Q, PhiMode.GRP, PhiMode.GRP, 1e-5));
  }

  @Test
  public void entriesCarryTheirOutcomeNetwork() {
    Assert.assertTrue(WITH_Q.hasOutcomes());
    Assert.assertFalse(PLAIN.hasOutcomes());
    PhiEntry back = WITH_Q.from(9, PhiEntry.ROLLBACK + " test");
    Assert.assertEquals(back.outcomes(), "/grp/q");
    Assert.assertEquals(back.outcomesSha256(), "");
    Assert.assertEquals(back.effectiveFrom(), 9);
    Assert.assertTrue(back.sameValue(WITH_Q));
    Assert.assertFalse(WITH_Q.sameValue(PLAIN));
    // run が書いた q は sha を持たないのでパスで比べ、run の外から引き継いだ q は sha で比べる
    Assert.assertFalse(WITH_Q.sameValue(network("g003-a1", "/grp/a", "/other/q", "")));
    PhiEntry inherited = network("g003-a1", "/grp/a", "/grp/q", "qq");
    Assert.assertFalse(WITH_Q.sameValue(inherited));
    Assert.assertFalse(inherited.sameValue(network("g003-a1", "/grp/a", "/grp/q", "qr")));
    Assert.assertTrue(
        inherited.sameValue(network("g003-a1", "/grp/a", "/other/q", "qq")), "the path of an outside q is not the value");
    // q が null なら「持たない」とみなす
    Assert.assertEquals(network("g003-a1", "/grp/a", null, null), PLAIN);

    Assert.assertThrows(IllegalArgumentException.class, () -> network("g003-a1", "/grp/a", "", "qq"));
    // run の外から来た項目は、チェックポイントの sha と、q を持てば q の sha が要る
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> new PhiEntry(
            PhiEntry.INITIAL_ID, PhiEntryKind.NETWORK, GrpFixtures.VARIANT, "/grp/a", "", "", "", "", "", 1, "", "x"));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> new PhiEntry(
            PhiEntry.INITIAL_ID, PhiEntryKind.NETWORK, GrpFixtures.VARIANT, "/grp/a", "aa", "", "", "/grp/q", "", 1, "",
            "x"));
    Assert.assertTrue(
        new PhiEntry(
                PhiEntry.INITIAL_ID, PhiEntryKind.NETWORK, GrpFixtures.VARIANT, "/grp/a", "aa", "", "", "/grp/q", "qq",
                1, "", "x")
            .hasOutcomes());
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> new PhiEntry(
            "g003-a1", PhiEntryKind.NETWORK, GrpFixtures.VARIANT, "/grp/a", "", "/grp/t.json", "tt", "/grp/q", "", 4,
            "", "x"));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> new PhiEntry(
            "g000", PhiEntryKind.SCORE_RANK, GrpFixtures.VARIANT, "", "", "", "", "/grp/q", "qq", 1, "", "x"));
    for (PhiMode mode : PhiMode.values())
      Assert.assertThrows(IllegalArgumentException.class, () -> network(mode.alias(), "/grp/a", "", ""));
  }

  @Test
  public void theInitialEntryAndItsCheckIncludeTheOutcomeNetwork() throws IOException {
    Path run = Files.createTempDirectory("prism-value");
    try {
      var files = new GrpFiles(run);
      Path leaf = GrpFixtures.network(run.resolve("leaf"), 1);
      Path q = GrpFixtures.outcomes(run.resolve("q"), GrpFixtures.VARIANT, 2);
      Path other = GrpFixtures.outcomes(run.resolve("q-other"), GrpFixtures.VARIANT, 3);
      PhiEntry initial = PhiHistory.initialize(files, GrpFixtures.grpStart(leaf, "phi.start.outcomes=" + q));
      Assert.assertEquals(initial.outcomes(), q.toString());
      Assert.assertEquals(initial.outcomesSha256(), GrpCheckpoint.sha256(q));
      Assert.assertEquals(PhiHistory.read(files), List.of(initial));
      // 同じ q を別の書き方のパスで指しても、g000 は同じ
      Path roundabout = run.resolve("leaf").resolve("..").resolve("q");
      Assert.assertEquals(PhiHistory.initialize(files, GrpFixtures.grpStart(leaf, "phi.start.outcomes=" + roundabout)), initial);
      // q を外した設定や、別の q を指す設定は、g000 と食い違う
      Assert.assertThrows(IOException.class, () -> PhiHistory.initialize(files, GrpFixtures.grpStart(leaf)));
      Assert.assertThrows(
          IOException.class, () -> PhiHistory.initialize(files, GrpFixtures.grpStart(leaf, "phi.start.outcomes=" + other)));
      Assert.assertEquals(PhiHistory.read(files).size(), 1);
      // 同じ id で q だけが違う行は足せない
      GrpFixtures.writeState(run, 1);
      Assert.assertThrows(
          IllegalArgumentException.class,
          () -> PhiHistory.append(
              files, HandEndPhi.checkpointEntry(PhiEntry.INITIAL_ID, GrpFixtures.VARIANT, leaf, "", other, 2, "", "x")));
      PhiHistory.append(files, initial.from(2, PhiEntry.ROLLBACK + " test"));
      Assert.assertEquals(PhiHistory.read(files).getLast().outcomesSha256(), initial.outcomesSha256());
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  @Test
  public void aRollbackCarriesAndChecksTheOutcomeNetwork() throws IOException {
    Path run = Files.createTempDirectory("prism-value-rollback");
    try {
      var files = new GrpFiles(run);
      GrpFixtures.writeState(run, 4);
      Path leaf = GrpFixtures.network(run.resolve("leaf"), 1);
      Path q = GrpFixtures.outcomes(run.resolve("q"), GrpFixtures.VARIANT, 2);
      PhiEntry initial = PhiHistory.initialize(files, GrpFixtures.grpStart(leaf, "phi.start.outcomes=" + q));
      GrpFixtures.ready(files, "g003-a1", 3, GrpFixtures.network(run.resolve("b"), 3));
      PhiControl.switchTo(run, "g003-a1", 4, "test").orElseThrow();
      PhiEntry back = PhiControl.switchTo(run, PhiEntry.INITIAL_ID, 6, "test").orElseThrow();
      Assert.assertEquals(back.outcomes(), initial.outcomes());
      Assert.assertEquals(back.outcomesSha256(), initial.outcomesSha256());
      Assert.assertTrue(back.rollback());
      Files.writeString(
          q.resolve(GrpCheckpoint.WEIGHTS_FILE), "x", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
      IllegalStateException changed =
          Assert.expectThrows(
              IllegalStateException.class, () -> PhiControl.switchTo(run, PhiEntry.INITIAL_ID, 8, "test"));
      Assert.assertTrue(changed.getMessage().startsWith("Outcome network "), changed.getMessage());
      Assert.assertEquals(PhiHistory.read(files).size(), 3);
    } finally {
      Snapshot.deleteTree(run);
    }
  }

  @Test
  public void thePhiReportReadsBackInEitherMode() throws IOException {
    Path work = Files.createTempDirectory("prism-phi-report");
    try {
      for (PhiValue value :
          List.of(
              PhiValue.of(WITH_Q, ModeLine.late(1e-5, 4, "x")),
              PhiValue.of(PLAIN, ModeLine.late(1e-5, 4, "x")),
              PhiValue.grp(PhiEntry.scoreOnly(PhiEntryKind.SCORE_RANK, GrpFixtures.VARIANT)))) {
        Path file = work.resolve(value.mode() + "-" + value.entry().id() + ".json");
        new StepReport(StepReport.Status.RAN, 1.5, 1, "test.jar", new PhiReport(value, 0.25)).write(file);
        PhiReport read = (PhiReport) StepReport.read(file, PhiReport.class).metrics();
        Assert.assertEquals(read, new PhiReport(value, 0.25), Files.readString(file));
      }
    } finally {
      Snapshot.deleteTree(work);
    }
  }
}
