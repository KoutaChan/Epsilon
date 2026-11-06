package com.epsilon.prism.grp;

import com.epsilon.prism.run.PrismConfig;
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
 * grpSwitch = AUTO の切り替え（反復 t の始め）: まだ入っていない READY の候補のうち、学習の始めが phi.jsonl の最後の項目の一番新しい候補を
 * effectiveFrom = t で入れ、ほかの候補（より新しい候補に負けたものと、最後の項目が変わる前に学んだもの）は refits.jsonl に STALE を書く。
 * 最後の項目が t より後に効くなら何も書かずに候補を残す。PROPOSE と空の phi.jsonl は何もしない。
 */
public class AutoSwitchTest {
  private static final PhiEntry G000 =
      new PhiEntry("g000", PhiEntryKind.FIXED_CHECKPOINT, "/grp/a", "sha", "", "", 1, "", "g000");

  private Path run;
  private GrpFiles files;

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("auto-switch");
    files = new GrpFiles(run);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(run);
  }

  /** 予定表の学習で grpSwitch を替えた設定。 */
  private static PrismConfig config(String grpSwitch) {
    return GrpFixtures.config(
        "phiSource", "SELF_GRP", "grpCheckpoint", "/grp", "grpRefit", "SCHEDULE", "grpSwitch", grpSwitch);
  }

  private static RefitRecord ready(String id, int holdout, String parent) {
    return new RefitRecord(
        id,
        RefitStatus.READY,
        RefitCause.SCHEDULE,
        new RefitWindow(holdout - 8, holdout - 1, holdout),
        GrpInit.WARM,
        parent,
        "/run/grp/candidates/" + id,
        "sha-" + id,
        null,
        "",
        30);
  }

  /** phi.jsonl に行を書く。 */
  private void history(PhiEntry... entries) throws IOException {
    for (PhiEntry entry : entries) PhiHistory.write(files.phiHistory(), PhiHistory.LINES.toJson(entry));
  }

  /** refits.jsonl に行を書く。 */
  private void refits(RefitRecord... records) throws IOException {
    for (RefitRecord record : records) record.append(files);
  }

  private List<String> refitIds() throws IOException {
    return RefitRecord.readAll(files).stream().map(RefitRecord::id).toList();
  }

  @Test
  public void pendingCandidatesAreReadyAndNotYetInTheHistory() {
    RefitRecord late = ready("g120-a1", 120, "g000");
    RefitRecord early = ready("g112-a1", 112, "g000");
    RefitRecord retried = ready("g112-a2", 112, "g000");
    RefitRecord stale = ready("g104-a1", 104, "g000").withStatus(RefitStatus.STALE, "superseded by g112-a1");
    RefitRecord switched = ready("g096-a1", 96, "g000");
    List<RefitRecord> records = List.of(late, ready("g104-a1", 104, "g000"), early, stale, switched, retried);
    // 取り置きの順（同じ取り置きは書いた順）。同じ id は最後の行で決め、phi.jsonl に入った候補は除く
    Assert.assertEquals(
        AutoSwitch.pending(records, List.of(G000, switched.entry(98, "AUTO: g096-a1 at iteration 98"))),
        List.of(early, retried, late));
  }

  @Test
  public void theNewestCandidateWinsAndTheOlderOnesBecomeStale() throws IOException {
    history(G000);
    RefitRecord older = ready("g010-a1", 10, "g000");
    RefitRecord first = ready("g012-a1", 12, "g000");
    RefitRecord newest = ready("g012-a2", 12, "g000");
    refits(older, first, newest);
    PhiEntry entry = AutoSwitch.apply(files, config("AUTO"), 14).orElseThrow();
    Assert.assertEquals(entry, newest.entry(14, "AUTO: g012-a2 at iteration 14"));
    Assert.assertEquals(PhiHistory.read(files), List.of(G000, entry));
    List<RefitRecord> records = RefitRecord.readAll(files);
    Assert.assertEquals(
        records.subList(3, records.size()),
        List.of(
            older.withStatus(RefitStatus.STALE, "superseded by g012-a2"),
            first.withStatus(RefitStatus.STALE, "superseded by g012-a2")));
    // 入った候補も STALE の候補ももう待っていない
    Assert.assertTrue(AutoSwitch.apply(files, config("AUTO"), 15).isEmpty());
    Assert.assertEquals(RefitRecord.readAll(files).size(), 5);
    Assert.assertEquals(PhiHistory.read(files).size(), 2);
  }

  @Test
  public void aCandidateTrainedAgainstAnEarlierEntryBecomesStale() throws IOException {
    RefitRecord g010 = ready("g010-a1", 10, "g000");
    history(G000, g010.entry(12, "AUTO: g010-a1 at iteration 12"));
    RefitRecord before = ready("g014-a1", 14, "g000");
    RefitRecord after = ready("g016-a1", 16, "g010-a1");
    refits(g010, before, after);
    PhiEntry entry = AutoSwitch.apply(files, config("AUTO"), 18).orElseThrow();
    Assert.assertEquals(entry, after.entry(18, "AUTO: g016-a1 at iteration 18"));
    Assert.assertEquals(
        RefitRecord.readAll(files).getLast(),
        before.withStatus(RefitStatus.STALE, "trained against g000, but the last entry is g010-a1"));
    Assert.assertEquals(refitIds(), List.of("g010-a1", "g014-a1", "g016-a1", "g014-a1"));
  }

  @Test
  public void anEntryEffectiveAfterTheIterationLeavesTheCandidatesPending() throws IOException {
    RefitRecord g010 = ready("g010-a1", 10, "g000");
    // 巻き戻し: 反復 20 から g000 に戻す行（id は g000 のまま）
    history(G000, g010.entry(12, "AUTO: g010-a1 at iteration 12"), G000.from(20, "rollback: g000 beat g010-a1"));
    RefitRecord candidate = ready("g016-a1", 16, "g010-a1");
    refits(g010, candidate);
    // 反復 19 の始めには何も書かず、候補は待ったまま
    Assert.assertTrue(AutoSwitch.apply(files, config("AUTO"), 19).isEmpty());
    Assert.assertEquals(refitIds(), List.of("g010-a1", "g016-a1"));
    Assert.assertEquals(PhiHistory.read(files).size(), 3);
    Assert.assertEquals(
        AutoSwitch.pending(RefitRecord.readAll(files), PhiHistory.read(files)).stream().map(RefitRecord::id).toList(),
        List.of("g016-a1"));
    // 反復 20 の始め: 候補は巻き戻した項目を相手に学んでいないので入れない
    Assert.assertTrue(AutoSwitch.apply(files, config("AUTO"), 20).isEmpty());
    Assert.assertEquals(
        RefitRecord.readAll(files).getLast(),
        candidate.withStatus(RefitStatus.STALE, "trained against g010-a1, but the last entry is g000"));
    Assert.assertEquals(PhiHistory.read(files).size(), 3);
  }

  @Test
  public void proposeNeverSwitches() throws IOException {
    history(G000);
    refits(ready("g010-a1", 10, "g000"), ready("g012-a1", 12, "g000"));
    Assert.assertTrue(AutoSwitch.apply(files, config("PROPOSE"), 14).isEmpty());
    Assert.assertEquals(PhiHistory.read(files), List.of(G000));
    Assert.assertEquals(refitIds(), List.of("g010-a1", "g012-a1"), "no STALE line either");
  }

  @Test
  public void anEmptyHistorySwitchesNothing() throws IOException {
    refits(ready("g012-a1", 12, "g000"));
    Assert.assertTrue(AutoSwitch.apply(files, config("AUTO"), 14).isEmpty());
    Assert.assertFalse(Files.exists(files.phiHistory()));
    Assert.assertEquals(refitIds(), List.of("g012-a1"));
  }
}
