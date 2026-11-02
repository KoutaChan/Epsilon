package com.epsilon.prism.grp;

import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Optional;
import org.testng.Assert;
import org.testng.annotations.Test;

public class AutoSwitchTest {
  private static final PhiEntry G000 =
      new PhiEntry("g000", PhiEntryKind.FIXED_CHECKPOINT, "/grp/a", "sha", "", "", 1, "", "g000");

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
        "jar",
        null,
        "",
        30);
  }

  @Test
  public void pendingCandidatesAreReadyAndNotYetInTheHistory() {
    RefitRecord late = ready("g120-a1", 120, "g000");
    RefitRecord early = ready("g112-a1", 112, "g000");
    RefitRecord skipped = AutoSwitch.skipped(ready("g104-a1", 104, "g000"), "late");
    RefitRecord switched = ready("g96-a1", 96, "g000");
    PhiEntry inHistory =
        new PhiEntry("g96-a1", PhiEntryKind.SELF_CHECKPOINT, "/c", "s", "", "", 98, "g000", "AUTO");
    List<RefitRecord> records =
        List.of(late, ready("g104-a1", 104, "g000"), early, skipped, switched);
    Assert.assertEquals(
        AutoSwitch.pending(records, List.of(G000, inHistory)), List.of(early, late));
  }

  @Test
  public void aCandidateSwitchesAtHoldoutPlusLag() {
    AutoSwitch.Step step = AutoSwitch.step(ready("g112-a1", 112, "g000"), List.of(G000), 2);
    Assert.assertNull(step.skip());
    Assert.assertEquals(step.effectiveFrom(), 114);
  }

  @Test
  public void staleCandidatesAreSkipped() {
    PhiEntry newer =
        new PhiEntry("g104-a1", PhiEntryKind.SELF_CHECKPOINT, "/c", "s", "", "", 106, "g000", "AUTO");
    // 学習の始めにした項目がもう最後の項目でない
    AutoSwitch.Step stale = AutoSwitch.step(ready("g112-a1", 112, "g000"), List.of(G000, newer), 2);
    Assert.assertNotNull(stale.skip());
    // 最後の項目より前に効くはずだった
    PhiEntry future =
        new PhiEntry("g104-a1", PhiEntryKind.SELF_CHECKPOINT, "/c", "s", "", "", 130, "g000", "AUTO");
    AutoSwitch.Step early = AutoSwitch.step(ready("g112-a1", 112, "g104-a1"), List.of(G000, future), 2);
    Assert.assertNotNull(early.skip());
    Assert.assertNotNull(AutoSwitch.step(ready("g112-a1", 112, "g000"), List.of(), 2).skip());
  }

  @Test
  public void skippedRowsDropThePathAndKeepTheAttempt() {
    RefitRecord candidate = ready("g112-a1", 112, "g000");
    RefitRecord skipped = AutoSwitch.skipped(candidate, "iteration 114 has started");
    Assert.assertEquals(skipped.status(), RefitStatus.SKIPPED_NOT_READY);
    Assert.assertEquals(skipped.id(), candidate.id());
    Assert.assertEquals(skipped.window(), candidate.window());
    Assert.assertEquals(skipped.path(), "");
    Assert.assertEquals(skipped.sha256(), "");
    Assert.assertEquals(skipped.detail(), "iteration 114 has started");
    Assert.assertFalse(skipped.status().failure());
  }

  @Test
  public void proposeNeverSwitches() throws IOException {
    GrpFiles files = new GrpFiles(Files.createTempDirectory("auto-switch"));
    var propose =
        RefitSchedulerTest.config("phiSource", "SELF_GRP", "grpCheckpoint", "/grp", "grpRefit", "SCHEDULE");
    Assert.assertEquals(AutoSwitch.apply(files, propose), Optional.empty());
  }
}
