package com.epsilon.prism.grp;

import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.Test;

public class RollbackCheckTest {
  private static final PhiEntry G000 =
      new PhiEntry("g000", PhiEntryKind.FIXED_CHECKPOINT, "/grp/a", "sha0", "1.1", "/calib.json", 1, "", "g000");
  private static final PhiEntry G48 =
      new PhiEntry("g48-a1", PhiEntryKind.SELF_CHECKPOINT, "/c/g48-a1", "sha48", "", "", 50, "g000", "AUTO");

  @Test
  public void theNewEntryIsCheckedOnceItHasTwoCohorts() {
    List<PhiEntry> history = List.of(G000, G48);
    Assert.assertEquals(RollbackCheck.target(history, Set.of(), 50, t -> true), Optional.empty());
    Assert.assertEquals(
        RollbackCheck.target(history, Set.of(), 51, t -> true),
        Optional.of(new RollbackCheck.Target(G48, G000)));
    Assert.assertEquals(
        RollbackCheck.target(history, Set.of(), 60, t -> true),
        Optional.of(new RollbackCheck.Target(G48, G000)));
    Assert.assertEquals(RollbackCheck.target(history, Set.of(), 51, t -> t != 51), Optional.empty());
    Assert.assertEquals(RollbackCheck.target(history, Set.of("g48-a1"), 51, t -> true), Optional.empty());
    Assert.assertEquals(RollbackCheck.target(List.of(G000), Set.of(), 51, t -> true), Optional.empty());
    // まだ効いていない項目は点検しない（今効いている項目が g000 で、前がない）
    Assert.assertEquals(RollbackCheck.target(history, Set.of(), 49, t -> true), Optional.empty());
  }

  /** 境目の点検で拒んだ候補の代わりに同じ項目を効かせ直した行は新しい切り替えでなく、本当に切り替えた行を前の項目と比べる。 */
  @Test
  public void aFallbackLineKeepsTheRealSwitch() {
    PhiEntry rejected = new PhiEntry("g49-a1", PhiEntryKind.SELF_CHECKPOINT, "/c/g49-a1", "sha49", "", "", 51, "g48-a1", "AUTO");
    PhiEntry fallback = G48.from(51, "rejected g49-a1 at iteration 51");
    List<PhiEntry> history = List.of(G000, G48, rejected, fallback);
    Assert.assertEquals(
        RollbackCheck.target(history, Set.of(), 52, t -> true), Optional.of(new RollbackCheck.Target(G48, G000)));
    Assert.assertEquals(RollbackCheck.target(history, Set.of("g48-a1"), 52, t -> true), Optional.empty());
  }

  @Test
  public void rollbackNeedsTwoStandardErrors() {
    Assert.assertTrue(RollbackCheck.rollback(new CellStats.Paired(-3e-5, 1e-5)));
    Assert.assertFalse(RollbackCheck.rollback(new CellStats.Paired(-1.5e-5, 1e-5)));
    Assert.assertFalse(RollbackCheck.rollback(new CellStats.Paired(4e-5, 1e-5)));
  }

  @Test
  public void theRollbackEntryRestoresThePreviousPhiAndIsNotCheckedAgain() {
    PhiEntry back = RollbackCheck.rollbackEntry(G000, G48, 60, new CellStats.Paired(-3e-5, 1e-5));
    Assert.assertEquals(back.id(), G000.id());
    Assert.assertEquals(back.kind(), G000.kind());
    Assert.assertEquals(back.path(), G000.path());
    Assert.assertEquals(back.sha256(), G000.sha256());
    Assert.assertEquals(back.temperatures(), G000.temperatures());
    Assert.assertEquals(back.calibration(), G000.calibration());
    Assert.assertEquals(back.effectiveFrom(), 60);
    Assert.assertTrue(RollbackCheck.isRollback(back));
    Assert.assertFalse(RollbackCheck.isRollback(G48));
    Assert.assertEquals(
        RollbackCheck.target(List.of(G000, G48, back), Set.of("g48-a1"), 70, t -> true), Optional.empty());
  }

  @Test
  public void onlySelfGrpLineagesAreChecked() throws IOException {
    GrpFiles files = new GrpFiles(Files.createTempDirectory("rollback"));
    Assert.assertEquals(RollbackCheck.check(files, RefitSchedulerTest.config()), Optional.empty());
  }
}
