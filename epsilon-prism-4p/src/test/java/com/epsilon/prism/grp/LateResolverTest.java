package com.epsilon.prism.grp;

import com.epsilon.grp.FinalRound;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.Phi;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 反復ごとの Φ の値を、GRP の項目とモードの行から決めるテスト（{@link PhiResolver}）。LATE の反復では、数え上げる局末の値が FinalRound の
 * 値と、ほかの局末の値がネットワークの値と、それぞれビットで同じになる。q の無い項目は、LATE を求めても GRP のモードのままで、Phi を開き直さない。初めて効く候補の
 * q が開けなければ、その候補を拒んで前の項目に戻す。既に効いた項目の q が開けなければ止める。固定の GRP も、q とモードの行があれば LATE になる。
 * 固定の GRP のファイルを替えれば別の値になり、前の値は開き直せない。
 */
public class LateResolverTest {

  private static final double CUTOFF = 1e-4;

  private Path run;
  private GrpFiles files;
  private Path leaf;
  private Path q;
  private final List<GrpState> covered = new ArrayList<>();
  private final List<GrpState> uncovered = new ArrayList<>();

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("prism-late-resolver");
    files = new GrpFiles(run);
    GrpFixtures.writeState(run, 1);
    leaf = GrpFixtures.network(run.resolve("leaf"), 11);
    q = GrpFixtures.outcomes(run.resolve("q"), GrpFixtures.VARIANT, 12);
    covered.clear();
    uncovered.clear();
    for (GameRecord game : GrpFixtures.games(0, 20, 5))
      for (GrpState state : game.handEnds()) (FinalRound.covers(state) ? covered : uncovered).add(state);
    Assert.assertTrue(covered.size() >= 4 && uncovered.size() >= 4, covered.size() + " / " + uncovered.size());
    covered.subList(4, covered.size()).clear();
    uncovered.subList(4, uncovered.size()).clear();
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(run);
  }

  @Test
  public void theModeLinesSwitchTheValueOfTheSameEntry() throws IOException {
    PhiSetup setup = GrpFixtures.grpStart(leaf, "phi.start.outcomes=" + q, "phi.late.from=3", "phi.late.cutoff=1e-4");
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      Assert.assertEquals(PhiHistory.readModes(files), List.of(ModeLine.late(CUTOFF, 3, "phi.late.from = 3")));
      ActivePhi first = resolver.resolve(1, setup.phi());
      Assert.assertEquals(first.value(), PhiValue.grp(first.entry()));
      Assert.assertEquals(first.phi().kind(), Phi.Kind.NETWORK);
      Assert.assertSame(resolver.resolve(2, setup.phi()).phi(), first.phi());

      ActivePhi late = resolver.resolve(3, setup.phi());
      Assert.assertEquals(late.entry(), first.entry());
      Assert.assertEquals(late.value().mode(), PhiMode.LATE);
      Assert.assertEquals((double) late.value().cutoff(), CUTOFF);
      Assert.assertEquals(late.phi().kind(), Phi.Kind.LATE);
      assertLate(late.phi());
      Assert.assertSame(resolver.resolve(4, setup.phi()).phi(), late.phi());

      PhiHistory.appendMode(files, ModeLine.grp(5, "switch: grp"));
      ActivePhi back = resolver.resolve(5, setup.phi());
      Assert.assertEquals(back.value(), first.value());
      assertNetwork(back.phi());
      // 記録した値で開き直しても、ビットで同じ値になる
      ActivePhi reopened = resolver.reopen(late.value());
      Assert.assertEquals(reopened.value(), late.value());
      assertLate(reopened.phi());
    }
    // 再開しても、同じ反復には同じ値が効く
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      Assert.assertEquals(PhiHistory.readModes(files).size(), 2, "the first LATE line is written once");
      ActivePhi fourth = resolver.resolve(4, setup.phi());
      Assert.assertEquals(fourth.value().mode(), PhiMode.LATE);
      assertLate(fourth.phi());
      Assert.assertEquals(resolver.resolve(6, setup.phi()).value().mode(), PhiMode.GRP);
    }
  }

  @Test
  public void anEntryWithoutAnOutcomeNetworkStaysInTheGrpMode() throws IOException {
    PhiSetup setup = GrpFixtures.scoreStart("phi.late.from=1");
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      ActivePhi first = resolver.resolve(1, setup.phi());
      Assert.assertEquals(first.value().requested(), PhiMode.LATE);
      Assert.assertEquals(first.value().mode(), PhiMode.GRP);
      Assert.assertEquals(first.phi().kind(), Phi.Kind.SCORE_RANK);
      PhiHistory.append(files, GrpFixtures.ready(files, "g001-a1", 1, leaf).entry(GrpFixtures.VARIANT, 3, "test"));
      ActivePhi network = resolver.resolve(3, setup.phi());
      Assert.assertEquals(network.entry().id(), "g001-a1");
      Assert.assertEquals(network.value().requested(), PhiMode.LATE);
      Assert.assertEquals(network.value().mode(), PhiMode.GRP);
      Assert.assertNull(network.value().cutoff());
      assertNetwork(network.phi());
      // GRP の行で戻しても値は変わらないので、開き直さない
      PhiHistory.appendMode(files, ModeLine.grp(4, "switch: grp"));
      ActivePhi grp = resolver.resolve(4, setup.phi());
      Assert.assertEquals(grp.value().requested(), PhiMode.GRP);
      Assert.assertTrue(grp.value().sameValue(network.value()));
      Assert.assertSame(grp.phi(), network.phi());
    }
  }

  @Test
  public void aFreshCandidateWhoseOutcomeNetworkCannotBeOpenedIsRejected() throws IOException {
    PhiSetup setup = GrpFixtures.scoreStart("phi.late.from=1");
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      resolver.resolve(1, setup.phi());
      GrpFixtures.ready(files, "g002-a1", 2, leaf);
      PhiHistory.append(files, withOutcomes("g002-a1", 3));
      corrupt(q);
      ActivePhi third = resolver.resolve(3, setup.phi());
      Assert.assertEquals(third.entry().id(), PhiEntry.INITIAL_ID);
      Assert.assertEquals(third.value().mode(), PhiMode.GRP);
      Assert.assertEquals(
          RefitRecord.latest(RefitRecord.readAll(files), "g002-a1").orElseThrow().status(), RefitStatus.REJECTED);
      PhiEntry kept = PhiHistory.read(files).getLast();
      Assert.assertEquals(kept.id(), PhiEntry.INITIAL_ID);
      Assert.assertEquals(kept.effectiveFrom(), 3);
      Assert.assertEquals(kept.reason(), "rejected g002-a1 at iteration 3");
    }
  }

  @Test
  public void anEntryInEffectWhoseOutcomeNetworkCannotBeOpenedStops() throws IOException {
    PhiSetup setup = GrpFixtures.scoreStart("phi.late.from=5");
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      GrpFixtures.ready(files, "g002-a1", 2, leaf);
      PhiHistory.append(files, withOutcomes("g002-a1", 3));
      Assert.assertEquals(resolver.resolve(3, setup.phi()).value().mode(), PhiMode.GRP);
      Assert.assertEquals(resolver.resolve(4, setup.phi()).entry().id(), "g002-a1");
      corrupt(q);
      IOException stopped = Assert.expectThrows(IOException.class, () -> resolver.resolve(5, setup.phi()));
      Assert.assertTrue(stopped.getMessage().contains("g002-a1"), stopped.getMessage());
      Assert.assertEquals(PhiHistory.read(files).size(), 2, "nothing is rejected");
      Assert.assertEquals(
          RefitRecord.latest(RefitRecord.readAll(files), "g002-a1").orElseThrow().status(), RefitStatus.READY);
    }
  }

  @Test
  public void aFixedGrpWithAnOutcomeNetworkEnumeratesFromItsStart() throws IOException {
    PhiSetup setup =
        GrpFixtures.setup(
            "phi.source=FIXED_GRP", "phi.grp=" + leaf, "phi.outcomes=" + q, "phi.late.from=2",
            "phi.late.cutoff=1e-4");
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      ActivePhi first = resolver.resolve(1, setup.phi());
      Assert.assertEquals(first.entry().id(), PhiEntry.FIXED_ID);
      Assert.assertEquals(first.entry().outcomesSha256(), GrpCheckpoint.sha256(q));
      Assert.assertEquals(first.value().mode(), PhiMode.GRP);
      assertNetwork(first.phi());
      ActivePhi late = resolver.resolve(2, setup.phi());
      Assert.assertEquals(late.value().mode(), PhiMode.LATE);
      assertLate(late.phi());
    }
    Assert.assertFalse(Files.exists(files.phiHistory()), "a fixed GRP writes no Φ entries");

    // LATE を使わない固定の GRP は、grp のディレクトリを作らない
    GrpFiles plain = new GrpFiles(run.resolve("plain"));
    PhiSetup without = GrpFixtures.setup("phi.source=FIXED_GRP", "phi.grp=" + leaf);
    try (PhiResolver resolver = PhiResolver.open(without, plain)) {
      Assert.assertEquals(resolver.resolve(1, without.phi()).value().mode(), PhiMode.GRP);
    }
    Assert.assertFalse(Files.exists(plain.directory()));
  }

  /**
   * 固定の GRP のファイルを替えると、次の反復は id が同じでも別の値になり、Phi（と終盤のメモ）を開き直す。Φ の段に書いた前の値は、
   * ファイルの sha が合わないので開き直せない。
   */
  @Test
  public void aReplacedFixedGrpIsANewValueAndTheRecordedOneIsRefused() throws IOException {
    Path fixed = GrpFixtures.network(run.resolve("fixed"), 21);
    PhiSetup setup =
        GrpFixtures.setup(
            "phi.source=FIXED_GRP", "phi.grp=" + fixed, "phi.outcomes=" + q, "phi.late.from=1",
            "phi.late.cutoff=1e-4");
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      ActivePhi first = resolver.resolve(1, setup.phi());
      Assert.assertEquals(first.value().mode(), PhiMode.LATE);
      Assert.assertSame(resolver.resolve(2, setup.phi()).phi(), first.phi());
      Snapshot.deleteTree(fixed);
      GrpFixtures.network(fixed, 22);
      ActivePhi replaced = resolver.resolve(3, setup.phi());
      Assert.assertEquals(replaced.entry().id(), PhiEntry.FIXED_ID);
      Assert.assertNotEquals(replaced.entry().sha256(), first.entry().sha256());
      Assert.assertFalse(replaced.value().sameValue(first.value()));
      Assert.assertNotSame(replaced.phi(), first.phi());
      IOException refused = Assert.expectThrows(IOException.class, () -> resolver.reopen(first.value()));
      Assert.assertTrue(refused.getMessage().startsWith("GRP checkpoint "), refused.getMessage());
      Assert.assertSame(resolver.reopen(replaced.value()).phi(), replaced.phi());
    }
  }

  /** テストの q を持たせた候補の項目。 */
  private PhiEntry withOutcomes(String id, int from) throws IOException {
    return HandEndPhi.checkpointEntry(id, GrpFixtures.VARIANT, leaf, "", q, from, PhiEntry.INITIAL_ID, "test");
  }

  private static void corrupt(Path checkpoint) throws IOException {
    Files.writeString(
        checkpoint.resolve(GrpCheckpoint.WEIGHTS_FILE), "x", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
  }

  /**
   * 数え上げる局末の値が FinalRound の値と、ほかの局末の値がネットワークの値と、それぞれビットで同じこと。数え上げる局末のどこかでは、
   * ネットワークと値が違うこと。
   */
  private void assertLate(Phi phi) throws IOException {
    GrpNetwork network = (GrpNetwork) GrpCheckpoint.read(leaf);
    Phi direct = Phi.late(FinalRound.of(GrpCheckpoint.readOutcomes(q), network, CUTOFF), GrpFixtures.UTILITY, 16);
    boolean differs = false;
    for (GrpState state : covered) {
      long[] actual = PhiResolverTest.bits(PhiResolverTest.values(phi, state));
      Assert.assertEquals(actual, PhiResolverTest.bits(PhiResolverTest.values(direct, state)), state.toString());
      differs |= !Arrays.equals(actual, PhiResolverTest.bits(PhiResolverTest.direct(leaf, state)));
    }
    Assert.assertTrue(differs, "the covered hand ends are enumerated");
    for (GrpState state : uncovered)
      Assert.assertEquals(
          PhiResolverTest.bits(PhiResolverTest.values(phi, state)),
          PhiResolverTest.bits(PhiResolverTest.direct(leaf, state)),
          state.toString());
  }

  /** 全ての局末の値が、ネットワークの値とビットで同じこと。 */
  private void assertNetwork(Phi phi) throws IOException {
    Assert.assertEquals(phi.kind(), Phi.Kind.NETWORK);
    for (List<GrpState> states : List.of(covered, uncovered))
      for (GrpState state : states)
        Assert.assertEquals(
            PhiResolverTest.bits(PhiResolverTest.values(phi, state)),
            PhiResolverTest.bits(PhiResolverTest.direct(leaf, state)),
            state.toString());
  }
}
