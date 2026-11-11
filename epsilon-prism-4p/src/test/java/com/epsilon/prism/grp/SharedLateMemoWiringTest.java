package com.epsilon.prism.grp;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.CrnScheme;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.grp.FinalRound;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.GrpKey;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.Phi;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.prism.collect.Collector;
import com.epsilon.prism.collect.RowSink;
import com.epsilon.prism.collect.Sampling;
import com.epsilon.prism.collect.TestSampling;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.noise.UraSettings;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.CollectReport;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 共有の終盤のメモと、それを使う側のつなぎのテスト。Φ の値（項目、q、打ち切り）が変われば Phi を開き直し、前の値のメモは読まない。
 * 収集の報告にある終盤の数えでは、複数のスレッドで集めても、当たり、外れ、メモを通らなかった数の和が呼び出しの数に合う。この数えは記録に
 * 書いて読み戻しても変わらない。裏の採点と check-config は、終盤のメモを 1 スロットにして開く。
 */
public class SharedLateMemoWiringTest {

  private static final Sampling SAMPLING =
      TestSampling.grouped(0.25, 0.03, 0.03, 0.03, 0.15, 1.0, 6, false, 0.3, 0.02, 0.1, 0.1, 0.5, 2);

  private static final Behaviour FLOOR = new Behaviour(0.05, 0, CrnScheme.TWO_STAGE_FLOOR);

  /** 裏ドラは実際にめくれた値でラベルを作り、局の 5% では期待値も求めて監査する。 */
  private static final UraSettings URA = new UraSettings.Realized(0.05);

  private Path run;
  private GrpFiles files;
  private Path leaf;
  private Path q;

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("prism-shared-late-memo");
    files = new GrpFiles(run);
    GrpFixtures.writeState(run, 1);
    leaf = GrpFixtures.network(run.resolve("leaf"), 11);
    q = GrpFixtures.outcomes(run.resolve("q"), GrpFixtures.VARIANT, 12);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(run);
  }

  /**
   * 打ち切り、q、葉の順に値を変える。どの反復の Phi も前の Phi とは別のもので、数え上げる局末の値は、その値の FinalRound で求めた値と
   * ビットで同じ。前の Phi のメモに同じ局末が残っていても、初めの 1 巡では 1 度も当たらない。値が同じ反復では同じ Phi を返し、メモに当たる。
   */
  @Test
  public void aNewValueNeverReadsTheLastValuesMemo() throws IOException {
    Path q2 = GrpFixtures.outcomes(run.resolve("q2"), GrpFixtures.VARIANT, 13);
    Path leaf2 = GrpFixtures.network(run.resolve("leaf2"), 14);
    List<GrpState> covered = covered();
    PhiSetup setup =
        GrpFixtures.grpStart(
            leaf, "phi.start.outcomes=" + q, "phi.late.from=1", "phi.late.cutoff=1e-4", "phi.late.memo=1024");
    try (PhiResolver resolver = PhiResolver.open(setup, files)) {
      List<Phi> opened = new ArrayList<>();
      List<long[][]> seen = new ArrayList<>();
      Phi first = assertFresh(resolver.resolve(1, setup.phi()), leaf, q, 1e-4, covered, opened, seen);
      ActivePhi same = resolver.resolve(2, setup.phi());
      Assert.assertSame(same.phi(), first);
      Phi.LateStats before = first.stats().late();
      for (GrpState state : covered) PhiResolverTest.values(first, state);
      Assert.assertEquals(first.stats().late().since(before).hits(), covered.size());

      PhiHistory.appendMode(files, ModeLine.late(0.01, 3, "switch: late"));
      assertFresh(resolver.resolve(3, setup.phi()), leaf, q, 0.01, covered, opened, seen);

      GrpFixtures.ready(files, "g003-a1", 3, leaf, q2);
      PhiHistory.append(
          files, HandEndPhi.checkpointEntry("g003-a1", GrpFixtures.VARIANT, leaf, "", q2, 4, PhiEntry.INITIAL_ID, "test"));
      ActivePhi newQ = resolver.resolve(4, setup.phi());
      Assert.assertEquals(newQ.entry().id(), "g003-a1");
      assertFresh(newQ, leaf, q2, 0.01, covered, opened, seen);

      GrpFixtures.ready(files, "g004-a1", 4, leaf2, q2);
      PhiHistory.append(
          files, HandEndPhi.checkpointEntry("g004-a1", GrpFixtures.VARIANT, leaf2, "", q2, 5, PhiEntry.INITIAL_ID, "test"));
      ActivePhi newLeaf = resolver.resolve(5, setup.phi());
      Assert.assertEquals(newLeaf.entry().id(), "g004-a1");
      assertFresh(newLeaf, leaf2, q2, 0.01, covered, opened, seen);
    }
  }

  /**
   * LATE の Phi を使い、4 本の卓のスレッドで 6 半荘を集める。報告にある Φ の数えは、収集の結果と同じ。終盤の数えでは、当たり、外れ、
   * メモを通らなかった数の和が呼び出しの数に合い、どの数も Φ 全体の数えに含まれる。記録の JSON に書いて読み戻しても同じ。GRP のモードの
   * Phi で同じ半荘を集めると、終盤の数えは全て 0。
   */
  @Test(timeOut = 1_800_000)
  public void theCollectReportCarriesTheSharedLateCounts() throws IOException {
    GrpNetwork network = (GrpNetwork) GrpCheckpoint.read(leaf);
    Phi late = Phi.late(FinalRound.of(GrpCheckpoint.readOutcomes(q), network, 1e-4), GrpFixtures.UTILITY, 64);
    CollectReport report = collect(late);
    Phi.Stats phi = report.phi();
    Phi.LateStats counts = phi.late();
    Assert.assertEquals(phi, late.stats());
    Assert.assertTrue(counts.requests() > 0 && counts.misses() > 0 && counts.hits() > 0, counts.toString());
    Assert.assertEquals(counts.hits() + counts.misses() + counts.bypasses(), counts.requests(), counts.toString());
    Assert.assertTrue(counts.requests() <= phi.requests(), phi.toString());
    Assert.assertTrue(counts.hits() <= phi.hits() && counts.misses() <= phi.misses(), phi.toString());
    Assert.assertTrue(phi.hits() + phi.misses() + phi.bypasses() <= phi.requests(), phi.toString());
    Assert.assertTrue(counts.nanos() > 0 && counts.nanos() <= phi.nanos(), phi.toString());
    Assert.assertTrue(counts.leafNetworks() > 0, counts.toString());

    String json = RecordFile.json(report);
    Assert.assertTrue(json.contains("\"late\""), json);
    CollectReport read = RecordFile.read(CollectReport.class, json, "test").value();
    Assert.assertEquals(read.phi(), phi);

    CollectReport grp = collect(Phi.of(network, GrpFixtures.UTILITY));
    Assert.assertEquals(grp.phi().late(), new Phi.LateStats(0, 0, 0, 0, 0, 0, 0, 0, 0));
    Assert.assertTrue(grp.phi().requests() > 0, grp.phi().toString());
  }

  /**
   * 裏の採点と check-config が開く Phi（{@link HandEndPhi#openSingleSlot}）のテスト。どちらのモードでも、値は系譜が既定のスロットの数で
   * 開いた Phi とビットで同じ。LATE のモードの終盤のメモは、最後に数え上げた卓だけを覚える。2 つの卓を A、B、A、A の順に呼ぶと、外れが
   * 3 回で、当たりは続けて呼んだ最後の 1 回だけになる。
   */
  @Test
  public void evaluationOpensTheLateMemoWithASingleSlot() throws IOException {
    PhiEntry entry = HandEndPhi.checkpointEntry(PhiEntry.INITIAL_ID, GrpFixtures.VARIANT, leaf, "", q, 1, "", "test");
    PhiValue late = PhiValue.of(entry, ModeLine.late(1e-4, 1, "test"));
    List<GrpState> covered = covered();
    for (PhiValue value : List.of(late, PhiValue.grp(entry))) {
      Phi single = HandEndPhi.openSingleSlot(value, GrpFixtures.UTILITY);
      Phi lineage = HandEndPhi.open(value, GrpFixtures.UTILITY, 1 << 20);
      for (GrpState state : covered)
        Assert.assertEquals(
            PhiResolverTest.bits(PhiResolverTest.values(single, state)),
            PhiResolverTest.bits(PhiResolverTest.values(lineage, state)),
            value.mode() + " " + state);
    }
    Phi single = HandEndPhi.openSingleSlot(late, GrpFixtures.UTILITY);
    Assert.assertEquals(single.kind(), Phi.Kind.LATE);
    GrpState a = covered.get(0);
    GrpState b = covered.get(1);
    for (GrpState state : List.of(a, b, a, a)) PhiResolverTest.values(single, state);
    Phi.LateStats stats = single.stats().late();
    Assert.assertEquals(stats.requests(), 4, stats.toString());
    Assert.assertEquals(stats.misses(), 3, stats.toString());
    Assert.assertEquals(stats.hits(), 1, stats.toString());
  }

  /**
   * 開いた LATE の Phi について、次のことを確かめる。前に開いたどの Phi とも別のものであること。数え上げる局末の値が、直に求めた値と
   * ビットで同じこと。初めの 1 巡で 1 度もメモに当たらないこと。前の値とどこかの局末で違うこと。
   */
  private static Phi assertFresh(
      ActivePhi active, Path leaf, Path q, double cutoff, List<GrpState> covered, List<Phi> opened, List<long[][]> seen)
      throws IOException {
    Assert.assertEquals(active.value().mode(), PhiMode.LATE);
    Assert.assertEquals((double) active.value().cutoff(), cutoff);
    Phi phi = active.phi();
    Assert.assertEquals(phi.kind(), Phi.Kind.LATE);
    for (Phi earlier : opened) Assert.assertNotSame(phi, earlier);
    Phi direct =
        Phi.late(
            FinalRound.of(GrpCheckpoint.readOutcomes(q), (GrpNetwork) GrpCheckpoint.read(leaf), cutoff),
            GrpFixtures.UTILITY, 16);
    Phi.LateStats before = phi.stats().late();
    long[][] bits = new long[covered.size()][];
    for (int index = 0; index < covered.size(); index++) {
      GrpState state = covered.get(index);
      bits[index] = PhiResolverTest.bits(PhiResolverTest.values(phi, state));
      Assert.assertEquals(bits[index], PhiResolverTest.bits(PhiResolverTest.values(direct, state)), state.toString());
    }
    Phi.LateStats pass = phi.stats().late().since(before);
    Assert.assertEquals(pass.requests(), covered.size(), pass.toString());
    Assert.assertEquals(pass.hits(), 0, pass.toString());
    if (!seen.isEmpty()) {
      long[][] last = seen.getLast();
      boolean differs = false;
      for (int index = 0; index < covered.size(); index++) differs |= !Arrays.equals(bits[index], last[index]);
      Assert.assertTrue(differs, "the new value changes some covered hand end");
    }
    opened.add(phi);
    seen.add(bits);
    return phi;
  }

  /** 合成の半荘から取った、数え上げる局末。どれもメモのキーを持ち、キーは互いに重ならない。 */
  private static List<GrpState> covered() {
    Set<Long> keys = new HashSet<>();
    List<GrpState> covered = new ArrayList<>();
    for (GameRecord game : GrpFixtures.games(0, 60, 5))
      for (GrpState state : game.handEnds())
        if (FinalRound.covers(state) && state.key() != GrpKey.NONE && keys.add(state.key())) covered.add(state);
    Assert.assertTrue(covered.size() >= 10, covered.size() + " covered");
    return covered;
  }

  /** 半荘 400〜405 を 4 本の卓のスレッドで集め、確定した行の数で報告を作る。 */
  private static CollectReport collect(Phi phi) throws IOException {
    List<Long> ids = new ArrayList<>();
    for (long id = 400; id < 406; id++) ids.add(id);
    Path directory = Files.createTempDirectory("prism-late-rows");
    try (var table = new TableLoop(new TableSettings(8, 4, 1000L, 4));
        var writer = new CohortWriter(directory, TENHOU_4P, 4, 0, 2, CohortWriter.Tap.NONE)) {
      var cohort =
          new Collector.Cohort(
              TENHOU_4P, 9_301L, 9_301L * 5, ids, DrawOrder.SEAT_STREAMS, SAMPLING, null, URA, FLOOR,
              TwinDiscard.TSUMOGIRI, null);
      Collector.Result result =
          Collector.collect(table, cohort, new GameLogObserverTest.HashedScorePolicy(), phi, new RowSink(writer));
      writer.seal();
      return CollectReport.of(1, result, result.rows().rows(), 0);
    } finally {
      Snapshot.deleteTree(directory);
    }
  }
}
