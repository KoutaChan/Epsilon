package com.epsilon.prism.grp;

import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.grp.train.GrpMetrics;
import com.epsilon.grp.train.GrpReport;
import com.epsilon.grp.train.HandEndValues;
import com.epsilon.grp.train.HandEnds;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 裏の採点（{@link PhiScorer}）を、小さな合成の記録と小さな乱数のネットワークで確かめる。採点するのは、まだ採点しておらず、最新の
 * コホートからの遅れが 8 以内のコホート。使用中の Φ（記録の先頭の id で引き直す）と、影の Φ（READY の候補のうち新しい 2 つ）を
 * 記録の半荘で測って 1 行を書く。測った値は、Φ を直に開いて測った値と同じ。影が開けなければその影だけを飛ばし、記録が読めないか、記録の
 * 先頭の id が phi.jsonl に無ければ何も書かない。書きかけの行は読み飛ばす。
 */
public class PhiScorerTest {
  private Path run;
  private GrpFiles files;
  private PhiSetup setup;
  private PhiEntry initial;

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("prism-phimon");
    files = new GrpFiles(run);
    setup = GrpFixtures.scoreStart("phi.grp.monitor=SCORE");
    initial = PhiHistory.initialize(files, setup);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(run);
  }

  private void write(int t, PhiEntry entry, int games) throws IOException {
    GrpFixtures.record(files, t, entry, GrpFixtures.games(1000L * t, games, t));
  }

  @Test
  public void pendingSkipsScoredAndLaggingCohorts() throws IOException {
    Assert.assertEquals(PhiScorer.pending(files), List.of());
    for (int t = 1; t <= 12; t++) write(t, initial, 3);
    Assert.assertEquals(PhiScorer.pending(files), List.of(4, 5, 6, 7, 8, 9, 10, 11, 12));
    PhiScorer.score(files, setup, 6).orElseThrow();
    Assert.assertEquals(PhiScorer.pending(files), List.of(4, 5, 7, 8, 9, 10, 11, 12));
    // 書きかけの一時ファイルは記録に数えない
    Files.writeString(files.records(13).resolveSibling("c00013.bin.tmp"), "partial");
    Assert.assertEquals(PhiScorer.pending(files).size(), 8);
  }

  @Test
  public void scoresTheActiveAndTheNewestReadyShadows() throws IOException {
    Path a = GrpFixtures.network(run.resolve("a"), 1);
    Path b = GrpFixtures.network(run.resolve("b"), 2);
    Path c = GrpFixtures.network(run.resolve("c"), 3);
    PhiEntry first = GrpFixtures.ready(files, "g001-a1", 1, a).entry(GrpFixtures.VARIANT, 5, "test");
    GrpFixtures.ready(files, "g002-a1", 2, b);
    GrpFixtures.ready(files, "g003-a1", 3, c);
    PhiHistory.append(files, first);
    write(4, initial, 20);
    write(5, first, 20);
    PhiMonitorLine line = PhiScorer.score(files, setup, 4).orElseThrow();
    Assert.assertEquals(line.cohort(), 4);
    Assert.assertEquals(line.phiId(), PhiEntry.INITIAL_ID);
    Assert.assertEquals(line.shadows().keySet(), Set.of("g002-a1", "g003-a1"));
    HandEnds ends = ends(4);
    Assert.assertEquals(line.active(), measure(initial, ends));
    Assert.assertEquals(line.active().games(), 20);
    PhiEntry shadow =
        new PhiEntry("g003-a1", PhiEntryKind.NETWORK, GrpFixtures.VARIANT, c.toString(), "", "", "", "", "", 1, "", "x");
    Assert.assertEquals(line.shadows().get("g003-a1"), measure(shadow, ends));
    // 使用中の候補は影から外す
    PhiMonitorLine second = PhiScorer.score(files, setup, 5).orElseThrow();
    Assert.assertEquals(second.phiId(), "g001-a1");
    Assert.assertFalse(second.shadows().containsKey("g001-a1"));
    Assert.assertEquals(second.active(), measure(first, ends(5)));
    List<PhiMonitorLine> lines = PhiMonitorLine.readAll(files);
    Assert.assertEquals(lines, List.of(line, second), "the lines survive the round trip");
    Assert.assertTrue(lines.get(1).summary().startsWith("phi[5] g001-a1 mse="), lines.get(1).summary());
  }

  @Test
  public void failedShadowsAreSkippedAndUnreadableRecordsWriteNothing() throws IOException {
    Path gone = GrpFixtures.network(run.resolve("gone"), 1);
    GrpFixtures.ready(files, "g001-a1", 1, gone);
    Snapshot.deleteTree(gone);
    write(2, initial, 3);
    PhiMonitorLine line = PhiScorer.score(files, setup, 2).orElseThrow();
    Assert.assertEquals(line.shadows(), Map.of());
    // 壊れた記録、無い記録、別のコホートの記録、phi.jsonl に無い項目で集めた記録は採点せず、何も書かない
    GrpFixtures.seal(files, 3);
    Files.writeString(files.records(3), "not a game record");
    Assert.assertEquals(PhiScorer.score(files, setup, 3), Optional.empty());
    Assert.assertEquals(PhiScorer.score(files, setup, 7), Optional.empty());
    write(8, initial, 2);
    Files.copy(files.records(8), files.records(9));
    Assert.assertEquals(PhiScorer.score(files, setup, 9), Optional.empty());
    PhiEntry unknown = GrpFixtures.selfEntry("g004-a1", 1, "ab");
    write(10, unknown, 2);
    Assert.assertEquals(PhiScorer.score(files, setup, 10), Optional.empty());
    Assert.assertEquals(PhiMonitorLine.readAll(files).size(), 1);
  }

  @Test
  public void brokenLinesAreSkipped() throws IOException {
    write(2, initial, 3);
    write(3, initial, 3);
    PhiScorer.score(files, setup, 2);
    // 書きかけで止まった、改行の無い行
    Files.writeString(files.monitor(), "{\"cohort\":3,\"phiId\":\"sc", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
    Assert.assertEquals(PhiMonitorLine.readAll(files).size(), 1);
    PhiScorer.score(files, setup, 3);
    List<PhiMonitorLine> lines = PhiMonitorLine.readAll(files);
    Assert.assertEquals(lines.stream().map(PhiMonitorLine::cohort).toList(), List.of(2, 3));
  }

  /** 元の GRP で集めた記録は、設定にある固定の GRP で引き直す。 */
  @Test
  public void theOriginalGrpComesFromTheSettings() throws IOException {
    Path original = GrpFixtures.network(run.resolve("original"), 5);
    PhiSetup fixed = GrpFixtures.setup("phi.source=FIXED_GRP", "phi.grp=" + original);
    PhiEntry entry = PhiScorer.original(fixed);
    Assert.assertEquals(entry.id(), PhiEntry.FIXED_ID);
    write(2, entry, 4);
    PhiMonitorLine line = PhiScorer.score(files, fixed, 2).orElseThrow();
    Assert.assertEquals(line.phiId(), PhiEntry.FIXED_ID);
    Assert.assertEquals(line.shadows(), Map.of());
    Assert.assertEquals(line.active(), measure(entry, ends(2)));
    Assert.assertThrows(IOException.class, () -> PhiScorer.original(setup));
  }

  private HandEnds ends(int t) throws IOException {
    return HandEnds.of(GrpFixtures.VARIANT, GrpFixtures.UTILITY, GameRecordCodec.read(files.records(t)).games());
  }

  private static GrpReport measure(PhiEntry entry, HandEnds ends) throws IOException {
    return GrpMetrics.measure(ends, HandEndValues.of(ends, HandEndPhi.open(entry, GrpFixtures.UTILITY)));
  }
}
