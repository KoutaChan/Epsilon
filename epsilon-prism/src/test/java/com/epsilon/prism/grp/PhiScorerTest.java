package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 裏の採点（{@link PhiScorer}）を小さな合成の記録で確かめる（測りは Φ の値から決まる偽の {@link CellReport}）:
 * 採点するコホートは未採点で遅れが 8 以内のもの。使用中の Φ と影の Φ を記録の半荘で評価して 1 行書き、EWMA は同じ項目なら λ = 1/8 で進め、
 * 項目が変わったら初めから。影が開けなければその影だけを飛ばし、記録が読めなければ何も書かない。書きかけの行は読み飛ばし、次の行は読める。
 */
public class PhiScorerTest {
  private static final EpsilonUtilityProfile PROFILE = EpsilonUtilityProfile.TENHOU;

  private Path run;
  private GrpFiles files;
  private PrismConfig config;

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("prism-phimon");
    files = new GrpFiles(run);
    config =
        PrismConfig.fromSettings(
            EpsilonSettings.of(
                Map.of(
                    PrismConfig.SETTINGS + "phiSource", "SCORE_RANK",
                    PrismConfig.SETTINGS + "phiMonitor", "SCORE")));
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(run);
  }

  @Test
  public void pendingSkipsScoredAndLaggingCohorts() throws IOException {
    Assert.assertEquals(PhiScorer.pending(files), List.of());
    for (int t = 1; t <= 12; t++) write(t, PhiEntry.SCORE_RANK_ID, "", 3);
    Assert.assertEquals(PhiScorer.pending(files), List.of(4, 5, 6, 7, 8, 9, 10, 11, 12));
    PhiScorer.score(files, config, 6, new Fake());
    Assert.assertEquals(PhiScorer.pending(files), List.of(4, 5, 7, 8, 9, 10, 11, 12));
    // 書きかけの一時ファイルは記録に数えない
    Files.writeString(files.games(13).resolveSibling("c00013.bin.gz.tmp"), "partial");
    Assert.assertEquals(PhiScorer.pending(files).size(), 8);
  }

  @Test
  public void scoresActiveAndShadowsAndRestartsTheEwmaOnASwitch() throws IOException {
    write(4, PhiEntry.SCORE_RANK_ID, "", 5);
    write(5, PhiEntry.SCORE_RANK_ID, "", 9);
    write(6, "g006-a1", "cd".repeat(32), 4);
    var fake = new Fake();
    PhiMonitorLine first = PhiScorer.score(files, config, 4, fake).orElseThrow();
    Assert.assertEquals(first.cohort(), 4);
    Assert.assertEquals(first.phiId(), PhiEntry.SCORE_RANK_ID);
    Assert.assertEquals(first.active().games(), 5);
    Assert.assertEquals(first.active().dSpg(), Fake.dSpg(5, PhiEntry.SCORE_RANK_ID));
    Assert.assertEquals(first.shadows().keySet(), Set.of("g001-a1"));
    Assert.assertEquals(first.shadows().get("g001-a1").dSpg(), Fake.dSpg(5, "g001-a1"));
    Assert.assertEquals(first.ewmaCohorts(), 1);
    Assert.assertEquals(first.dSpgEwma(), first.active().dSpg());
    // 同じ項目: λ = 1/8 で進める
    PhiMonitorLine second = PhiScorer.score(files, config, 5, fake).orElseThrow();
    Assert.assertEquals(second.ewmaCohorts(), 2);
    Assert.assertEquals(
        second.dSpgEwma(), first.dSpgEwma() + (second.active().dSpg() - first.dSpgEwma()) / 8, 1e-15);
    // 項目が変わったら初めから
    PhiMonitorLine third = PhiScorer.score(files, config, 6, fake).orElseThrow();
    Assert.assertEquals(third.phiId(), "g006-a1");
    Assert.assertEquals(third.ewmaCohorts(), 1);
    Assert.assertEquals(third.dSpgEwma(), third.active().dSpg());
    List<PhiMonitorLine> lines = PhiMonitorLine.readAll(files);
    Assert.assertEquals(lines.stream().map(PhiMonitorLine::cohort).toList(), List.of(4, 5, 6));
    Assert.assertEquals(lines.get(1).dSpgEwma(), second.dSpgEwma());
    Assert.assertEquals(lines.get(2).shadows().get("g001-a1").b4(), third.shadows().get("g001-a1").b4());
    Assert.assertTrue(Double.isNaN(lines.get(2).active().fourthEce()), "NaN survives the round trip");
    String summary = lines.get(1).summary();
    Assert.assertTrue(summary.startsWith("phi[5] score-rank D="), summary);
    Assert.assertTrue(summary.contains("z=S3:"), summary);
  }

  @Test
  public void failedShadowsAreSkippedAndUnreadableLogsWriteNothing() throws IOException {
    write(2, PhiEntry.SCORE_RANK_ID, "", 3);
    var fake = new Fake();
    fake.brokenShadow = true;
    PhiMonitorLine line = PhiScorer.score(files, config, 2, fake).orElseThrow();
    Assert.assertEquals(line.shadows(), Map.of());
    // 壊れた記録・ない記録・別のコホートの記録は採点しない（何も書かない）
    Files.writeString(files.games(3), "not a game log");
    Assert.assertEquals(PhiScorer.score(files, config, 3, fake), Optional.empty());
    Assert.assertEquals(PhiScorer.score(files, config, 7, fake), Optional.empty());
    GameLog.write(files.games(8), GameLogTest.synthetic(9, 2, 1));
    Assert.assertEquals(PhiScorer.score(files, config, 8, fake), Optional.empty());
    Assert.assertEquals(PhiMonitorLine.readAll(files).size(), 1);
  }

  @Test
  public void brokenLinesAreSkipped() throws IOException {
    write(2, PhiEntry.SCORE_RANK_ID, "", 3);
    write(3, PhiEntry.SCORE_RANK_ID, "", 3);
    var fake = new Fake();
    PhiScorer.score(files, config, 2, fake);
    // 書きかけで止まった行（改行なし）
    Files.writeString(files.monitor(), "{\"cohort\":3,\"phiId\":\"sc", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
    Assert.assertEquals(PhiMonitorLine.readAll(files).size(), 1);
    PhiScorer.score(files, config, 3, fake);
    List<PhiMonitorLine> lines = PhiMonitorLine.readAll(files);
    Assert.assertEquals(lines.stream().map(PhiMonitorLine::cohort).toList(), List.of(2, 3));
    Assert.assertEquals(lines.get(1).ewmaCohorts(), 2);
  }

  @Test
  public void productionToolsResolveScoreOnlyEntries() throws IOException {
    var header = new GameLogHeader(1, PhiEntry.SCORE_RANK_ID, "", 0.005, 0, 0, 0);
    Assert.assertEquals(PhiScorer.PRODUCTION.active(files, config, header), PhiEntry.scoreRank());
    Assert.assertEquals(PhiScorer.PRODUCTION.shadows(files, config, PhiEntry.scoreRank()), List.of());
    // 元の GRP の記録は grpCheckpoint がなければ引き直せない
    var fixed = new GameLogHeader(1, PhiEntry.FIXED_ID, "ab", 0.005, 0, 0, 0);
    Assert.assertThrows(IOException.class, () -> PhiScorer.PRODUCTION.active(files, config, fixed));
    // 点数の順位の値は点数の順位の効用
    write(1, PhiEntry.SCORE_RANK_ID, "", 2);
    CohortGames games = GameLog.read(files.games(1));
    try (HandEndPhi phi = PhiScorer.PRODUCTION.open(PhiEntry.scoreRank(), PROFILE)) {
      PhiValues values = phi.gameValues(games.games());
      int continuing = 0;
      for (RecordedGame game : games.games()) continuing += game.continuingHandEnds();
      Assert.assertEquals(values.size(), continuing);
    }
  }

  /** コホート t の合成の記録（Φ の項目 id・sha）を書く。 */
  private void write(int t, String phiId, String sha, int games) throws IOException {
    CohortGames synthetic = GameLogTest.synthetic(t, games, t);
    GameLog.write(
        files.games(t),
        new CohortGames(new GameLogHeader(t, phiId, sha, 0.005, 0, 1000, 4), synthetic.games()));
  }

  /**
   * 偽の道具: 使用中の項目は記録の先頭の id と sha の SELF_CHECKPOINT（点数の順位は本物）、影は g001-a1 の 1 つ、Φ は全て点数の順位、測りは
   * 半荘の数と開いた項目の id から決まる値（{@link #dSpg}。採点は項目ごとに開いてすぐ測るので、最後に開いた項目の id を使う）。
   */
  static final class Fake implements PhiScorer.Tools {
    boolean brokenShadow;
    private String opened = "";

    static double dSpg(int games, String id) {
      return games * 1e-4 + id.length() * 1e-6;
    }

    @Override
    public PhiEntry active(GrpFiles files, PrismConfig config, GameLogHeader header) {
      if (header.phiId().equals(PhiEntry.SCORE_RANK_ID)) return PhiEntry.scoreRank();
      return new PhiEntry(header.phiId(), PhiEntryKind.SELF_CHECKPOINT, "/grp", header.phiSha256(), "", "", 1, "", "test");
    }

    @Override
    public List<PhiEntry> shadows(GrpFiles files, PrismConfig config, PhiEntry active) {
      return List.of(new PhiEntry("g001-a1", PhiEntryKind.SELF_CHECKPOINT, "/shadow", "ef", "", "", 1, "", "test"));
    }

    @Override
    public HandEndPhi open(PhiEntry entry, EpsilonUtilityProfile profile) throws IOException {
      if (brokenShadow && entry.id().equals("g001-a1")) throw new IOException("unreadable shadow");
      opened = entry.id();
      return ScorePhi.offline(PhiEntryKind.SCORE_RANK, profile);
    }

    @Override
    public CellReport measure(List<RecordedGame> games, PhiValues values, EpsilonUtilityProfile profile) {
      int continuing = 0;
      for (RecordedGame game : games) continuing += game.continuingHandEnds();
      Assert.assertEquals(values.size(), continuing);
      double[] b4 = new double[HandGroup.values().length];
      Arrays.fill(b4, 0.01);
      double[] z = new double[HandGroup.values().length];
      z[HandGroup.S3.ordinal()] = -2.5;
      return new CellReport(
          games.size(), 4L * continuing, 0.01, dSpg(games.size(), opened), 1e-5, b4, b4.clone(), b4.clone(), z, 1.2, 0.5, Double.NaN);
    }
  }
}
