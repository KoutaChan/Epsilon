package com.epsilon.prism.grp;

import com.epsilon.prism.rows.CohortWriter;
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
 * 学習の JVM の GRP の裏の作業（{@link GrpBackground}）: 採点も SELF_GRP もなければ頼まれても何もしない。1 回の作業は未採点のコホートを
 * 1 度だけ採点して phimon.jsonl と drift.jsonl に書く。学習の取り置きは対局の記録がある最新の封印済みのコホート。
 */
public class GrpBackgroundTest {
  private Path run;
  private GrpFiles files;

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("prism-background");
    files = new GrpFiles(run);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(run);
  }

  @Test
  public void withoutScoringOrSelfGrpRequestsDoNothing() throws IOException {
    PrismConfig config = GrpFixtures.config("phiSource", "SCORE_RANK", "phiMonitor", "RECORD", "phiDrift", "OFF");
    GameLog.write(files.games(3), scoreRankGames(3));
    seal(3);
    try (GrpBackground background = GrpBackground.start(files, config)) {
      background.request();
      background.request();
    }
    Assert.assertFalse(Files.exists(files.monitor()), "nothing is scored");
    Assert.assertFalse(Files.exists(files.drift()));
  }

  @Test(timeOut = 60_000)
  public void aPassScoresEachRecordedCohortOnce() throws IOException {
    PrismConfig config = GrpFixtures.config("phiSource", "SCORE_RANK", "phiMonitor", "SCORE", "phiDrift", "REPORT");
    GameLog.write(files.games(3), scoreRankGames(3));
    try (GrpBackground background = GrpBackground.start(files, config)) {
      background.pass();
      background.pass();
    }
    Assert.assertEquals(PhiMonitorLine.readAll(files).stream().map(PhiMonitorLine::cohort).toList(), List.of(3));
    Assert.assertEquals(PhiHistory.lines(files.drift()).size(), 1, "the drift alarm advances once per scored cohort");
  }

  @Test
  public void lastSealedNeedsSealedRowsAndAGameLog() throws IOException {
    Assert.assertEquals(GrpBackground.lastSealed(files), 0);
    GameLog.write(files.games(3), GameLogTest.synthetic(3, 4, 3));
    Assert.assertEquals(GrpBackground.lastSealed(files), 0, "cohort 3 has a game log but no sealed rows");
    seal(3);
    GameLog.write(files.games(4), GameLogTest.synthetic(4, 4, 4));
    seal(6);
    Assert.assertEquals(GrpBackground.lastSealed(files), 3, "cohort 4 is not sealed and cohort 6 has no game log");
    seal(4);
    Assert.assertEquals(GrpBackground.lastSealed(files), 4);
  }

  /** 点数の順位で集めたコホートの記録。 */
  private static CohortGames scoreRankGames(int cohort) {
    return new CohortGames(
        new GameLogHeader(cohort, PhiEntry.SCORE_RANK_ID, PhiEntry.scoreRank().sha256(), 0.005, 0, 10, 0),
        GameLogTest.synthetic(cohort, 40, cohort).games());
  }

  /** コホートの行を封印済みにする。 */
  private void seal(int cohort) throws IOException {
    Files.createDirectories(files.cohort(cohort));
    Files.writeString(files.cohort(cohort).resolve(CohortWriter.MANIFEST), "{}");
  }
}
