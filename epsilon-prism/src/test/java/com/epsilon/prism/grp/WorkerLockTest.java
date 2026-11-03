package com.epsilon.prism.grp;

import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 裏の作業の部品: 錠（同じ種類は 1 つだけ、種類どうしは独立、外せば取り直せる）、心拍（pid・状態・終わり）、主クラス（引数の誤りだけ 0 で
 * なく、作業の失敗でも 0 で錠を残さない）、学習の JVM の採点のスレッド（錠を持つ作業があれば採点せず、閉じれば錠を残さない）。
 */
public class WorkerLockTest {
  private Path run;
  private GrpFiles files;

  @BeforeMethod
  public void createRun() throws IOException {
    run = Files.createTempDirectory("prism-worker");
    files = new GrpFiles(run);
  }

  @AfterMethod(alwaysRun = true)
  public void deleteRun() throws IOException {
    Snapshot.deleteTree(run);
  }

  @Test
  public void locksAreExclusivePerKind() throws IOException {
    Optional<WorkerLock> score = WorkerLock.tryAcquire(files, WorkerLock.Kind.SCORE);
    Assert.assertTrue(score.isPresent());
    Assert.assertTrue(WorkerLock.tryAcquire(files, WorkerLock.Kind.SCORE).isEmpty(), "second score lock");
    try (WorkerLock refit = WorkerLock.tryAcquire(files, WorkerLock.Kind.REFIT).orElseThrow()) {
      Assert.assertTrue(Files.exists(files.lock("refit")));
    }
    score.get().close();
    try (WorkerLock again = WorkerLock.tryAcquire(files, WorkerLock.Kind.SCORE).orElseThrow()) {
      Assert.assertTrue(Files.exists(files.lock("score")));
    }
  }

  @Test
  public void heartbeatWritesStateAndDone() throws IOException {
    try (var heartbeat = WorkerHeartbeat.start(files, WorkerLock.Kind.SCORE, "start")) {
      Assert.assertEquals(heartbeat(files).get("state").getAsString(), "start");
      heartbeat.state("scoring c00007");
      JsonObject beat = heartbeat(files);
      Assert.assertEquals(beat.get("state").getAsString(), "scoring c00007");
      Assert.assertEquals(beat.get("pid").getAsLong(), ProcessHandle.current().pid());
    }
    Assert.assertEquals(heartbeat(files).get("state").getAsString(), "done");
  }

  @Test
  public void workerMainExitsZeroAndReleasesItsLocks() throws IOException {
    Assert.assertEquals(GrpWorkerMain.run(new String[] {"--run"}), 2);
    Assert.assertEquals(GrpWorkerMain.run(new String[] {"--hold-dir", run.toString()}), 2);
    Assert.assertEquals(GrpWorkerMain.run(new String[] {"--run", run.toString(), "--copy-jar"}), 2);
    Assert.assertEquals(GrpWorkerMain.run(new String[] {"--run", run.toString(), "--bogus", "1"}), 2);
    String[] work = {"--run", run.toString()};
    // config.json がなくても系譜は止めない
    Assert.assertEquals(GrpWorkerMain.run(work), 0);
    PrismConfig config =
        PrismConfig.fromSettings(
            EpsilonSettings.of(
                Map.of(
                    PrismConfig.SETTINGS + "phiSource", "SCORE_RANK",
                    PrismConfig.SETTINGS + "phiMonitor", "SCORE",
                    PrismConfig.SETTINGS + "phiDrift", "OFF")));
    Files.writeString(
        run.resolve("config.json"), new GsonBuilder().serializeSpecialFloatingPointValues().create().toJson(config));
    GameLog.write(files.games(3), GameLogTest.synthetic(3, 4, 3));
    Assert.assertEquals(GrpWorkerMain.run(work), 0);
    Assert.assertEquals(heartbeat(files).get("state").getAsString(), "done");
    try (WorkerLock score = WorkerLock.tryAcquire(files, WorkerLock.Kind.SCORE).orElseThrow()) {
      Assert.assertTrue(Files.exists(files.lock("score")));
    }
    Assert.assertEquals(GrpWorkerMain.lastSealed(files), 0, "cohort 3 has a game log but no sealed rows");
  }

  @Test(timeOut = 60_000)
  public void scoringThreadScoresSealedCohortsOnce() throws IOException {
    PrismConfig config =
        PrismConfig.fromSettings(
            EpsilonSettings.of(
                Map.of(
                    PrismConfig.SETTINGS + "phiSource", "SCORE_RANK",
                    PrismConfig.SETTINGS + "phiMonitor", "SCORE",
                    PrismConfig.SETTINGS + "phiDrift", "OFF")));
    GameLog.write(
        files.games(3),
        new CohortGames(
            new GameLogHeader(3, PhiEntry.SCORE_RANK_ID, PhiEntry.scoreRank().sha256(), 0.005, 0, 10, 0),
            GameLogTest.synthetic(3, 40, 3).games()));
    // 錠を持つ作業がいれば採点しない
    try (WorkerLock held = WorkerLock.tryAcquire(files, WorkerLock.Kind.SCORE).orElseThrow();
        var scoring = PhiScoring.start(files, config)) {
      scoring.request();
    }
    Assert.assertEquals(PhiMonitorLine.readAll(files).size(), 0);
    try (var scoring = PhiScoring.start(files, config)) {
      scoring.request();
      scoring.request();
    }
    Assert.assertEquals(PhiMonitorLine.readAll(files).stream().map(PhiMonitorLine::cohort).toList(), List.of(3));
    try (WorkerLock score = WorkerLock.tryAcquire(files, WorkerLock.Kind.SCORE).orElseThrow()) {
      Assert.assertTrue(Files.exists(files.lock("score")));
    }
    // 採点しない設定ではスレッドを持たない
    try (var off = PhiScoring.start(files, PrismConfig.fromSettings(EpsilonSettings.of(Map.of(
        PrismConfig.SETTINGS + "phiSource", "SCORE_RANK", PrismConfig.SETTINGS + "phiMonitor", "RECORD",
        PrismConfig.SETTINGS + "phiDrift", "OFF"))))) {
      off.request();
    }
  }

  private static JsonObject heartbeat(GrpFiles files) throws IOException {
    return JsonParser.parseString(Files.readString(files.heartbeat(WorkerLock.Kind.SCORE))).getAsJsonObject();
  }
}
