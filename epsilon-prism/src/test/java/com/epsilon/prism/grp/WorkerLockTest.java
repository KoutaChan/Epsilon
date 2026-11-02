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
import java.nio.file.attribute.FileTime;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 裏の作業の部品: 錠（同じ種類は 1 つだけ、種類どうしは独立、外せば取り直せる）、心拍（pid・状態・終わり）、計測の札（自分の札は数えず、ほかの
 * 札が消えるまで待つ）、jar の写し（sha の名前、同じ sha は写し直さない）、主クラス（引数の誤りだけ 0 でなく、作業の失敗でも 0 で錠を残さない）。
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

  @Test(timeOut = 30_000)
  public void foreignHoldsAreAwaited() throws Exception {
    Path holds = run.resolve("holds");
    ForeignHolds.awaitClear(holds, "HOLD.train.v2b", Duration.ofMillis(10));
    Files.createDirectories(holds);
    Files.createFile(holds.resolve("HOLD.train.v2b"));
    Files.createFile(holds.resolve("WAITING.train"));
    ForeignHolds.awaitClear(holds, "HOLD.train.v2b", Duration.ofMillis(10));
    Path foreign = Files.createFile(holds.resolve("HOLD.train.darch"));
    Assert.assertEquals(ForeignHolds.foreign(holds, "HOLD.train.v2b"), List.of("HOLD.train.darch"));
    Thread remover =
        new Thread(
            () -> {
              try {
                Thread.sleep(300);
                Files.delete(foreign);
              } catch (IOException | InterruptedException ignored) {
                // 試験の失敗は待ちの時間切れで分かる
              }
            });
    long started = System.nanoTime();
    remover.start();
    ForeignHolds.awaitClear(holds, "HOLD.train.v2b", Duration.ofMillis(20));
    Assert.assertTrue(System.nanoTime() - started >= 250_000_000L, "returned before the hold was removed");
    remover.join();
  }

  @Test
  public void jarCopiesAreNamedBySha() throws IOException {
    Path jar = run.resolve("epsilon-prism.jar");
    Files.write(jar, "jar bytes".getBytes());
    Path copy = WorkerJar.copy(files, jar);
    String sha = WorkerJar.sha256(jar);
    Assert.assertEquals(copy, files.jars().resolve(sha.substring(0, 16) + ".jar"));
    Assert.assertEquals(Files.readAllBytes(copy), Files.readAllBytes(jar));
    FileTime written = FileTime.fromMillis(1_000_000L);
    Files.setLastModifiedTime(copy, written);
    Assert.assertEquals(WorkerJar.copy(files, jar), copy);
    Assert.assertEquals(Files.getLastModifiedTime(copy), written, "the same sha is not copied again");
    // 中身が変われば別の写し（前の写しは残す）
    Files.write(jar, "new jar bytes".getBytes());
    Path next = WorkerJar.copy(files, jar);
    Assert.assertNotEquals(next, copy);
    try (var listed = Files.list(files.jars())) {
      Assert.assertEquals(listed.count(), 2L);
    }
    Assert.assertThrows(IOException.class, () -> WorkerJar.copy(files, run.resolve("missing.jar")));
    // 試験はクラスのディレクトリから動くので sha はない
    Assert.assertEquals(WorkerJar.runningSha256(), "");
  }

  @Test
  public void workerMainExitsZeroAndReleasesItsLocks() throws IOException {
    Assert.assertEquals(GrpWorkerMain.run(new String[] {"--run"}), 2);
    Assert.assertEquals(GrpWorkerMain.run(new String[] {"--run", run.toString()}), 2);
    Assert.assertEquals(GrpWorkerMain.run(new String[] {"--run", run.toString(), "--copy-jar", "--own-hold", "x"}), 2);
    Assert.assertEquals(GrpWorkerMain.run(new String[] {"--run", run.toString(), "--bogus", "1"}), 2);
    String[] work = {"--run", run.toString(), "--hold-dir", run.resolve("holds").toString(), "--own-hold", "HOLD.train.v2b"};
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
    // 写しは今の jar（試験はクラスのディレクトリ）が写せなくても、起動できるパスを返す
    Assert.assertEquals(GrpWorkerMain.copyJar(files), WorkerJar.running());
    Assert.assertEquals(GrpWorkerMain.lastSealed(files), 0, "cohort 3 has a game log but no sealed rows");
  }

  private static JsonObject heartbeat(GrpFiles files) throws IOException {
    return JsonParser.parseString(Files.readString(files.heartbeat(WorkerLock.Kind.SCORE))).getAsJsonObject();
  }
}
