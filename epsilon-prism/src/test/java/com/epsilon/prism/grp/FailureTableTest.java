package com.epsilon.prism.grp;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.ai.grp.EpsilonGrpInference;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.prism.collect.GrpPhi;
import com.epsilon.prism.collect.PhiCalibration;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 学習の JVM の側の失敗の扱い: まだ効いたことのない候補が読めない・sha が違う・READY でない・境目の点検に落ちるなら REJECTED と
 * 前の項目の行を書いて続け（再開しても同じ）、既に効いたことのある項目が読めなければ止める。有限でない Φ は完了を渡さずに評価の失敗にする。
 * 学習の失敗は refits.jsonl の行にして続ける（{@link GrpRefitterTest}）。
 */
public class FailureTableTest {

  @Test(timeOut = 600_000)
  public void candidatesThatCannotBeTrustedAreRejectedAndThePreviousPhiStays() throws Exception {
    Path run = Files.createTempDirectory("prism-failure");
    ExecutorService driver = Executors.newSingleThreadExecutor();
    try {
      var files = new GrpFiles(run);
      GrpFixtures.writeState(run, 1);
      List<RecordedGame> games = GrpFixtures.games(0, 32, 5);
      Path good = GrpFixtures.checkpoint(run.resolve("good"));
      Path gone = GrpFixtures.checkpoint(run.resolve("gone"));
      try (PhiResolver resolver = PhiResolver.open(GrpFixtures.scoreStart(), GrpFixtures.PROFILE, driver::execute, files);
          HandEndPhi phi = PhiResolverTest.direct(good)) {
        resolver.resolve(1);
        // 3: 境目の点検に落ちる（見本を 1e-3 ずらす）
        RefitRecord boundary = GrpFixtures.ready(files, "g003-a1", good, phi, games);
        BoundaryCheck sample = BoundaryCheck.read(files.boundary("g003-a1"));
        sample.utilities()[0][0] += 1e-3;
        sample.write(files.boundary("g003-a1"));
        PhiHistory.append(files, boundary.entry(3, "test"));
        // 4: checkpoint が消えた
        PhiHistory.append(files, GrpFixtures.ready(files, "g004-a1", gone, phi, games).entry(4, "test"));
        Snapshot.deleteTree(gone);
        // 5: sha が違う
        RefitRecord wrong = GrpFixtures.ready(files, "g005-a1", good, phi, games);
        wrong = new RefitRecord(
            wrong.id(), RefitStatus.READY, wrong.cause(), wrong.window(), wrong.init(), wrong.parent(), wrong.path(),
            "0".repeat(64), wrong.jarSha256(), null, "", 1);
        wrong.append(files);
        PhiHistory.append(files, wrong.entry(5, "test"));
        // 6: refits.jsonl に READY の行がない
        PhiHistory.append(
            files, new PhiEntry("g006-a1", PhiEntryKind.SELF_CHECKPOINT, good.toString(), boundary.sha256(), "", "", 6, "g000", "test"));
        for (int t = 3; t <= 6; t++) {
          ActivePhi active = resolver.resolve(t);
          Assert.assertEquals(active.entry().id(), PhiEntry.INITIAL_ID, "t=" + t);
          Assert.assertEquals(active.entry().kind(), PhiEntryKind.SCORE_RANK, "t=" + t);
          Assert.assertFalse(active.resetsLedger());
        }
        List<RefitRecord> records = RefitRecord.readAll(files);
        for (String id : List.of("g003-a1", "g004-a1", "g005-a1", "g006-a1"))
          Assert.assertEquals(RefitRecord.latest(records, id).orElseThrow().status(), RefitStatus.REJECTED, id);
        Assert.assertTrue(RefitRecord.latest(records, "g003-a1").orElseThrow().detail().contains("boundary"));
        List<PhiEntry> entries = PhiHistory.read(files);
        for (int t = 3; t <= 6; t++) {
          PhiEntry effective = PhiHistory.effectiveAt(entries, t).orElseThrow();
          Assert.assertEquals(effective.id(), PhiEntry.INITIAL_ID);
          Assert.assertTrue(effective.reason().startsWith("rejected g00" + t + "-a1"), effective.reason());
        }
        // 拒んだ候補は grp-switch でも入れ直せない
        GrpFixtures.writeConfig(run, GrpFixtures.scoreStart());
        Assert.assertThrows(
            IllegalArgumentException.class,
            () -> new GrpSwitchWorkflow().execute(GrpRefitterTest.arguments("--run", run.toString(), "--to", "g003-a1")));
      }
      // 再開しても同じ歴史
      try (PhiResolver resolver = PhiResolver.open(GrpFixtures.scoreStart(), GrpFixtures.PROFILE, driver::execute, files)) {
        Assert.assertEquals(resolver.resolve(4).entry().kind(), PhiEntryKind.SCORE_RANK);
        // 既に効いたことのある項目（7 で効いた候補）は、読めなくなったら止める
        try (HandEndPhi phi = PhiResolverTest.direct(good)) {
          PhiHistory.append(files, GrpFixtures.ready(files, "g007-a1", good, phi, games).entry(7, "test"));
        }
        Assert.assertEquals(resolver.resolve(7).entry().id(), "g007-a1");
      }
      Snapshot.deleteTree(good);
      try (PhiResolver resolver = PhiResolver.open(GrpFixtures.scoreStart(), GrpFixtures.PROFILE, driver::execute, files)) {
        Assert.assertThrows(IOException.class, () -> resolver.resolve(8));
      }
    } finally {
      driver.shutdownNow();
      Snapshot.deleteTree(run);
    }
  }

  @Test(timeOut = 300_000)
  public void anUnreadableG000StopsTheLineage() throws Exception {
    Path run = Files.createTempDirectory("prism-failure");
    ExecutorService driver = Executors.newSingleThreadExecutor();
    try {
      Path decision = run.resolve("decision");
      GrpFixtures.checkpoint(decision);
      var files = new GrpFiles(run);
      try (PhiResolver resolver = PhiResolver.open(GrpFixtures.grpStart(decision), GrpFixtures.PROFILE, driver::execute, files)) {
        Assert.assertEquals(resolver.resolve(1).entry().kind(), PhiEntryKind.FIXED_CHECKPOINT);
      }
      Snapshot.deleteTree(decision);
      try (PhiResolver resolver = PhiResolver.open(GrpFixtures.grpStart(decision), GrpFixtures.PROFILE, driver::execute, files)) {
        Assert.assertThrows(IOException.class, () -> resolver.resolve(2));
      }
      Assert.assertTrue(RefitRecord.readAll(files).isEmpty(), "g000 is never rejected");
    } finally {
      driver.shutdownNow();
      Snapshot.deleteTree(run);
    }
  }

  @Test(timeOut = 300_000)
  public void aNonFinitePhiFailsTheEvaluationInsteadOfReachingALabel() throws Exception {
    Path run = Files.createTempDirectory("prism-failure");
    ExecutorService executor = Executors.newSingleThreadExecutor();
    try {
      Path checkpoint = GrpFixtures.checkpoint(run.resolve("decision"));
      Path table = run.resolve("nan.json");
      Files.writeString(table, "{\"stages\": [0], \"lo\": -1, \"hi\": 0.667, \"bins\": 2, \"table\": {\"0\": [NaN, NaN]}}");
      List<Throwable> failures = Collections.synchronizedList(new ArrayList<>());
      List<double[]> completed = Collections.synchronizedList(new ArrayList<>());
      Consumer<Runnable> driver =
          task ->
              executor.execute(
                  () -> {
                    try {
                      task.run();
                    } catch (RuntimeException failure) {
                      failures.add(failure);
                    }
                  });
      HandEndInput input = GrpFixtures.games(0, 1, 1).getFirst().input(1);
      try (Model model = EpsilonGrpCheckpointManager.load(checkpoint, Device.cpu());
          var phi =
              new GrpPhi(
                  new EpsilonGrpInference(model, PhiResolverTest.MAX_BATCH),
                  GrpFixtures.PROFILE,
                  driver,
                  256,
                  1024,
                  new double[0],
                  PhiCalibration.load(table))) {
        driver.accept(() -> phi.request(input.prefix(), input.next().scores(), completed::add));
        long deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(2);
        while (failures.isEmpty() && System.nanoTime() < deadline) Thread.sleep(10);
        Assert.assertEquals(failures.size(), 1, "the evaluation must fail");
        Assert.assertTrue(failures.getFirst().getCause().getMessage().contains("not finite"), failures.getFirst().toString());
        driver.accept(() -> phi.request(input.prefix(), input.next().scores(), completed::add));
        deadline = System.nanoTime() + TimeUnit.MINUTES.toNanos(1);
        while (failures.size() < 2 && System.nanoTime() < deadline) Thread.sleep(10);
        Assert.assertTrue(failures.get(1) instanceof IllegalStateException, failures.get(1).toString());
        Assert.assertTrue(completed.isEmpty(), "a non-finite Φ never reaches a label");
      }
    } finally {
      executor.shutdownNow();
      Snapshot.deleteTree(run);
    }
  }
}
