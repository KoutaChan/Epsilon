package com.epsilon.agent.run;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.run.FakeRun.Point;
import com.epsilon.agent.run.FakeRun.Steps;
import com.epsilon.agent.run.RunStatus.State;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.SettingsException;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Stream;
import org.testng.annotations.Test;

/**
 * 段の並びを途中から続ける仕組み。完了マーカーの前後や反復を閉じる途中で落ちても、同じ結果と報告になる。ほかに、全ての
 * セーフポイントでの止め方、停止の取り消し、起動し直し、jar の入れ替え、設定の行の固定、途中で作ったジョブの回る順、status.json を
 * 確かめる。
 */
public class RunnerTest {

  /** 落とさずに回した結果。 */
  private static FakeRun.Outcome reference(String... sets) throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-reference"), sets);
    assertEquals(world.finish(), 1);
    return world.outcome();
  }

  @Test
  public void aRunWithoutFaultsRunsEveryStepOnceAndClosesEveryIteration() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run"));
    assertEquals(world.train(), Ending.FINISHED);
    FakeRun.Outcome outcome = world.outcome();
    assertEquals(outcome.iterations().size(), 2);
    assertEquals(outcome.extra().size(), 2);
    assertEquals(outcome.evaluations().size(), 2);
    assertEquals(outcome.jobs().keySet(), Set.of("v00002-x1n3", "v00003-x1n3"));
    for (int t = 1; t <= 2; t++) {
      for (Steps step : Steps.values()) {
        assertEquals(world.ran(t, step), 1, t + " " + step);
      }
    }
    // PHI = 7 + 1、COLLECT = 8 * 3 + 7、PUBLISH = 31 + 1。2 回目の反復の PHI は 32 + 2
    assertEquals(outcome.state(), "{\"next\":3,\"value\":110}\n");
    world.assertNoWork();
    RunStatus status = world.status();
    assertEquals(status.state(), State.FINISHED);
    assertEquals(status.iteration(), Integer.valueOf(3));
    assertEquals(status.launchRow(), Integer.valueOf(1));
    assertEquals(status.pid(), ProcessHandle.current().pid());
  }

  @Test
  public void aCrashBeforeOrAfterEveryDonePointContinuesToTheSameResultAndReport() throws Exception {
    FakeRun.Outcome expected = reference();
    for (int t = 1; t <= 2; t++) {
      for (Steps step : Steps.values()) {
        for (Point point : List.of(Point.BEFORE_SEAL, Point.AFTER_SEAL)) {
          FakeRun world = FakeRun.create(Files.createTempDirectory("run-crash"));
          world.fail(point, t, step);
          assertEquals(world.finish(), 2, point + " " + t + " " + step);
          assertEquals(world.crashed, point);
          assertEquals(world.outcome(), expected, point + " " + t + " " + step);
          // 完了マーカーの後で落ちた段は回し直さない。前で落ちた段だけをもう 1 回回す
          assertEquals(world.ran(t, step), point == Point.BEFORE_SEAL ? 2 : 1, point + " " + t + " " + step);
          for (Steps earlier : Steps.values()) {
            if (earlier.ordinal() < step.ordinal()) {
              assertEquals(world.ran(t, earlier), 1, point + " " + t + " " + step + " reran " + earlier);
            }
          }
          world.assertNoWork();
        }
      }
    }
  }

  @Test
  public void aCrashWhileClosingAnIterationContinuesWithoutDoubledLines() throws Exception {
    FakeRun.Outcome expected = reference();
    for (int t = 1; t <= 2; t++) {
      for (Point point : List.of(Point.CLOSE, Point.REPORT, Point.JOBS, Point.CLEAN)) {
        FakeRun world = FakeRun.create(Files.createTempDirectory("run-close"));
        world.fail(point, t);
        assertEquals(world.finish(), 2, point + " " + t);
        assertEquals(world.crashed, point);
        assertEquals(world.outcome(), expected, point + " " + t);
        for (Steps step : Steps.values()) {
          assertEquals(world.ran(t, step), 1, point + " " + t + " " + step);
        }
        world.assertNoWork();
      }
    }
  }

  @Test
  public void aStepFailureWritesFailedWithTheReasonAndTheNextLaunchContinues() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-fail"));
    world.fail(Point.STEP_FAIL, 1, Steps.COLLECT);
    IllegalStateException failure = expectThrows(IllegalStateException.class, world::train);
    assertEquals(failure.getMessage(), "boom at STEP_FAIL 1 COLLECT");
    RunStatus status = world.status();
    assertEquals(status.state(), State.FAILED);
    assertEquals(status.reason(), "IllegalStateException: boom at STEP_FAIL 1 COLLECT");
    assertEquals(status.iteration(), Integer.valueOf(1));
    assertEquals(status.settingsRow(), Integer.valueOf(1));
    assertEquals(world.train(), Ending.FINISHED);
    assertEquals(world.outcome(), reference());
    assertEquals(world.ran(1, Steps.PHI), 1);
  }

  /** 落とさずに回したとき、セーフポイントで止めるかを確かめた回数。 */
  private static int safePoints() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-count"));
    AtomicInteger calls = new AtomicInteger();
    assertEquals(world.train(world.options(() -> calls.incrementAndGet() < 0)), Ending.FINISHED);
    return calls.get();
  }

  @Test
  public void aStopRequestOrSignalAtEverySafePointContinuesToTheSameResult() throws Exception {
    FakeRun.Outcome expected = reference();
    int points = safePoints();
    // 反復ごとに 13 か所（段の前後で 6、段の中の待ちで 1、閉じた後で 1、ジョブの前後とチャンクで 4、ジョブの後で 1）。
    // ほかに、起動のときと、起動直後のジョブの後で 2 か所
    assertEquals(points, 2 + 2 * 13);
    for (boolean signal : List.of(false, true)) {
      for (int n = 1; n <= points; n++) {
        FakeRun world = FakeRun.create(Files.createTempDirectory("run-stop"));
        ControlRequests requests = new ControlRequests(world.run);
        int target = n;
        AtomicInteger calls = new AtomicInteger();
        if (!signal && n == 1) {
          requests.submit(new Request.Stop(Request.At.STEP));
        }
        Ending ending =
            world.train(
                world.options(
                    () -> {
                      int call = calls.incrementAndGet();
                      if (!signal && call == target - 1) {
                        try {
                          requests.submit(new Request.Stop(Request.At.STEP));
                        } catch (IOException failure) {
                          throw new IllegalStateException(failure);
                        }
                      }
                      return signal && call == target;
                    }));
        String where = (signal ? "signal" : "request") + " at safe point " + n;
        assertEquals(ending, Ending.STOPPED, where);
        RunStatus status = world.status();
        assertEquals(status.state(), State.STOPPED, where);
        assertTrue(requests.stopped(), where);
        assertEquals(requests.stoppedReason(), status.reason(), where);
        if (signal) {
          assertTrue(status.reason().endsWith(" by a signal"), status.reason());
          assertNull(status.obeyed(), where);
        } else {
          assertEquals(requests.pending(), List.of(), where);
          assertEquals(RetiredRequests.read(world.run).size(), 1, where);
          assertEquals(status.obeyed(), Integer.valueOf(1), where);
        }
        world.finish();
        assertEquals(world.outcome(), expected, where);
      }
    }
  }

  @Test
  public void aStopAtTheIterationWaitsForTheIterationAndItsMarkStays() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-stop-iteration"));
    ControlRequests requests = new ControlRequests(world.run);
    requests.submit(new Request.Stop(Request.At.ITERATION));
    assertEquals(world.train(), Ending.STOPPED);
    assertEquals(world.status().reason(), "Stopped at ITERATION by request 1");
    assertEquals(Files.readString(world.run.state(), StandardCharsets.UTF_8), "{\"next\":2,\"value\":32}\n");
    // ジョブの前で止まった（バージョン 2 のジョブはまだ回していない）
    assertEquals(world.outcome().evaluations(), List.of());
    // 停止マーカーは次の起動では消えない
    assertEquals(world.train(), Ending.FINISHED);
    assertTrue(requests.stopped());
    assertEquals(requests.stoppedReason(), "Stopped at ITERATION by request 1");
  }

  @Test
  public void aStopAtJobsWaitsForTheVersionsJobs() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-stop-jobs"));
    new ControlRequests(world.run).submit(new Request.Stop(Request.At.JOBS));
    assertEquals(world.train(), Ending.STOPPED);
    // 起動直後にジョブを回した後のセーフポイントで止まる
    assertEquals(world.status().reason(), "Stopped at JOBS by request 1");
    assertEquals(world.outcome().iterations(), List.of());
  }

  @Test
  public void aCancelledStopRequestDoesNotStopTheRun() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-cancel"));
    ControlRequests requests = new ControlRequests(world.run);
    ControlRequest stop = requests.submit(new Request.Stop(Request.At.ITERATION));
    requests.submit(new Request.Cancel(stop.number()));
    assertEquals(world.train(), Ending.FINISHED);
    assertEquals(requests.pending(), List.of());
    assertEquals(
        RetiredRequests.read(world.run).stream().map(ControlRequest::fileName).toList(),
        List.of("1-stop.json", "2-cancel.json"));
    assertFalse(requests.stopped());
    assertEquals(world.outcome(), reference());
  }

  /** GRP の学び直しの頼みには段と仕事を回す仕組みは従わず、取り消しの頼みが来れば次の安全点で消す。 */
  @Test
  public void aGrpRefitRequestStaysUntilACancelRemovesIt() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-grp-refit"));
    ControlRequests requests = new ControlRequests(world.run);
    ControlRequest refit = requests.submit(new Request.GrpRefit(1));
    assertEquals(world.train(), Ending.FINISHED);
    assertEquals(requests.pending(), List.of(refit));
    assertEquals(world.status().requests(), List.of(refit.number()));
    requests.submit(new Request.Cancel(refit.number()));
    assertEquals(world.train(), Ending.FINISHED);
    assertEquals(requests.pending(), List.of());
    assertEquals(
        RetiredRequests.read(world.run).stream().map(ControlRequest::fileName).toList(),
        List.of("1-grp-refit.json", "2-cancel.json"));
    assertEquals(world.outcome(), reference());
  }

  @Test
  public void aRestartRequestRestartsAtTheIterationAndContinues() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-restart"));
    ControlRequests requests = new ControlRequests(world.run);
    requests.submit(new Request.Restart(Request.At.ITERATION));
    assertEquals(world.train(), Ending.RESTART);
    RunStatus status = world.status();
    assertEquals(status.state(), State.RESTARTING);
    assertEquals(status.obeyed(), Integer.valueOf(1));
    assertEquals(status.reason(), "Restarting at ITERATION by request 1");
    assertFalse(requests.stopped());
    assertEquals(world.train(), Ending.FINISHED);
    assertEquals(world.outcome(), reference());
  }

  @Test
  public void theLimitOfACallStopsAfterThatManyIterations() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-limit"));
    assertEquals(world.train(new Runner.Options(FakeRun.JAR, 1, null, () -> false)), Ending.LIMIT);
    assertEquals(world.outcome().iterations().size(), 1);
    assertEquals(world.status().reason(), "Closed 1 iterations, the limit of this call");
    assertEquals(world.train(), Ending.FINISHED);
    assertEquals(world.outcome(), reference());
  }

  @Test
  public void jobsCreatedForPublishedVersionsRunOldestFirstBeforeTheNextIteration() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-created-jobs"), "iterations=3");
    Jobs jobs = new Jobs(world.run.jobs(), world.run.report(RunDirectory.EVALUATIONS));
    // 反復 2 の途中で、公開した過去のバージョンのジョブを新しい順に作る（jobs create と同じ）
    world.stepHooks.put(
        "2:PHI",
        () -> {
          for (int version : List.of(2, 1)) {
            Node.Table definition = RecordFile.tree(new FakeRun.JobDef(version, "late", 3, 2));
            assertTrue(jobs.create(JobFile.pending(version, "late", 0, 1, definition)));
          }
        });
    world.stepHooks.put("3:PHI", () -> world.trail.add("iteration 3"));
    assertEquals(world.train(), Ending.FINISHED);
    // 反復 2 を閉じた後、次の反復の前に、作ったジョブと公開したバージョン 3 のジョブをバージョンの古い順に回した
    assertEquals(
        world.trail,
        List.of("v00002-x1n3", "v00001-late", "v00002-late", "v00003-x1n3", "iteration 3", "v00004-x1n3"));
    assertEquals(world.outcome().jobs().get("v00001-late"), JobFile.State.DONE);
    assertEquals(world.outcome().jobs().get("v00002-late"), JobFile.State.DONE);
  }

  @Test
  public void aRowAddedDuringAnIterationTakesEffectFromTheNextIteration() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-rows"));
    List<SettingsRow> added = new ArrayList<>();
    world.stepHooks.put(
        "1:PHI",
        () -> {
          try (SettingsRows.Locked locked = new SettingsRows(world.run).lock()) {
            SettingsRow last = locked.latest();
            Node.Table changed = FakeRun.tree("factor=5", "batch=128");
            added.add(
                locked.append(
                    locked.firstOpenIteration(),
                    FakeRun.JAR,
                    "more",
                    List.of(
                        new SettingsRow.ChangedKey(
                            "batch", last.settings().get("batch"), changed.get("batch"), Change.MACHINE)),
                    changed));
          }
        });
    assertEquals(world.train(), Ending.FINISHED);
    assertEquals(added.getFirst().row(), 2);
    assertEquals(added.getFirst().from(), 2);
    FakeRun.Outcome outcome = world.outcome();
    assertTrue(outcome.iterations().get(0).startsWith("1 v2 row 1 "), outcome.iterations().get(0));
    assertTrue(outcome.iterations().get(1).startsWith("2 v3 row 2 "), outcome.iterations().get(1));
    // 反復 1 は設定の行 1（factor 3）で、反復 2 は行 2（factor 5）で集めた
    assertEquals(outcome.extra().get(0), "{\"iteration\":1,\"value\":32,\"closed\":1}");
    assertEquals(outcome.state(), "{\"next\":3,\"value\":178}\n");
    // 反復 1 が公開するバージョン 2 のジョブは行 1 のままで、行 2 はバージョン 3 のジョブから効く
    Jobs jobs = new Jobs(world.run.jobs(), world.run.report(RunDirectory.EVALUATIONS));
    assertEquals(jobs.read("v00002-x1n3").row(), 1);
    assertEquals(jobs.read("v00003-x1n3").row(), 2);
    // 速さだけに効くキー（batch）は起動のときの行の値のままで、起動し直すと効く
    assertEquals(world.status().pendingRestart(), List.of("batch"));
    assertEquals(world.launch.row(), 1);
  }

  @Test
  public void aJarThatAddsKeysAddsARowWithTheirDefaultsBeforeFixingTheIteration() throws Exception {
    FakeRun world = new FakeRun(Files.createTempDirectory("run-added"));
    try (SettingsRows.Locked locked = new SettingsRows(world.run).lock()) {
      locked.append(1, "old00001", SettingsRow.CREATED, List.of(), FakeRun.tree().without("chunk"));
    }
    assertEquals(world.train(), Ending.FINISHED);
    List<SettingsRow> rows = new SettingsRows(world.run).read();
    assertEquals(rows.size(), 2);
    SettingsRow row = rows.get(1);
    assertEquals(row.from(), 1);
    assertEquals(row.jar(), FakeRun.JAR);
    assertEquals(row.reason(), "Jar test0001 adds chunk");
    assertEquals(row.changes().size(), 1);
    assertEquals(row.changes().getFirst().key(), "chunk");
    assertNull(row.changes().getFirst().before());
    assertEquals(row.changes().getFirst().change(), Change.SCHEDULE);
    assertEquals(row.settings(), FakeRun.tree());
    assertTrue(world.outcome().iterations().getFirst().startsWith("1 v2 row 2 "));
  }

  @Test
  public void aBrokenSettingsRowRefusesTheRunWithFailedAndTheReason() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-broken"));
    Files.writeString(
        world.run.settings(), "{\"row\": 2}\n", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
    expectThrows(SettingsException.class, () -> new SettingsRows(world.run).read());
    assertEquals(world.train(), Ending.REFUSED);
    RunStatus status = world.status();
    assertEquals(status.state(), State.FAILED);
    assertTrue(status.reason().contains("settings.jsonl:2"), status.reason());
    assertEquals(world.opens, 0);
  }

  @Test
  public void settingsRowsMustEndWithANewlineAndCountFromOne() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-rows-order"));
    String first = Files.readString(world.run.settings(), StandardCharsets.UTF_8);
    Files.writeString(world.run.settings(), first.strip(), StandardCharsets.UTF_8);
    SettingsException unterminated = expectThrows(SettingsException.class, () -> new SettingsRows(world.run).read());
    assertTrue(unterminated.problems().getFirst().endsWith("does not end with a newline"));
    Files.writeString(world.run.settings(), first + first, StandardCharsets.UTF_8);
    SettingsException doubled = expectThrows(SettingsException.class, () -> new SettingsRows(world.run).read());
    assertTrue(doubled.problems().getFirst().startsWith("Settings row 1 at "), doubled.problems().getFirst());
    assertTrue(doubled.problems().getFirst().endsWith("is out of order (expected 2)"));
    Files.delete(world.run.settings());
    assertEquals(new SettingsRows(world.run).read(), List.of());
    assertEquals(world.train(), Ending.REFUSED);
    assertTrue(world.status().reason().startsWith("No settings rows in "), world.status().reason());
  }

  @Test
  public void aWriteThatFailsHalfwayLeavesTheFormerFile() throws Exception {
    Path directory = Files.createTempDirectory("durable");
    Path file = directory.resolve("settings.jsonl");
    DurableFiles.write(file, "row 1\n");
    expectThrows(
        IOException.class,
        () ->
            DurableFiles.write(
                file,
                stream -> {
                  stream.write("row 1\nrow 2 half".getBytes(StandardCharsets.UTF_8));
                  throw new IOException("disk full");
                }));
    assertEquals(Files.readString(file, StandardCharsets.UTF_8), "row 1\n");
    try (Stream<Path> listed = Files.list(directory)) {
      assertEquals(listed.map(path -> path.getFileName().toString()).toList(), List.of("settings.jsonl"));
    }
  }

  @Test
  public void anIterationWithAJarBoundStepDoneRefusesAnotherJar() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-jar-bound"));
    world.fail(Point.AFTER_SEAL, 1, Steps.COLLECT);
    expectThrows(FakeRun.Crash.class, world::train);
    Runner.Options other = new Runner.Options("other002", null, null, () -> false);
    assertEquals(world.train(other), Ending.REFUSED);
    RunStatus status = world.status();
    assertEquals(status.state(), State.FAILED);
    assertEquals(status.reason(), "Iteration 1 started with jar test0001; restore that jar to finish it");
    assertEquals(status.jar(), "other002");
    assertEquals(world.train(), Ending.FINISHED);
    assertEquals(world.outcome(), reference());
  }

  @Test
  public void anIterationWithOnlyTheJarFreeStepDoneIsFixedAgainWithTheNewJar() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-jar-free"));
    world.fail(Point.AFTER_SEAL, 1, Steps.PHI);
    expectThrows(FakeRun.Crash.class, world::train);
    Runner.Options other = new Runner.Options("other002", null, null, () -> false);
    assertEquals(world.train(other), Ending.FINISHED);
    assertEquals(world.ran(1, Steps.PHI), 1);
    FakeRun.Outcome outcome = world.outcome();
    // PHI の報告は、その段を回した jar のまま。後の段は新しい jar で回した
    assertTrue(outcome.iterations().getFirst().contains("PHI=DONE/1/test0001/"), outcome.iterations().getFirst());
    assertTrue(outcome.iterations().getFirst().contains("COLLECT=DONE/1/other002/"), outcome.iterations().getFirst());
  }

  @Test
  public void theHeartbeatAdvancesWhileAStepRuns() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-heartbeat"));
    List<RunStatus> seen = new ArrayList<>();
    world.stepHooks.put(
        "1:PHI",
        () -> {
          seen.add(world.status());
          Thread.sleep(300);
          seen.add(world.status());
        });
    assertEquals(world.train(), Ending.FINISHED);
    RunStatus before = seen.get(0);
    RunStatus after = seen.get(1);
    assertEquals(before.state(), State.RUNNING);
    assertEquals(before.step(), "PHI");
    assertEquals(before.iteration(), Integer.valueOf(1));
    assertEquals(before.settingsRow(), Integer.valueOf(1));
    assertEquals(before.jar(), FakeRun.JAR);
    assertTrue(after.updated().isAfter(before.updated()), before.updated() + " " + after.updated());
    assertEquals(after.since(), before.since());
    assertFalse(after.unclean(after.updated()));
  }

  @Test
  public void aLiveStatusIsUncleanWhenItsHeartbeatIsOldOrItsProcessIsGone() throws Exception {
    RunStatus fresh = world().status();
    Instant now = fresh.updated();
    assertFalse(fresh.unclean(now));
    assertTrue(fresh.unclean(now.plus(RunStatus.STALE).plusSeconds(1)));
    RunStatus reused = withPid(fresh, fresh.pid(), fresh.pidStart() + 1, State.RUNNING);
    assertTrue(reused.unclean(now));
    RunStatus finished = withPid(fresh, fresh.pid(), fresh.pidStart() + 1, State.FINISHED);
    assertFalse(finished.unclean(now.plusSeconds(3600)));
  }

  private static FakeRun world() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-unclean"));
    world.stepHooks.put("1:PHI", () -> Files.copy(world.run.status(), world.run.root().resolve("live.json")));
    world.train();
    Files.move(
        world.run.root().resolve("live.json"), world.run.status(), StandardCopyOption.REPLACE_EXISTING);
    return world;
  }

  private static RunStatus withPid(RunStatus status, long pid, Long start, State state) {
    return new RunStatus(
        state, pid, start, status.jar(), status.iteration(), status.step(), status.job(), status.since(),
        status.updated(), status.settingsRow(), status.launchRow(), status.pendingRestart(), status.requests(),
        status.obeyed(), status.reason());
  }

  @Test
  public void aSecondRunnerInTheSameJvmCannotTakeTheTrainLock() throws Exception {
    FakeRun world = FakeRun.create(Files.createTempDirectory("run-lock"));
    try (TrainLock held = TrainLock.hold(world.run.trainLock())) {
      IOException refused = expectThrows(IOException.class, world::train);
      assertEquals(refused.getMessage(), "This JVM already holds " + world.run.trainLock());
      assertTrue(TrainLock.running(world.run.trainLock()));
    }
    assertFalse(TrainLock.running(world.run.trainLock()));
    assertEquals(world.train(), Ending.FINISHED);
  }
}
