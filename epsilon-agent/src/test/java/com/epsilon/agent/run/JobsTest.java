package com.epsilon.agent.run;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.run.FakeRun.Point;
import com.epsilon.agent.run.RunStatus.State;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import com.epsilon.agent.settings.SettingsException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.OptionalLong;
import org.testng.annotations.Test;

/**
 * ジョブの扱い。ジョブは何度作っても最初の記録のままで、計画は 1 回だけ決まる。チャンクごとに止めて続けても、結果は変わらない。
 * games.tsv を書いた後、job.json を書く前に落ちても、行は二重にならない。GPU を譲った試行と結果を書いた試行は、落ちた回数に入らない。
 * 2 回落ちたジョブは FAILED になり、やり直すと最後まで終わる。
 */
public class JobsTest {

  /** 反復が 1 回の run。8 牌山を 2 牌山ずつ打つジョブを 1 つ作る。 */
  private static FakeRun world(String label) throws Exception {
    return FakeRun.create(Files.createTempDirectory(label), "iterations=1", "walls=8");
  }

  private static final String JOB = "v00002-x1n3";

  private static FakeRun.Outcome reference() throws Exception {
    FakeRun world = world("jobs-reference");
    assertEquals(world.finish(), 1);
    return world.outcome();
  }

  private static JobFile job(FakeRun world) throws Exception {
    return new Jobs(world.run.jobs(), world.run.report(RunDirectory.EVALUATIONS)).read(JOB);
  }

  @Test
  public void aJobPlansOnceAndItsResultCountsEveryWallOnce() throws Exception {
    FakeRun world = world("jobs-plain");
    assertEquals(world.train(), Ending.FINISHED);
    JobFile job = job(world);
    assertEquals(job.state(), JobFile.State.DONE);
    assertEquals(job.walls(), 8);
    assertEquals(job.attempts(), 1);
    assertEquals(job.cleanStops(), 0);
    assertEquals(job.failures(), 0);
    assertEquals(world.plans, 1);
    assertEquals(world.settles, 1);
    FakeRun.Outcome outcome = world.outcome();
    assertEquals(outcome.evaluations().size(), 1);
    assertEquals(FakeRun.field(outcome.evaluations().getFirst(), "games"), new Node.Numeral("8", "expected"));
    assertEquals(DurableFiles.lines(world.run.job(JOB).resolve(JobOutputs.GAMES)).size(), 8);
    assertEquals(
        Files.readString(world.run.job(JOB).resolve(Jobs.RESULT), StandardCharsets.UTF_8),
        outcome.evaluations().getFirst() + "\n");
  }

  @Test
  public void creatingAJobTwiceKeepsTheFirstRecord() throws Exception {
    Path directory = Files.createTempDirectory("jobs-create");
    Jobs jobs = new Jobs(directory, directory.resolve("evaluations.jsonl"));
    JobFile first = JobFile.pending(2, "x1n3", 0, 1, RecordFile.tree(new FakeRun.JobDef(2, "x1n3", 8, 2)));
    assertTrue(jobs.create(first));
    String written = Files.readString(directory.resolve(JOB).resolve(Jobs.JOB), StandardCharsets.UTF_8);
    JobFile again = JobFile.pending(2, "x1n3", 0, 3, RecordFile.tree(new FakeRun.JobDef(2, "x1n3", 9, 3)));
    assertFalse(jobs.create(again));
    assertEquals(Files.readString(directory.resolve(JOB).resolve(Jobs.JOB), StandardCharsets.UTF_8), written);
    assertEquals(jobs.list().jobs(), List.of(first));
  }

  /** 落ちた回数に入るのは、落ちた試行だけ。結果を書いた試行と、まだ動いている試行は入らない。 */
  @Test
  public void onlyFallenAttemptsAreFailures() {
    JobFile running = JobFile.pending(2, "x1n3", 0, 1, RecordFile.tree(new FakeRun.JobDef(2, "x1n3", 8, 2))).attempted();
    assertEquals(running.failures(), 0);
    assertEquals(running.finished().failures(), 0);
    assertEquals(running.stoppedCleanly().failures(), 0);
    JobFile fallen = running.fell("boom");
    assertEquals(fallen.failures(), 1);
    assertEquals(fallen.attempted().failures(), 1);
    assertEquals(fallen.attempted().finished().failures(), 1);
    assertEquals(fallen.attempted().fell("boom").failed().failures(), 2);
  }

  @Test
  public void stoppingAtEveryChunkAndContinuingGivesTheSameResultAndPlan() throws Exception {
    FakeRun.Outcome expected = reference();
    FakeRun world = world("jobs-chunks");
    ControlRequests requests = new ControlRequests(world.run);
    for (int chunk = 1; chunk <= 3; chunk++) {
      world.chunkHooks.put(chunk, () -> requests.submit(new Request.Stop(Request.At.STEP)));
    }
    for (int stop = 1; stop <= 3; stop++) {
      assertEquals(world.train(), Ending.STOPPED);
      JobFile job = job(world);
      assertEquals(job.state(), JobFile.State.PENDING);
      assertEquals(job.walls(), 2L * stop);
      assertEquals(job.cleanStops(), stop);
      assertEquals(world.status().reason(), "Stopped at CHUNK " + JOB + " by request " + stop);
    }
    assertEquals(world.train(), Ending.FINISHED);
    assertEquals(world.plans, 1);
    assertEquals(world.outcome(), expected);
    JobFile job = job(world);
    assertEquals(job.attempts(), 4);
    assertEquals(job.cleanStops(), 3);
  }

  @Test
  public void aCrashAfterTheGamesAndBeforeTheJobRecordDoesNotDoubleLines() throws Exception {
    FakeRun.Outcome expected = reference();
    for (int chunk = 1; chunk <= 4; chunk++) {
      FakeRun world = world("jobs-crash");
      world.fail(Point.CHUNK_CRASH, chunk);
      assertEquals(world.finish(), 2, "chunk " + chunk);
      assertEquals(world.outcome(), expected, "chunk " + chunk);
      JobFile job = job(world);
      // 落ちた試行と、終わった試行
      assertEquals(job.attempts(), 2, "chunk " + chunk);
      assertEquals(job.cleanStops(), 0, "chunk " + chunk);
      assertEquals(job.failures(), 1, "chunk " + chunk);
      assertEquals(job.state(), JobFile.State.DONE);
      assertEquals(world.plans, 1);
    }
  }

  @Test
  public void aCrashAfterTheResultSettlesAndReportsOnce() throws Exception {
    FakeRun.Outcome expected = reference();
    FakeRun world = world("jobs-settle");
    world.fail(Point.SETTLE, 2);
    assertEquals(world.finish(), 2);
    assertEquals(world.outcome(), expected);
    assertEquals(world.settles, 1);
    assertEquals(world.chunks, 4);
  }

  @Test
  public void yieldingTheGpusThreeTimesIsNotAFailure() throws Exception {
    FakeRun.Outcome expected = reference();
    FakeRun world = world("jobs-yield");
    Path requests = Files.createTempDirectory("gpu-requests");
    GpuRequests gpu = new GpuRequests(requests);
    long pid = ProcessHandle.current().pid();
    OptionalLong start = ProcessStart.of(pid);
    assertTrue(start.isPresent(), "/proc is needed for the GPU requests");
    for (int chunk = 1; chunk <= 3; chunk++) {
      world.chunkHooks.put(
          chunk, () -> gpu.write(new GpuRequests.Request("probe", pid, start.getAsLong(), "gpu.sh hold probe")));
    }
    Runner.Options options = new Runner.Options(FakeRun.JAR, null, gpu, () -> false);
    for (int yielded = 1; yielded <= 3; yielded++) {
      assertEquals(world.train(options), Ending.YIELD);
      RunStatus status = world.status();
      assertEquals(status.state(), State.YIELDED);
      assertEquals(
          status.reason(),
          "Yielding the GPUs at CHUNK " + JOB + " to probe (pid " + pid + ": gpu.sh hold probe)");
      JobFile job = job(world);
      assertEquals(job.state(), JobFile.State.PENDING);
      assertEquals(job.cleanStops(), yielded);
      assertEquals(job.failures(), 0);
      Files.delete(requests.resolve("probe"));
    }
    assertEquals(world.train(options), Ending.FINISHED);
    JobFile job = job(world);
    assertEquals(job.state(), JobFile.State.DONE);
    assertEquals(job.attempts(), 4);
    assertEquals(world.outcome(), expected);
  }

  @Test
  public void twoCrashesFailTheJobAndRetryFinishesIt() throws Exception {
    FakeRun.Outcome expected = reference();
    FakeRun world = world("jobs-failed");
    world.fail(Point.CHUNK_CRASH, 1);
    world.fail(Point.CHUNK_CRASH, 3);
    expectThrows(FakeRun.Crash.class, world::train);
    expectThrows(FakeRun.Crash.class, world::train);
    JobFile crashed = job(world);
    assertEquals(crashed.state(), JobFile.State.RUNNING);
    assertEquals(crashed.attempts(), 2);
    // 2 回目に落ちたことは、次の起動が記録するまで数に入らない
    assertEquals(crashed.failures(), 1);
    assertEquals(crashed.walls(), 2);
    // 3 回目の起動は、2 回落ちたジョブを始めずに FAILED にして先へ進む
    assertEquals(world.train(), Ending.FINISHED);
    JobFile failed = job(world);
    assertEquals(failed.state(), JobFile.State.FAILED);
    assertEquals(failed.attempts(), 2);
    assertEquals(failed.failures(), 2);
    assertEquals(failed.reason(), "Attempt 2 ended with its JVM");
    assertEquals(world.chunks, 3);
    assertEquals(world.outcome().evaluations(), List.of());
    Jobs jobs = new Jobs(world.run.jobs(), world.run.report(RunDirectory.EVALUATIONS));
    JobFile retried = jobs.retry(JOB);
    assertEquals(retried.state(), JobFile.State.PENDING);
    assertEquals(retried.attempts(), 0);
    assertNull(retried.reason());
    IllegalArgumentException notFailed = expectThrows(IllegalArgumentException.class, () -> jobs.retry(JOB));
    assertEquals(notFailed.getMessage(), "Job " + JOB + " is PENDING, not FAILED");
    assertEquals(world.runner(world.options(() -> false)).runJobs(), Ending.FINISHED);
    assertEquals(world.outcome(), expected);
    assertEquals(world.plans, 1);
  }

  @Test
  public void twoFailuresInOneLaunchFailTheJobWithTheLastReason() throws Exception {
    FakeRun world = world("jobs-fell");
    world.fail(Point.CHUNK_FAIL, 2);
    world.fail(Point.CHUNK_FAIL, 3);
    assertEquals(world.train(), Ending.FINISHED);
    JobFile job = job(world);
    assertEquals(job.state(), JobFile.State.FAILED);
    assertEquals(job.attempts(), 2);
    assertEquals(job.cleanStops(), 0);
    // 1 回目の試行は 2 牌山を記録してから落ち、2 回目の試行はその続きで落ちた
    assertEquals(job.walls(), 2);
    assertEquals(job.reason(), "IllegalStateException: boom at CHUNK_FAIL 3");
    assertEquals(DurableFiles.lines(world.run.job(JOB).resolve(JobOutputs.GAMES)).size(), 4);
  }

  /**
   * チャンクの後のセーフポイントの確かめが失敗したら（読めない要求のファイル）、次の試行を始めずにその例外で止まる。その試行は落ちた
   * 回数に入らず、ファイルを除いた次の起動でジョブは続きから最後まで回る。
   */
  @Test
  public void aFailingSafePointCheckStopsTheJobsWithoutCountingAsAFailure() throws Exception {
    FakeRun.Outcome expected = reference();
    FakeRun world = world("jobs-check");
    Path broken = world.run.control().resolve("9-stop.json");
    world.chunkHooks.put(
        1,
        () -> {
          Files.createDirectories(world.run.control());
          Files.writeString(broken, "not a request\n");
        });
    expectThrows(SettingsException.class, world::train);
    JobFile job = job(world);
    assertEquals(job.state(), JobFile.State.PENDING);
    assertEquals(job.walls(), 2);
    assertEquals(job.attempts(), 1);
    assertEquals(job.failures(), 0);
    assertEquals(world.chunks, 1);
    Files.delete(broken);
    assertEquals(world.train(), Ending.FINISHED);
    assertEquals(job(world).state(), JobFile.State.DONE);
    assertEquals(world.outcome(), expected);
  }

  /** 仕事を回す共有の資源の作り物。塊の途中で壊せて、壊れた後はどの仕事も開けない。 */
  private static final class FragileJobs implements JobRunner {
    private final JobRunner jobs;
    boolean breaksAtNextChunk;
    boolean broken;
    int opens;

    FragileJobs(JobRunner jobs) {
      this.jobs = jobs;
    }

    @Override
    public Node.Table plan(JobFile job) throws Exception {
      return jobs.plan(job);
    }

    @Override
    public JobWork open(JobFile job, Path directory) throws Exception {
      requireUsable();
      opens++;
      JobWork work = jobs.open(job, directory);
      return new JobWork() {
        @Override
        public Chunk next(long walls, JobOutputs outputs) throws Exception {
          if (breaksAtNextChunk) {
            breaksAtNextChunk = false;
            broken = true;
            throw new IllegalStateException("The device was lost");
          }
          return work.next(walls, outputs);
        }

        @Override
        public void close() throws Exception {
          work.close();
        }
      };
    }

    @Override
    public void requireUsable() {
      if (broken) {
        throw new IllegalStateException("The shared pipeline failed");
      }
    }
  }

  /**
   * 試みの途中で共有の資源が壊れたら、落ちた試みを 1 回と数え、次の試みもほかの仕事も始めずにその例外で止まる。打っていない仕事は
   * FAILED にならず、資源を作り直した次の起動で全ての仕事が済む。
   */
  @Test
  public void brokenSharedResourcesStopTheJobsWithoutFailingTheOnesThatNeverPlayed() throws Exception {
    Path directory = Files.createTempDirectory("jobs-broken");
    Jobs jobs = new Jobs(directory, directory.resolve("evaluations.jsonl"));
    List<String> tables = List.of("a", "b", "c");
    for (int order = 0; order < tables.size(); order++) {
      String table = tables.get(order);
      assertTrue(jobs.create(JobFile.pending(2, table, order, 1, RecordFile.tree(new FakeRun.JobDef(2, table, 4, 2)))));
    }
    FakeRun world = new FakeRun(Files.createTempDirectory("jobs-broken-run"));
    FragileJobs fragile = new FragileJobs(new FakeRun.FakeJobs(world));
    fragile.breaksAtNextChunk = true;
    RunControl control = RunControl.detached(() -> false);
    IllegalStateException stopped = null;
    try {
      jobs.runAll(fragile, control);
    } catch (IllegalStateException failure) {
      stopped = failure;
    }
    List<JobFile> after = jobs.list().jobs();
    assertEquals(
        after.stream().map(JobFile::state).toList(),
        List.of(JobFile.State.PENDING, JobFile.State.PENDING, JobFile.State.PENDING));
    assertEquals(after.stream().map(JobFile::failures).toList(), List.of(1, 0, 0));
    assertEquals(after.getFirst().reason(), "IllegalStateException: The device was lost");
    assertEquals(fragile.opens, 1);
    assertNotNull(stopped, "the jobs went on with broken resources");
    assertEquals(stopped.getMessage(), "The shared pipeline failed");

    fragile.broken = false;
    assertEquals(jobs.runAll(fragile, control), Optional.empty());
    assertEquals(
        jobs.list().jobs().stream().map(JobFile::state).toList(),
        List.of(JobFile.State.DONE, JobFile.State.DONE, JobFile.State.DONE));
    assertEquals(DurableFiles.lines(directory.resolve("evaluations.jsonl")).size(), 3);
  }

  @Test
  public void aYieldWhileTheRunIsPausedStillYieldsAndThePauseHoldsAfterTheRestart() throws Exception {
    FakeRun world = world("jobs-pause");
    ControlRequests requests = new ControlRequests(world.run);
    requests.submit(new Request.Pause(null));
    Path directory = Files.createTempDirectory("gpu-requests");
    GpuRequests gpu = new GpuRequests(directory);
    Runner.Options options = new Runner.Options(FakeRun.JAR, null, gpu, () -> false);
    long pid = ProcessHandle.current().pid();
    long start = ProcessStart.of(pid).orElseThrow();
    Ending[] ending = new Ending[1];
    Thread first = run(world, options, ending);
    awaitState(world, State.PAUSED);
    assertEquals(world.status().reason(), "Paused at LAUNCH by request 1");
    gpu.write(new GpuRequests.Request("probe", pid, start, "hold"));
    first.join(10_000);
    assertEquals(ending[0], Ending.YIELD);
    assertEquals(world.status().state(), State.YIELDED);
    assertEquals(world.opens, 0);
    Files.delete(directory.resolve("probe"));
    Thread second = run(world, options, ending);
    awaitState(world, State.PAUSED);
    assertEquals(requests.pending().size(), 1);
    requests.submit(new Request.Resume());
    second.join(10_000);
    assertEquals(ending[0], Ending.FINISHED);
    assertEquals(requests.pending(), List.of());
    assertEquals(
        RetiredRequests.read(world.run).stream().map(ControlRequest::fileName).toList(),
        List.of("1-pause.json", "2-resume.json"));
    assertEquals(world.outcome(), reference());
  }

  @Test
  public void aPauseBeforeAStepWaitsOnlyThere() throws Exception {
    FakeRun world = world("jobs-pause-step");
    ControlRequests requests = new ControlRequests(world.run);
    requests.submit(new Request.Pause("COLLECT"));
    Ending[] ending = new Ending[1];
    Thread thread = run(world, world.options(() -> false), ending);
    awaitState(world, State.PAUSED);
    RunStatus paused = world.status();
    assertEquals(paused.reason(), "Paused at BEFORE_STEP COLLECT by request 1");
    assertEquals(paused.obeyed(), Integer.valueOf(1));
    assertEquals(world.ran(1, FakeRun.Steps.PHI), 1);
    assertEquals(world.ran(1, FakeRun.Steps.COLLECT), 0);
    requests.submit(new Request.Resume());
    thread.join(10_000);
    assertEquals(ending[0], Ending.FINISHED);
    assertEquals(world.status().state(), State.FINISHED);
  }

  @Test
  public void deadAndReusedPidsDoNotTakeTheGpus() throws Exception {
    FakeRun world = world("jobs-dead");
    Path directory = Files.createTempDirectory("gpu-requests");
    GpuRequests gpu = new GpuRequests(directory);
    long pid = ProcessHandle.current().pid();
    long start = ProcessStart.of(pid).orElseThrow();
    Process gone = new ProcessBuilder("true").start();
    gone.waitFor();
    gpu.write(new GpuRequests.Request("dead", gone.pid(), start, "dead"));
    gpu.write(new GpuRequests.Request("reused", pid, start + 1, "reused"));
    Files.writeString(directory.resolve("broken"), "not a request\n");
    Files.writeString(directory.resolve(".hidden"), pid + " " + start + " hidden\n");
    assertEquals(gpu.alive(), List.of());
    assertEquals(world.train(new Runner.Options(FakeRun.JAR, null, gpu, () -> false)), Ending.FINISHED);
    gpu.write(new GpuRequests.Request("alive", pid, start, "probe"));
    assertEquals(gpu.alive().stream().map(GpuRequests.Request::name).toList(), List.of("alive"));
  }

  private static Thread run(FakeRun world, Runner.Options options, Ending[] ending) {
    Thread thread =
        new Thread(
            () -> {
              try {
                ending[0] = world.train(options);
              } catch (Exception failure) {
                throw new IllegalStateException(failure);
              }
            });
    thread.start();
    return thread;
  }

  private static void awaitState(FakeRun world, State state) throws Exception {
    long deadline = System.nanoTime() + 10_000_000_000L;
    while (System.nanoTime() < deadline) {
      if (Files.exists(world.run.status()) && world.status().state() == state) {
        return;
      }
      Thread.sleep(5);
    }
    throw new AssertionError("The run did not reach " + state);
  }
}
