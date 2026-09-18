package com.epsilon.ai.decision.duel;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionDuelArenaSettings;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.DecisionRequest;
import com.google.gson.Gson;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 移行前の固定牌山結果と、非同期推論の容量・借用寿命を検証する。 */
public class EpsilonDecisionDuelArenaTest {
  @DataProvider
  public Object[][] concurrency() {
    return new Object[][] {{1, 1}, {3, 8}};
  }

  @Test(dataProvider = "concurrency", timeOut = 30000)
  public void matchesPreMigrationSeatAndWallOutcomes(int workers, int gamesInFlight)
      throws Exception {
    Baseline expected = baseline();
    try (TestPolicy candidate = new TestPolicy(1);
        TestPolicy opponent = new TestPolicy(0)) {
      var result = evaluate(candidate, opponent, workers, gamesInFlight);
      Assert.assertEquals(result.rotationOutcomes(), expected.rotations());
      Assert.assertEquals(result.wallOutcomes(), expected.walls());
      Assert.assertEquals(result.result().games(), 8);
      Assert.assertEquals(result.result().wallSeeds(), 2);
      Assert.assertEquals(result.result().pairedRankDeltaMean(), -2.0 / 3.0, 1e-12);
      Assert.assertEquals(result.result().pairedRankDeltaSe(), 1.0 / 3.0, 1e-12);
      Assert.assertEquals(result.metrics().completedGames(), 8);
      Assert.assertEquals(candidate.active, 0);
      Assert.assertEquals(opponent.active, 0);
      Assert.assertFalse(candidate.closed || opponent.closed);
      if (gamesInFlight > 1) Assert.assertTrue(candidate.waits + opponent.waits > 0);
    }
  }

  @Test(timeOut = 30000)
  public void streamingSegmentsUseTheSameWallsWithoutOverlap() throws Exception {
    var actual = new ArrayList<DuelEvaluation.WallOutcome>();
    try (TestPolicy candidate = new TestPolicy(1);
        TestPolicy opponent = new TestPolicy(0)) {
      for (int firstWall = 3; firstWall <= 4; firstWall++) {
        var result =
            EpsilonDecisionDuelArena.evaluateDuelStreaming(
                Path.of("candidate"),
                Path.of("opponent"),
                candidate,
                opponent,
                4,
                91826L,
                firstWall,
                4,
                actual::add,
                EpsilonUtilityProfile.TENHOU,
                new DecisionDuelArenaSettings(2, 2, 4),
                new InferenceBatchingSettings(0));
        Assert.assertEquals(result.result().games(), 4);
        Assert.assertEquals(result.result().wallSeeds(), 1);
      }
    }
    actual.sort(Comparator.comparingLong(DuelEvaluation.WallOutcome::wallIndex));
    Assert.assertEquals(actual, baseline().walls());
  }

  @DataProvider
  public Object[][] failures() {
    return new Object[][] {{true}, {false}};
  }

  @Test(dataProvider = "failures", timeOut = 15000)
  public void failureOrInterruptDrainsOtherBorrowedInputs(boolean interrupt) throws Exception {
    ControlledPolicy candidate = new ControlledPolicy();
    ControlledPolicy opponent = new ControlledPolicy();
    AtomicReference<Throwable> failure = new AtomicReference<>();
    Thread runner =
        Thread.ofPlatform()
            .start(
                () -> {
                  try {
                    evaluate(candidate, opponent, 2, 4);
                  } catch (Throwable problem) {
                    failure.set(problem);
                  }
                });
    try {
      Assert.assertTrue(candidate.started.await(5, TimeUnit.SECONDS));
      Assert.assertTrue(opponent.started.await(5, TimeUnit.SECONDS));
      int turn = opponent.requests.getFirst().observation().turnNumber();
      if (interrupt) runner.interrupt();
      else candidate.result.completeExceptionally(new IllegalStateException("deliberate failure"));
      runner.join(50);
      Assert.assertTrue(runner.isAlive(), "Outstanding inference must be drained");
      Assert.assertEquals(opponent.requests.getFirst().observation().turnNumber(), turn);
      candidate.result.complete(new int[candidate.requests.size()]);
      opponent.result.complete(new int[opponent.requests.size()]);
      runner.join(5000);
      Assert.assertFalse(runner.isAlive());
      Assert.assertNotNull(failure.get());
      Assert.assertFalse(candidate.closed || opponent.closed);
      if (interrupt) Assert.assertTrue(runner.isInterrupted());
      else Assert.assertEquals(failure.get().getMessage(), "deliberate failure");
    } finally {
      candidate.result.complete(new int[candidate.requests.size()]);
      opponent.result.complete(new int[opponent.requests.size()]);
      runner.interrupt();
      runner.join(5000);
    }
  }

  private static EpsilonDecisionDuelArena.Evaluation evaluate(
      BatchedPolicy candidate, BatchedPolicy opponent, int workers, int gamesInFlight) {
    return EpsilonDecisionDuelArena.evaluateDuel(
        Path.of("candidate"),
        Path.of("opponent"),
        candidate,
        opponent,
        8,
        91826L,
        3L,
        gamesInFlight,
        EpsilonUtilityProfile.TENHOU,
        new DecisionDuelArenaSettings(workers, 2, 4),
        new InferenceBatchingSettings(0));
  }

  private static Baseline baseline() throws Exception {
    try (var reader =
        new InputStreamReader(
            EpsilonDecisionDuelArenaTest.class.getResourceAsStream("duel-before-common.json"),
            StandardCharsets.UTF_8)) {
      return new Gson().fromJson(reader, Baseline.class);
    }
  }

  private record Baseline(
      List<EpsilonDecisionDuelArena.RotationOutcome> rotations,
      List<DuelEvaluation.WallOutcome> walls) {}

  private static final class ControlledPolicy implements BatchedPolicy {
    final CountDownLatch started = new CountDownLatch(1);
    final CompletableFuture<int[]> result = new CompletableFuture<>();
    List<DecisionRequest> requests = List.of();
    boolean reserved, closed;

    public Object batchKey(DecisionRequest request) {
      return 0;
    }

    public int maxBatchSize(Object key) {
      return 16;
    }

    public Ingress tryAcquire(Object key, Runnable wakeup) {
      if (reserved) return null;
      reserved = true;
      return new Ingress() {
        public CompletableFuture<int[]> submit(List<DecisionRequest> rows) {
          requests = rows;
          started.countDown();
          return result;
        }

        public void close() {}
      };
    }

    public void close() {
      closed = true;
    }
  }

  private static final class TestPolicy implements BatchedPolicy {
    final int slot;
    final ScheduledExecutorService executor = Executors.newScheduledThreadPool(2);
    final List<Runnable> waiters = new ArrayList<>();
    int active, sequence, waits;
    boolean closed;

    TestPolicy(int slot) {
      this.slot = slot;
    }

    public Object batchKey(DecisionRequest request) {
      Assert.assertTrue(request.legalActions().size() > 1, "Forced actions need no inference");
      return request.legalActions().size() % 3;
    }

    public int maxBatchSize(Object key) {
      return 4;
    }

    public synchronized Ingress tryAcquire(Object key, Runnable wakeup) {
      if (active == 1) {
        waits++;
        waiters.add(wakeup);
        return null;
      }
      active++;
      return new Ingress() {
        boolean consumed;

        public CompletableFuture<int[]> submit(List<DecisionRequest> requests) {
          Assert.assertFalse(consumed);
          consumed = true;
          var result = new CompletableFuture<int[]>();
          executor.schedule(
              () -> {
                try {
                  int[] selected = new int[requests.size()];
                  for (int row = 0; row < selected.length; row++)
                    selected[row] = Math.min(slot, requests.get(row).legalActions().size() - 1);
                  release();
                  result.complete(selected);
                } catch (Throwable failure) {
                  result.completeExceptionally(failure);
                }
              },
              sequence++ % 2,
              TimeUnit.MILLISECONDS);
          return result;
        }

        public void close() {
          if (!consumed) {
            consumed = true;
            release();
          }
        }
      };
    }

    private synchronized void release() {
      active--;
      List<Runnable> notify = new ArrayList<>(waiters);
      waiters.clear();
      for (Runnable waiter : notify) waiter.run();
    }

    public void close() {
      closed = true;
      executor.close();
    }
  }
}
