package com.epsilon.arena;

import com.epsilon.core.Action;
import com.epsilon.runtime.MeasurementStatus;
import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.DecisionRequest;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

public class ArenaRunnerTest {
  @Test(timeOut = 30000)
  public void partialBatchesFinishReplacementGamesAfterInputProductionDrains() throws Exception {
    // バッチごとのタイマー待ちを挟まず、非同期推論の完了と後続対局への切り替えを検証する。
    try (TestPolicy policy = new TestPolicy(4, 2, false, false)) {
      ArenaResult result =
          new ArenaRunner(four(policy), new RunSettings(8, 2, 2, 213L, 1_000_000)).run();
      for (ArenaResult.PlayerResult player : result.players()) {
        Assert.assertEquals(player.first() + player.second() + player.third() + player.fourth(), 8);
      }
      Assert.assertTrue(policy.rows.get() > 0);
      Assert.assertEquals(policy.active, 0);
      Assert.assertTrue(result.inference().getFirst().batchFillRatio() < 1.0);
    }
  }

  @DataProvider
  public Object[][] inputShapes() {
    return new Object[][] {{4, 2, false}, {19, 7, true}};
  }

  @Test(dataProvider = "inputShapes", timeOut = 30000)
  public void unrelatedInputsAndOutOfOrderBatchesFinishAllSeatRotations(
      int widthA, int bucketsA, boolean projectedInputA) throws Exception {
    // bの入力形式とバッチの分類を固定し、aの入力形式と分類だけを変更する。
    try (TestPolicy a = new TestPolicy(widthA, bucketsA, projectedInputA, true);
        TestPolicy b = new TestPolicy(11, 5, false, true)) {
      List<Participant> players =
          List.of(
              new Participant("a", a),
              new Participant("b", b),
              new Participant("a", a),
              new Participant("b", b));
      ArenaResult result = new ArenaRunner(players, new RunSettings(4, 4, 3, 213L, 0)).run();
      Assert.assertEquals(result.players().size(), 4);
      for (ArenaResult.PlayerResult player : result.players()) {
        Assert.assertEquals(player.first() + player.second() + player.third() + player.fourth(), 4);
      }
      // 同じ判断規則で各牌山の4通りの席替えを行うと、成績の差は厳密に相殺される。
      Assert.assertEquals(result.paired().rankAdvantage(), 0.0, 1e-12);
      Assert.assertEquals(result.paired().scoreAdvantage(), 0.0, 1e-9);
      Assert.assertEquals(result.inference().size(), 2);
      Assert.assertTrue(
          result.inference().stream()
              .allMatch(
                  value -> value.providerTimings().status() == MeasurementStatus.UNAVAILABLE));
      Assert.assertTrue(a.rows.get() > 0 && b.rows.get() > 0);
      Assert.assertTrue(a.capacityWaits.get() + b.capacityWaits.get() > 0);
      Assert.assertFalse(a.sawForced.get() || b.sawForced.get());
      Assert.assertEquals(a.active, 0);
      Assert.assertEquals(b.active, 0);
    }
  }

  @Test(timeOut = 10000)
  public void invalidModelOutputFailsBeforeAnyIllegalCommit() {
    BatchedPolicy broken = immediatePolicy(false);
    IllegalArgumentException failure =
        Assert.expectThrows(
            IllegalArgumentException.class,
            () -> new ArenaRunner(four(broken), new RunSettings(4, 4, 2, 1, 0)).run());
    Assert.assertTrue(failure.getMessage().contains("invalid legal action"));
  }

  @Test(timeOut = 10000)
  public void asynchronousFailureIsPropagatedWithoutFallingBackToAnAction() {
    BatchedPolicy broken = immediatePolicy(true);
    java.util.concurrent.CompletionException failure =
        Assert.expectThrows(
            java.util.concurrent.CompletionException.class,
            () -> new ArenaRunner(four(broken), new RunSettings(4, 4, 2, 1, 0)).run());
    Assert.assertTrue(failure.getCause().getMessage().contains("deliberate inference failure"));
  }

  @Test(timeOut = 10000)
  public void interruptWaitsForBorrowedBatchToFinishBeforeReturning() throws Exception {
    CountDownLatch submitted = new CountDownLatch(1);
    CompletableFuture<int[]> nativeBatch = new CompletableFuture<>();
    AtomicReference<List<DecisionRequest>> borrowed = new AtomicReference<>();
    AtomicReference<Throwable> exit = new AtomicReference<>();
    AtomicBoolean reserved = new AtomicBoolean();
    BatchedPolicy blocked =
        new BatchedPolicy() {
          public Object batchKey(DecisionRequest request) {
            return 1;
          }

          public int maxBatchSize(Object key) {
            return 16;
          }

          public Ingress tryAcquire(Object key, Runnable wakeup) {
            if (!reserved.compareAndSet(false, true)) return null;
            return new Ingress() {
              public CompletableFuture<int[]> submit(List<DecisionRequest> requests) {
                borrowed.set(requests);
                submitted.countDown();
                return nativeBatch;
              }

              public void close() {}
            };
          }

          public void close() {}
        };
    Thread runner =
        Thread.ofPlatform()
            .start(
                () -> {
                  try {
                    new ArenaRunner(four(blocked), new RunSettings(4, 4, 2, 1, 0)).run();
                  } catch (Throwable error) {
                    exit.set(error);
                  }
                });
    try {
      Assert.assertTrue(submitted.await(5, TimeUnit.SECONDS));
      List<DecisionRequest> requests = borrowed.get();
      int originalTurn = requests.getFirst().observation().turnNumber();
      runner.interrupt();
      runner.join(50);
      Assert.assertTrue(runner.isAlive(), "Native input must remain alive until batch completion");
      Assert.assertEquals(requests.getFirst().observation().turnNumber(), originalTurn);
      nativeBatch.complete(new int[requests.size()]);
      runner.join(5000);
      Assert.assertFalse(runner.isAlive());
      Assert.assertTrue(exit.get() instanceof InterruptedException);
    } finally {
      nativeBatch.complete(new int[borrowed.get() == null ? 0 : borrowed.get().size()]);
      runner.interrupt();
      runner.join(5000);
    }
  }

  private static List<Participant> four(BatchedPolicy policy) {
    Participant participant = new Participant("test", policy);
    return List.of(participant, participant, participant, participant);
  }

  @Test(timeOut = 10000)
  public void cancelledBatchStillDrainsAnotherOutstandingModel() throws Exception {
    ControlledPolicy a = new ControlledPolicy();
    ControlledPolicy b = new ControlledPolicy();
    AtomicReference<Throwable> exit = new AtomicReference<>();
    Thread runner =
        Thread.ofPlatform()
            .start(
                () -> {
                  try {
                    new ArenaRunner(
                            List.of(
                                new Participant("a", a),
                                new Participant("b", b),
                                new Participant("a", a),
                                new Participant("b", b)),
                            new RunSettings(4, 4, 2, 1, 0))
                        .run();
                  } catch (Throwable error) {
                    exit.set(error);
                  }
                });
    try {
      Assert.assertTrue(a.started.await(5, TimeUnit.SECONDS));
      Assert.assertTrue(b.started.await(5, TimeUnit.SECONDS));
      a.result.cancel(false);
      runner.join(50);
      Assert.assertTrue(
          runner.isAlive(), "Cancellation must not abandon another model's borrowed input");
      b.result.complete(new int[b.rows]);
      runner.join(5000);
      Assert.assertFalse(runner.isAlive());
      Assert.assertNotNull(exit.get());
    } finally {
      a.result.complete(new int[a.rows]);
      b.result.complete(new int[b.rows]);
      runner.interrupt();
      runner.join(5000);
    }
  }

  private static final class ControlledPolicy implements BatchedPolicy {
    final CountDownLatch started = new CountDownLatch(1);
    final CompletableFuture<int[]> result = new CompletableFuture<>();
    boolean reserved;
    int rows;

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
        public CompletableFuture<int[]> submit(List<DecisionRequest> requests) {
          rows = requests.size();
          started.countDown();
          return result;
        }

        public void close() {}
      };
    }

    public void close() {}
  }

  private static BatchedPolicy immediatePolicy(boolean fail) {
    return new BatchedPolicy() {
      public Object batchKey(DecisionRequest request) {
        return 0;
      }

      public int maxBatchSize(Object key) {
        return 16;
      }

      public Ingress tryAcquire(Object key, Runnable wakeup) {
        return new Ingress() {
          public CompletableFuture<int[]> submit(List<DecisionRequest> requests) {
            if (fail)
              return CompletableFuture.failedFuture(
                  new IllegalStateException("deliberate inference failure"));
            int[] invalid = new int[requests.size()];
            java.util.Arrays.fill(invalid, Integer.MAX_VALUE);
            return CompletableFuture.completedFuture(invalid);
          }

          public void close() {}
        };
      }

      public void close() {}
    };
  }

  /** 入力フィールド、テンソル配置、バッチの分類と計算処理を、この模擬推論の内部で管理する。 */
  private static final class TestPolicy implements BatchedPolicy {
    final int width, buckets;
    final boolean projectedInput, staggerCompletions;
    final ScheduledExecutorService executor = Executors.newScheduledThreadPool(2);
    final AtomicInteger rows = new AtomicInteger();
    final AtomicInteger capacityWaits = new AtomicInteger();
    final AtomicBoolean sawForced = new AtomicBoolean();
    final List<Runnable> waiters = new ArrayList<>();
    int active;
    int sequence;

    TestPolicy(int width, int buckets, boolean projectedInput, boolean staggerCompletions) {
      this.width = width;
      this.buckets = buckets;
      this.projectedInput = projectedInput;
      this.staggerCompletions = staggerCompletions;
    }

    public Object batchKey(DecisionRequest request) {
      return request.legalActions().size() % buckets;
    }

    public int maxBatchSize(Object key) {
      return 16;
    }

    public synchronized Ingress tryAcquire(Object key, Runnable capacityAvailable) {
      if (active == 2) {
        capacityWaits.incrementAndGet();
        waiters.add(capacityAvailable);
        return null;
      }
      active++;
      int delay = sequence++ % 2;
      return new Ingress() {
        boolean submitted;

        public CompletableFuture<int[]> submit(List<DecisionRequest> requests) {
          submitted = true;
          CompletableFuture<int[]> result = new CompletableFuture<>();
          // 一方は文脈と合法手を同じ配列に格納し、他方は追加フィールドと別の行動テンソルを使う。
          // どちらのバッファも対戦環境や共通実装には公開しない。
          int[][] encoded = new int[requests.size()][];
          float[][][] actionFeatures = projectedInput ? new float[requests.size()][][] : null;
          for (int row = 0; row < requests.size(); row++) {
            DecisionRequest request = requests.get(row);
            Assert.assertEquals(batchKey(request), key);
            List<Action> actions = request.legalActions();
            encoded[row] = new int[width + (projectedInput ? 0 : actions.size())];
            if (projectedInput) {
              encoded[row][4] = request.observation().roundIndex();
              actionFeatures[row] = new float[actions.size()][2];
            }
            for (int action = 0; action < actions.size(); action++) {
              boolean wins =
                  actions.get(action).type() == Action.Type.RON_AGARI
                      || actions.get(action).type() == Action.Type.TSUMO_AGARI;
              if (projectedInput) {
                actionFeatures[row][action][0] = wins ? 1 : -1;
                actionFeatures[row][action][1] = wins ? action : -action;
              } else {
                encoded[row][width + action] = wins ? 1 : 0;
              }
            }
            encoded[row][0] = request.observation().turnNumber();
            encoded[row][1] = request.legalActions().hashCode();
            encoded[row][2] = request.observation().score(request.player());
            if (request.legalActions().size() == 1) sawForced.set(true);
          }
          Runnable completeBatch =
              () -> {
                try {
                  int[] selected = new int[requests.size()];
                  for (int row = 0; row < requests.size(); row++) {
                    DecisionRequest request = requests.get(row);
                    Assert.assertEquals(request.observation().turnNumber(), encoded[row][0]);
                    Assert.assertEquals(request.legalActions().hashCode(), encoded[row][1]);
                    Assert.assertEquals(
                        request.observation().score(request.player()), encoded[row][2]);
                    if (projectedInput)
                      Assert.assertEquals(request.observation().roundIndex(), encoded[row][4]);
                    selected[row] =
                        choose(encoded[row], projectedInput ? actionFeatures[row] : null);
                  }
                  rows.addAndGet(requests.size());
                  release();
                  result.complete(selected);
                } catch (Throwable error) {
                  release();
                  result.completeExceptionally(error);
                }
              };
          if (staggerCompletions) executor.schedule(completeBatch, delay, TimeUnit.MILLISECONDS);
          else executor.execute(completeBatch);
          return result;
        }

        public void close() {
          if (!submitted) {
            submitted = true;
            release();
          }
        }
      };
    }

    /** どちらの入力形式でも和了を優先し、和了できなければ先頭の合法手を選ぶ。 */
    private int choose(int[] context, float[][] actionFeatures) {
      int actions = actionFeatures == null ? context.length - width : actionFeatures.length;
      double scale = 1.0 + Math.abs(context[2]) / 100_000.0;
      int selected = 0;
      double maximum = Double.NEGATIVE_INFINITY;
      for (int action = 0; action < actions; action++) {
        double logit;
        if (actionFeatures == null) {
          int wins = context[width + action];
          logit = (2_000 * wins + (wins == 1 ? action : -action)) * scale;
        } else {
          double hidden = Math.max(0.0, actionFeatures[action][0] + 1.0);
          logit = (1_000 * hidden + actionFeatures[action][1]) * scale + context[4];
        }
        if (logit > maximum) {
          maximum = logit;
          selected = action;
        }
      }
      return selected;
    }

    private void release() {
      List<Runnable> signals;
      synchronized (this) {
        active--;
        signals = List.copyOf(waiters);
        waiters.clear();
      }
      signals.forEach(Runnable::run);
    }

    public void close() {
      executor.shutdownNow();
    }
  }
}
