package com.epsilon.agent.runtime;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.runtime.InferenceAdmission.Reason;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import org.testng.annotations.Test;

/**
 * 受付がバッチを出す条件を確かめる。出すのは、待ちの期限が切れたとき、バッチが満杯になったとき、実行先のバッチが足りないとき
 * （このときは満杯でなくても出す）の 3 つ。予約できなかった評価器の行は、容量が空くまで受付に残る。
 * ほかに、予約の所有者の移し方と、受付と予約の組み合わせも確かめる。
 */
public class InferenceAdmissionTest {

  private static final long WAIT = 1_000;

  @Test
  public void aBlockedEvaluatorKeepsItsRowsAndWakesWhenCapacityFrees() {
    Object evaluator = new Object();
    CompletableFuture<Void> available = new CompletableFuture<>();
    AtomicInteger releases = new AtomicInteger();
    InferenceAdmission<Object, String, Integer> admission =
        new InferenceAdmission<>(
            WAIT,
            (owner, key) -> 2,
            (owner, key) ->
                available.isDone()
                    ? DecisionInferenceIngress.Attempt.acquired(
                        new DecisionInferenceIngress(owner, null, releases::incrementAndGet))
                    : DecisionInferenceIngress.Attempt.blocked(available),
            owner -> true);
    admission.add(evaluator, "shape", 1, 0);
    admission.add(evaluator, "shape", 2, 0);
    assertNull(admission.admit(0, false));
    assertTrue(admission.waitingForIngress());
    assertTrue(admission.hasQueuedRows());
    CompletableFuture<Void> wake = admission.ingressAvailable();
    assertFalse(wake.isDone());

    available.complete(null);
    assertTrue(wake.isDone());
    var batch = admission.admit(0, false);
    assertEquals(batch.rows(), List.of(1, 2));
    assertEquals(batch.reason(), Reason.FULL);
    assertFalse(admission.waitingForIngress());
    admission.encodingFailed(batch);
    assertEquals(releases.get(), 1);
    assertFalse(admission.hasInFlightBatches());
    assertFalse(admission.hasQueuedRows());
  }

  @Test
  public void aHandedOffIngressIsReleasedOnceByItsNewOwner() {
    Object evaluator = new Object();
    AtomicInteger releases = new AtomicInteger();
    InferenceAdmission<Object, String, Integer> admission =
        new InferenceAdmission<>(
            WAIT,
            (owner, key) -> 4,
            (owner, key) ->
                DecisionInferenceIngress.Attempt.acquired(
                    new DecisionInferenceIngress(owner, "route", releases::incrementAndGet)),
            owner -> true);
    admission.add(evaluator, "shape", 10, 0);
    var batch = admission.admit(0, false);
    assertEquals(batch.reason(), Reason.SUPPLY);
    expectThrows(IllegalArgumentException.class, () -> batch.ingress().handoff(new Object()));
    var ownership = batch.ingress().handoff(evaluator);
    expectThrows(IllegalStateException.class, () -> batch.ingress().abort());
    assertEquals(ownership.move(), "route");
    expectThrows(IllegalStateException.class, ownership::release);
    admission.inferenceFinished();
    assertEquals(releases.get(), 0);
    assertFalse(admission.hasInFlightBatches());
  }

  /** 評価器が予約を返すか移すかした後で失敗しても、受付はバッチが終わったと数え、予約をもう一度返さない。 */
  @Test
  public void aFailureAfterTheEvaluatorSettledTheIngressEndsTheBatchWithoutASecondRelease() {
    AtomicInteger releases = new AtomicInteger();
    InferenceAdmission<String, String, Integer> admission =
        new InferenceAdmission<>(
            WAIT,
            (owner, key) -> 1,
            (owner, key) ->
                DecisionInferenceIngress.Attempt.acquired(
                    new DecisionInferenceIngress(owner, null, releases::incrementAndGet)),
            owner -> true);
    admission.add("model", "shape", 1, 0);
    var aborted = admission.admit(0, false);
    aborted.ingress().abort();
    admission.encodingFailed(aborted);
    assertEquals(releases.get(), 1);
    assertFalse(admission.hasInFlightBatches());

    admission.add("model", "shape", 2, 0);
    var handedOff = admission.admit(0, false);
    handedOff.ingress().handoff("model").release();
    admission.encodingFailed(handedOff);
    assertEquals(releases.get(), 2);
    assertFalse(admission.hasInFlightBatches());
  }

  @Test
  public void anExpiredSparseKeyGoesBeforeFullBatches() {
    var admission = direct(owner -> false);
    admission.add("model", "rare", 99, 0);
    for (int row = 0; row < 4; row++) {
      admission.add("model", "common", row, 500);
    }
    var full = admission.admit(999, false);
    assertEquals(full.reason(), Reason.FULL);
    assertEquals(full.key(), "common");
    admission.encodingFailed(full);
    for (int row = 0; row < 4; row++) {
      admission.add("model", "common", row, 999);
    }
    var rare = admission.admit(WAIT, false);
    assertEquals(rare.key(), "rare");
    assertEquals(rare.reason(), Reason.DEADLINE);
    assertEquals(rare.queueWaitNanos(), WAIT);
    admission.encodingFailed(rare);
  }

  @Test
  public void aPartialBatchWaitsUntilTheDeviceNeedsWork() {
    AtomicBoolean needsWork = new AtomicBoolean();
    var admission = direct(owner -> needsWork.get());
    admission.add("model", "shape", 1, 0);
    assertNull(admission.admit(0, false));
    assertFalse(admission.waitingForIngress());
    needsWork.set(true);
    var batch = admission.admit(500, false);
    assertEquals(batch.reason(), Reason.SUPPLY);
    assertEquals(batch.rows(), List.of(1));
    admission.encodingFailed(batch);
  }

  @Test
  public void runningGenerationHoldsPartialRowsUntilTheyFillABatch() {
    var admission = direct(owner -> true);
    admission.add("model", "shape", 0, 0);
    assertNull(admission.admit(0, true));
    for (int row = 1; row < 4; row++) {
      admission.add("model", "shape", row, 100);
    }
    var full = admission.admit(100, true);
    assertEquals(full.reason(), Reason.FULL);
    assertEquals(full.rows(), List.of(0, 1, 2, 3));
    full.ingress().handoff("model").release();

    admission.add("model", "shape", 99, 100);
    assertNull(admission.admit(100, true));
    assertTrue(admission.hasInFlightBatches());
    var tail = admission.admit(200, false);
    assertEquals(tail.reason(), Reason.SUPPLY);
    assertEquals(tail.rows(), List.of(99));
    admission.encodingFailed(tail);
    admission.inferenceFinished();
    assertFalse(admission.hasInFlightBatches());

    InferenceAdmission.Metrics metrics = admission.metrics();
    assertEquals(metrics.admitted().get(Reason.FULL), new InferenceAdmission.Metrics.Count(1, 4));
    assertEquals(metrics.admitted().get(Reason.SUPPLY), new InferenceAdmission.Metrics.Count(1, 1));
    assertEquals(metrics.admitted().get(Reason.DEADLINE), new InferenceAdmission.Metrics.Count(0, 0));
    assertEquals(metrics.batches(), 2);
    assertEquals(metrics.maxOldestWaitNanos(), 100);
  }

  @Test
  public void aSupplyBatchTakesTheFullestKeyAndEvaluatorsTakeTurns() {
    var admission = direct(owner -> true);
    admission.add("a", "small", 1, 0);
    admission.add("a", "large", 2, 10);
    admission.add("a", "large", 3, 10);
    admission.add("b", "small", 4, 0);
    var first = admission.admit(20, false);
    assertEquals(first.evaluator(), "a");
    assertEquals(first.key(), "large");
    admission.encodingFailed(first);
    var second = admission.admit(20, false);
    assertEquals(second.evaluator(), "b");
    admission.encodingFailed(second);
    var third = admission.admit(20, false);
    assertEquals(third.evaluator(), "a");
    assertEquals(third.key(), "small");
    admission.encodingFailed(third);
    assertFalse(admission.hasQueuedRows());
  }

  @Test
  public void anEventWaitEndsAtTheOldestDeadline() throws InterruptedException {
    var admission = direct(owner -> false);
    var events = new LinkedBlockingQueue<String>();
    admission.add("model", "shape", 1, System.nanoTime() - WAIT);
    assertNull(admission.awaitEvent(events));
    events.add("ready");
    assertEquals(admission.awaitEvent(events), "ready");
  }

  @Test
  public void reservationsFromTheDispatcherBoundTheAdmittedBatches() throws Exception {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      Device device = new Device(1);
      var registration = dispatcher.register(List.of(new Backend(device)));
      try (registration) {
        InferenceAdmission<InferenceDispatcher.Registration<Integer, Integer>, String, Integer> admission =
            new InferenceAdmission<>(
                WAIT,
                (owner, key) -> 2,
                (owner, key) -> {
                  var attempt = owner.tryReserve();
                  if (!attempt.acquired()) {
                    return DecisionInferenceIngress.Attempt.blocked(attempt.available());
                  }
                  var reservation = attempt.reservation();
                  return DecisionInferenceIngress.Attempt.acquired(
                      new DecisionInferenceIngress(owner, reservation, reservation::close));
                },
                InferenceDispatcher.Registration::needsWork);
        for (int row = 0; row < 4; row++) {
          admission.add(registration, "shape", row, 0);
        }
        var first = admission.admit(0, true);
        assertNotNull(first);
        assertNull(admission.admit(0, true));
        assertTrue(admission.waitingForIngress());
        CompletableFuture<Void> wake = admission.ingressAvailable();

        @SuppressWarnings("unchecked")
        var reservation =
            (InferenceDispatcher.Reservation<Integer, Integer>) first.ingress().handoff(registration).move();
        CompletableFuture<Integer> result = reservation.submit(first.rows().size());
        wake.get(5, TimeUnit.SECONDS);
        assertEquals(result.get(5, TimeUnit.SECONDS), Integer.valueOf(2));
        admission.inferenceFinished();

        var second = admission.admit(0, true);
        assertSame(second.evaluator(), registration);
        assertEquals(second.rows(), List.of(2, 3));
        admission.encodingFailed(second);
      }
    }
  }

  private static InferenceAdmission<String, String, Integer> direct(Predicate<String> needsWork) {
    return new InferenceAdmission<>(
        WAIT,
        (owner, key) -> 4,
        (owner, key) -> DecisionInferenceIngress.Attempt.acquired(DecisionInferenceIngress.direct(owner)),
        needsWork);
  }

  /** スロットが 1 つで、バッチをすぐに返すデバイス。 */
  private record Device(int hostCapacity) implements InferenceDispatcher.Device {
    @Override
    public Object key() {
      return "gpu0";
    }

    @Override
    public int slots() {
      return 1;
    }

    @Override
    public int suppliedBatches() {
      return 0;
    }

    @Override
    public long pendingWork() {
      return 0;
    }

    @Override
    public CompletableFuture<Void> capacityAvailable() {
      return new CompletableFuture<>();
    }
  }

  /** スロットへ移したら、すぐに行の数を返すバックエンド。 */
  private record Backend(Device device) implements InferenceDispatcher.Backend<Integer, Integer> {
    @Override
    public CompletableFuture<Integer> trySubmit(Integer batch, Runnable slotHandoff) {
      slotHandoff.run();
      return CompletableFuture.completedFuture(batch);
    }
  }
}
