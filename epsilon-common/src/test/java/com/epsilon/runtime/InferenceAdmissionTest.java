package com.epsilon.runtime;

import com.epsilon.config.settings.InferenceBatchingSettings;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 推論容量の予約、所有権の移譲、期限付きバッチ処理、解放通知と失敗通知の競合を検証する。 */
public class InferenceAdmissionTest {
  @Test
  public void failedReservationPreservesRowsAndCapturesCapacityWake() {
    Object evaluator = new Object();
    var available = new CompletableFuture<Void>();
    var releases = new AtomicInteger();
    var admission =
        new InferenceAdmission<Object, String, Integer>(
            new InferenceBatchingSettings(1),
            (e, k) -> 2,
            (e, key) ->
                available.isDone()
                    ? DecisionInferenceIngress.Attempt.acquired(
                        new DecisionInferenceIngress(e, null, releases::incrementAndGet))
                    : DecisionInferenceIngress.Attempt.blocked(available),
            e -> true);
    admission.add(evaluator, "shape", 1, 0);
    admission.add(evaluator, "shape", 2, 0);
    Assert.assertNull(admission.startNextEncoding(1000, false));
    Assert.assertTrue(admission.hasQueuedRows());
    Assert.assertFalse(admission.hasInFlightBatches());
    var wake = admission.ingressAvailable();
    Assert.assertFalse(wake.isDone());
    available.complete(null);
    Assert.assertTrue(wake.isDone());
    var batch = admission.startNextEncoding(1000, false);
    Assert.assertEquals(batch.rows(), List.of(1, 2));
    admission.encodingFailed(batch);
    Assert.assertEquals(releases.get(), 1);
    Assert.assertFalse(admission.hasInFlightBatches());
    Assert.assertFalse(admission.hasQueuedRows());
  }

  @Test
  public void inferenceCompletionDoesNotReleaseTransferredReservationTwice() {
    Object evaluator = new Object();
    var releases = new AtomicInteger();
    var admission =
        new InferenceAdmission<Object, String, Integer>(
            new InferenceBatchingSettings(1),
            (e, k) -> 4,
            (e, key) ->
                DecisionInferenceIngress.Attempt.acquired(
                    new DecisionInferenceIngress(e, "route", releases::incrementAndGet)),
            e -> true);
    admission.add(evaluator, "shape", 10, 0);
    var batch = admission.startNextEncoding(0, false);
    Assert.assertEquals(batch.reason(), BatchDispatchReason.SUPPLY);
    batch.ingress().handoff(evaluator).release();
    admission.inferenceFinished();
    Assert.assertEquals(releases.get(), 1);
    Assert.assertFalse(admission.hasInFlightBatches());
  }

  @Test
  public void fullBatchesDoNotStarveAnExpiredSparseKey() {
    var admission = directAdmission(e -> false);
    admission.add("model", "rare", 99, 0);
    for (int row = 0; row < 4; row++) admission.add("model", "common", row, 500);
    var full = admission.startNextEncoding(999, false);
    Assert.assertEquals(full.reason(), BatchDispatchReason.FULL);
    admission.encodingFailed(full);
    for (int row = 0; row < 4; row++) admission.add("model", "common", row, 999);
    var rare = admission.startNextEncoding(1000, false);
    Assert.assertEquals(rare.key(), "rare");
    Assert.assertEquals(rare.reason(), BatchDispatchReason.DEADLINE);
    admission.encodingFailed(rare);
  }

  @Test
  public void partialBatchWaitsWhileSuppliedThenStartsAsSoonAsDeviceNeedsWork() {
    var needsWork = new AtomicBoolean();
    var admission = directAdmission(e -> needsWork.get());
    admission.add("model", "shape", 1, 0);
    Assert.assertNull(admission.startNextEncoding(0, false));
    Assert.assertFalse(admission.isWaitingForIngress());
    needsWork.set(true);
    var batch = admission.startNextEncoding(500, false);
    Assert.assertEquals(batch.reason(), BatchDispatchReason.SUPPLY);
    admission.encodingFailed(batch);
  }

  @Test
  public void activeInputGenerationKeepsPartialRowsUntilTheyFormAFullBatch() {
    var reservations = new AtomicInteger();
    var releases = new AtomicInteger();
    Object evaluator = new Object();
    var admission =
        new InferenceAdmission<Object, String, Object>(
            new InferenceBatchingSettings(1),
            (e, key) -> 4,
            (e, key) -> {
              reservations.incrementAndGet();
              return DecisionInferenceIngress.Attempt.acquired(
                  new DecisionInferenceIngress(e, null, releases::incrementAndGet));
            },
            e -> true);
    var rows = List.of(new Object(), new Object(), new Object(), new Object());
    admission.add(evaluator, "shape", rows.getFirst(), 0);
    Assert.assertNull(admission.startNextEncoding(0, true));
    Assert.assertTrue(admission.hasQueuedRows());
    Assert.assertFalse(admission.hasInFlightBatches());
    Assert.assertEquals(reservations.get(), 0);

    for (int index = 1; index < rows.size(); index++) {
      admission.add(evaluator, "shape", rows.get(index), 100);
    }
    var batch = admission.startNextEncoding(100, true);
    Assert.assertNotNull(batch);
    Assert.assertEquals(batch.reason(), BatchDispatchReason.FULL);
    Assert.assertEquals(batch.rows().size(), rows.size());
    for (int index = 0; index < rows.size(); index++) {
      Assert.assertSame(batch.rows().get(index), rows.get(index));
    }
    Assert.assertEquals(reservations.get(), 1);
    Assert.assertFalse(admission.hasQueuedRows());
    admission.encodingFailed(batch);
    Assert.assertEquals(releases.get(), 1);
    Assert.assertFalse(admission.hasInFlightBatches());
  }

  @Test
  public void tailStartsAfterInputGenerationEndsWhileEarlierInferenceIsOutstanding() {
    var admission = directAdmission(e -> true);
    for (int row = 0; row < 4; row++) admission.add("model", "shape", row, 0);
    var first = admission.startNextEncoding(0, true);
    Assert.assertEquals(first.reason(), BatchDispatchReason.FULL);
    first.ingress().handoff("model").release();

    admission.add("model", "shape", 99, 100);
    Assert.assertNull(admission.startNextEncoding(100, true));
    Assert.assertTrue(admission.hasInFlightBatches());
    var tail = admission.startNextEncoding(200, false);
    Assert.assertNotNull(tail);
    Assert.assertEquals(tail.reason(), BatchDispatchReason.SUPPLY);
    Assert.assertEquals(tail.rows(), List.of(99));
    tail.ingress().handoff("model").release();

    admission.inferenceFinished();
    Assert.assertTrue(admission.hasInFlightBatches());
    admission.inferenceFinished();
    Assert.assertFalse(admission.hasInFlightBatches());
    Assert.assertFalse(admission.hasQueuedRows());
  }

  @Test
  public void activeInputGenerationDoesNotStarveAnOlderModelAtItsDeadline() {
    var admission = directAdmission(e -> true);
    String oldModel = new String("model");
    String newModel = new String("model");
    admission.add(oldModel, "shape", 99, 0);
    for (int row = 0; row < 4; row++) admission.add(newModel, "shape", row, 500);
    var full = admission.startNextEncoding(999, true);
    Assert.assertSame(full.evaluator(), newModel);
    Assert.assertEquals(full.reason(), BatchDispatchReason.FULL);
    Assert.assertEquals(full.rows(), List.of(0, 1, 2, 3));
    admission.encodingFailed(full);

    for (int row = 4; row < 8; row++) admission.add(newModel, "shape", row, 999);
    var expired = admission.startNextEncoding(1000, true);
    Assert.assertSame(expired.evaluator(), oldModel);
    Assert.assertEquals(expired.reason(), BatchDispatchReason.DEADLINE);
    Assert.assertEquals(expired.rows(), List.of(99));
    admission.encodingFailed(expired);
    var remaining = admission.startNextEncoding(1000, true);
    Assert.assertSame(remaining.evaluator(), newModel);
    Assert.assertEquals(remaining.rows(), List.of(4, 5, 6, 7));
    admission.encodingFailed(remaining);
    Assert.assertFalse(admission.hasInFlightBatches());
    Assert.assertFalse(admission.hasQueuedRows());
  }

  @Test
  public void supplyForAnotherEvaluatorDoesNotBlockAnEligibleDevice() {
    var admission = directAdmission(e -> e.equals("gpu1"));
    admission.add("gpu0", "shape", 1, 0);
    admission.add("gpu1", "shape", 2, 0);
    var batch = admission.startNextEncoding(0, false);
    Assert.assertEquals(batch.evaluator(), "gpu1");
    admission.encodingFailed(batch);
    Assert.assertTrue(admission.hasQueuedRows());
  }

  @Test
  public void releaseDuringCapacityAttemptDoesNotLoseNotification() {
    var available = new CompletableFuture<Void>();
    var admission =
        new InferenceAdmission<String, String, Integer>(
            new InferenceBatchingSettings(0),
            (e, key) -> 1,
            (e, key) -> {
              available.complete(null);
              return DecisionInferenceIngress.Attempt.blocked(available);
            },
            e -> true);
    admission.add("model", "shape", 1, 0);
    Assert.assertNull(admission.startNextEncoding(0, false));
    Assert.assertTrue(admission.ingressAvailable().isDone());
    Assert.assertTrue(admission.hasQueuedRows());
  }

  @Test
  public void laterEvaluatorReleaseCompletesTheAlreadySubscribedWake() {
    Object first = new Object();
    Object second = new Object();
    var firstAvailable = new CompletableFuture<Void>();
    var secondAvailable = new CompletableFuture<Void>();
    var admission =
        new InferenceAdmission<Object, String, Integer>(
            new InferenceBatchingSettings(1),
            (e, key) -> 1,
            (e, key) ->
                DecisionInferenceIngress.Attempt.blocked(
                    e == first ? firstAvailable : secondAvailable),
            e -> true);
    admission.add(first, "shape", 1, 0);
    Assert.assertNull(admission.startNextEncoding(0, false));
    var subscribed = admission.ingressAvailable();
    var notifications = new AtomicInteger();
    subscribed.thenRun(notifications::incrementAndGet);

    admission.add(second, "shape", 2, 0);
    Assert.assertNull(admission.startNextEncoding(0, false));
    Assert.assertSame(admission.ingressAvailable(), subscribed);
    secondAvailable.complete(null);
    Assert.assertTrue(subscribed.isDone());
    Assert.assertEquals(notifications.get(), 1);
    firstAvailable.complete(null);
    Assert.assertEquals(notifications.get(), 1);
  }

  @Test
  public void laterCapacityFailureReachesTheAlreadySubscribedWake() {
    Object first = new Object();
    Object second = new Object();
    var firstAvailable = new CompletableFuture<Void>();
    var secondAvailable = new CompletableFuture<Void>();
    var admission =
        new InferenceAdmission<Object, String, Integer>(
            new InferenceBatchingSettings(1),
            (e, key) -> 1,
            (e, key) ->
                DecisionInferenceIngress.Attempt.blocked(
                    e == first ? firstAvailable : secondAvailable),
            e -> true);
    admission.add(first, "shape", 1, 0);
    Assert.assertNull(admission.startNextEncoding(0, false));
    var subscribed = admission.ingressAvailable();
    admission.add(second, "shape", 2, 0);
    Assert.assertNull(admission.startNextEncoding(0, false));
    var failure = new IllegalStateException("capacity failed");
    secondAvailable.completeExceptionally(failure);
    Assert.assertSame(
        Assert.expectThrows(CompletionException.class, subscribed::join).getCause(), failure);
  }

  @Test
  public void expiredDeadlineReturnsToDispatchWithoutATimerEvent() throws InterruptedException {
    var admission = directAdmission(e -> false);
    long enqueued = System.nanoTime() - 1_000;
    admission.add("model", "shape", 1, enqueued);
    Assert.assertNull(admission.startNextEncoding(enqueued, false));
    Assert.assertFalse(admission.isWaitingForIngress());
    Assert.assertNull(admission.awaitEvent(new LinkedBlockingQueue<>()));
    var batch = admission.startNextEncoding(System.nanoTime(), false);
    Assert.assertEquals(batch.reason(), BatchDispatchReason.DEADLINE);
    admission.encodingFailed(batch);
  }

  @Test(timeOut = 10000)
  public void queuedProducerCompletionReleasesTheTailBeforeTheDeadline()
      throws InterruptedException {
    var admission =
        new InferenceAdmission<String, String, Integer>(
            new InferenceBatchingSettings(60_000_000),
            (e, key) -> 4,
            (e, key) ->
                DecisionInferenceIngress.Attempt.acquired(DecisionInferenceIngress.direct(e)),
            e -> true);
    long now = System.nanoTime();
    admission.add("model", "shape", 1, now);
    Assert.assertNull(admission.startNextEncoding(now, true));
    var events = new LinkedBlockingQueue<String>();
    events.add("producer completed");
    Assert.assertEquals(admission.awaitEvent(events), "producer completed");
    var tail = admission.startNextEncoding(now, false);
    Assert.assertEquals(tail.reason(), BatchDispatchReason.SUPPLY);
    Assert.assertEquals(tail.rows(), List.of(1));
    admission.encodingFailed(tail);
    Assert.assertFalse(admission.hasQueuedRows());
    Assert.assertFalse(admission.hasInFlightBatches());
  }

  private static InferenceAdmission<String, String, Integer> directAdmission(
      java.util.function.Predicate<String> needsWork) {
    return new InferenceAdmission<>(
        new InferenceBatchingSettings(1),
        (e, key) -> 4,
        (e, key) -> DecisionInferenceIngress.Attempt.acquired(DecisionInferenceIngress.direct(e)),
        needsWork);
  }
}
