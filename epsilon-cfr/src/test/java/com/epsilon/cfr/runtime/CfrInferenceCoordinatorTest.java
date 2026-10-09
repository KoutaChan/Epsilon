package com.epsilon.cfr.runtime;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.collection.Prediction;
import com.epsilon.cfr.data.ObservationPayload;
import com.epsilon.cfr.model.CfrModel;
import com.epsilon.cfr.model.CfrModelConfig;
import com.epsilon.cfr.model.CfrObservationCodec;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.runtime.InferenceDispatcher;
import com.epsilon.spi.PolicyExecutionContext;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 入力計画をCPUプールへ送った後も、受付予約と借用観測の寿命が保たれることを確認する。 */
@Test(groups = "native")
public class CfrInferenceCoordinatorTest {
  public void cancellingTheCallerDoesNotReturnInputsBeforeInternalWorkAndCloseFinish()
      throws Exception {
    var context = context();
    try (var execution = new PolicyExecutionContext(1, 1, 1);
        var model = model();
        var coordinator = new CfrInferenceCoordinator(List.of(model), execution);
        var closer = Executors.newVirtualThreadPerTaskExecutor()) {
      var release = new CompletableFuture<Void>();
      var entered = new CountDownLatch(1);
      var device = execution.devices().cpuInference(Device.cpu(), 1);
      var blocker = device.trySubmit(1, entered::countDown, () -> release.join());
      try {
        Assert.assertTrue(entered.await(10, TimeUnit.SECONDS));
        var prediction = coordinator.submit(List.of(context));
        await(() -> coordinator.inputStorageSnapshot().borrowedBuffers() > 0);
        Assert.assertTrue(prediction.cancel(true));
        Assert.assertTrue(
            coordinator.inputStorageSnapshot().borrowedBuffers() > 0,
            "Caller cancellation must not recycle input still queued for inference");
        var closing = closer.submit(coordinator::close);
        Assert.expectThrows(TimeoutException.class, () -> closing.get(200, TimeUnit.MILLISECONDS));
        release.complete(null);
        closing.get(10, TimeUnit.SECONDS);
        Assert.assertTrue(prediction.isCancelled());
        var storage = coordinator.inputStorageSnapshot();
        Assert.assertEquals(storage.borrowedBuffers(), 0);
        Assert.assertEquals(storage.retainedBuffers(), 0);
        Assert.assertEquals(storage.retainedBytes(), 0L);
      } finally {
        release.complete(null);
        blocker.get(10, TimeUnit.SECONDS);
      }
    }
  }

  public void payloadTailRetainsHostReservationAndCloseWaitsAfterInferenceCompletes()
      throws Exception {
    var context = context();
    try (var execution = new PolicyExecutionContext(1, 1, 1);
        var model = model();
        var closer = Executors.newVirtualThreadPerTaskExecutor()) {
      var encoded = model.encode(List.of(context));
      var policies = model.evaluateEncoded(encoded);
      var payloads = encoded.encodePayloadsAsync(execution).join();
      var saved = new CompletableFuture<List<ObservationPayload>>();
      var computed = new CountDownLatch(1);
      var device = execution.devices().cpuInference(Device.cpu(), 1);
      var backend =
          new InferenceDispatcher.Backend<CfrModel.EncodedBatch, List<Prediction>>() {
            @Override
            public InferenceDispatcher.Device device() {
              return device;
            }

            @Override
            public CompletableFuture<List<Prediction>> trySubmit(
                CfrModel.EncodedBatch batch, Runnable handoff) {
              var completion = new CfrPredictionCompletion(() -> saved, handoff);
              completion.handoff();
              completion.complete(policies);
              computed.countDown();
              return completion.result();
            }
          };
      try (var registration =
          execution.devices().inferenceDispatcher().register(List.of(backend))) {
        try {
          var first = registration.tryReserve();
          Assert.assertTrue(first.acquired());
          var result = first.reservation().submit(encoded);
          Assert.assertTrue(computed.await(10, TimeUnit.SECONDS));
          Assert.assertFalse(result.isDone(), "Payload encoding is still using the host input");
          Assert.assertFalse(
              registration.tryReserve().acquired(),
              "Payload tails must remain within host capacity");
          var closing = closer.submit(registration::close);
          Assert.expectThrows(
              TimeoutException.class, () -> closing.get(200, TimeUnit.MILLISECONDS));
          saved.complete(payloads);
          var actual = result.get(10, TimeUnit.SECONDS);
          Assert.assertSame(actual.getFirst().observationPayload(), payloads.getFirst());
          closing.get(10, TimeUnit.SECONDS);
        } finally {
          saved.complete(payloads);
        }
      }
    }
  }

  public void inferenceFailureWaitsForPayloadAndPreservesBothFailures() throws Exception {
    var saved = new CompletableFuture<List<ObservationPayload>>();
    var handoffs = new AtomicInteger();
    var completion = new CfrPredictionCompletion(() -> saved, handoffs::incrementAndGet);
    var inferenceFailure = new IllegalStateException("Inference failed");
    var saveFailure = new IllegalArgumentException("Payload failed");
    completion.handoff();
    completion.fail(inferenceFailure);
    Assert.assertFalse(completion.result().isDone());
    Assert.assertEquals(handoffs.get(), 0);
    saved.completeExceptionally(saveFailure);
    var failure =
        Assert.expectThrows(
            ExecutionException.class, () -> completion.result().get(10, TimeUnit.SECONDS));
    Assert.assertSame(failure.getCause(), inferenceFailure);
    Assert.assertEquals(inferenceFailure.getSuppressed(), new Throwable[] {saveFailure});
    Assert.assertEquals(handoffs.get(), 1);
  }

  public void payloadFailureWaitsForInferenceReleaseAndEarlyFailureDoesNotStartEncoding()
      throws Exception {
    var saveFailure = new IllegalArgumentException("Payload failed");
    var handoffs = new AtomicInteger();
    var completion =
        new CfrPredictionCompletion(
            () -> CompletableFuture.failedFuture(saveFailure), handoffs::incrementAndGet);
    completion.handoff();
    Assert.assertEquals(handoffs.get(), 1);
    Assert.assertFalse(completion.result().isDone(), "Inference still owns its device resources");
    try (var model = model()) {
      completion.complete(model.evaluateEncoded(model.encode(List.of())));
    }
    var failure =
        Assert.expectThrows(
            ExecutionException.class, () -> completion.result().get(10, TimeUnit.SECONDS));
    Assert.assertSame(failure.getCause(), saveFailure);

    var encodes = new AtomicInteger();
    var early =
        new CfrPredictionCompletion(
            () -> {
              encodes.incrementAndGet();
              return CompletableFuture.completedFuture(List.of());
            },
            handoffs::incrementAndGet);
    var rejected = new IllegalStateException("Backend rejected the batch");
    early.fail(rejected);
    Assert.assertEquals(encodes.get(), 0);
    Assert.assertEquals(handoffs.get(), 1);
    var earlyFailure =
        Assert.expectThrows(
            ExecutionException.class, () -> early.result().get(10, TimeUnit.SECONDS));
    Assert.assertSame(earlyFailure.getCause(), rejected);
  }

  public void planningFailureReleasesReservationAndAllowsTheNextBatch() throws Exception {
    var context = context();
    try (var execution = new PolicyExecutionContext(1);
        var model = model();
        var coordinator = new CfrInferenceCoordinator(List.of(model), execution)) {
      var release = new CompletableFuture<Void>();
      CountDownLatch entered = new CountDownLatch(1);
      execution.encodeRowsAsync(
          1,
          (from, to) -> {
            entered.countDown();
            release.join();
          });
      try {
        Assert.assertTrue(entered.await(10, TimeUnit.SECONDS));
        Assert.expectThrows(
            ExecutionException.class,
            () -> coordinator.submit(List.of()).get(10, TimeUnit.SECONDS));
        var invalid = new DecisionContext(1, 1, 0, context.observation(), List.of());
        var failed = coordinator.submit(List.of(invalid));
        Assert.assertFalse(failed.isDone(), "Input planning must run on the reserved CPU worker");
        release.complete(null);
        var error =
            Assert.expectThrows(ExecutionException.class, () -> failed.get(10, TimeUnit.SECONDS));
        Assert.assertTrue(error.getCause() instanceof IllegalArgumentException);
        var actual = coordinator.submit(List.of(context)).get(10, TimeUnit.SECONDS);
        Assert.assertEquals(
            actual.getFirst().observationPayload(), CfrObservationCodec.encodePayload(context));
        Assert.assertEquals(coordinator.inputStorageSnapshot().borrowedBuffers(), 0);
        Assert.assertTrue(coordinator.inputStorageSnapshot().retainedBuffers() > 0);
      } finally {
        release.complete(null);
      }
    }
  }

  public void closeWaitsForQueuedPlanningAndKeepsTheObservationOwnedUntilCompletion()
      throws Exception {
    var context = context();
    var expected = CfrObservationCodec.encodePayload(context);
    try (var execution = new PolicyExecutionContext(1);
        var model = model();
        var coordinator = new CfrInferenceCoordinator(List.of(model), execution);
        var closer = Executors.newVirtualThreadPerTaskExecutor()) {
      var release = new CompletableFuture<Void>();
      CountDownLatch entered = new CountDownLatch(1);
      execution.encodeRowsAsync(
          1,
          (from, to) -> {
            entered.countDown();
            release.join();
          });
      try {
        Assert.assertTrue(entered.await(10, TimeUnit.SECONDS));
        var prediction = coordinator.submit(List.of(context));
        var closing = closer.submit(coordinator::close);
        Assert.expectThrows(TimeoutException.class, () -> closing.get(200, TimeUnit.MILLISECONDS));
        Assert.assertFalse(prediction.isDone());
        Assert.assertEquals(CfrObservationCodec.encodePayload(context), expected);
        release.complete(null);
        Assert.assertEquals(
            prediction.get(10, TimeUnit.SECONDS).getFirst().observationPayload(), expected);
        closing.get(10, TimeUnit.SECONDS);
        Assert.assertEquals(coordinator.inputStorageSnapshot().borrowedBuffers(), 0);
        Assert.assertEquals(coordinator.inputStorageSnapshot().retainedBytes(), 0L);
        Assert.expectThrows(IllegalStateException.class, coordinator::tryAcquireInferenceIngress);
      } finally {
        release.complete(null);
      }
    }
  }

  private static void await(BooleanSupplier condition) throws InterruptedException {
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(10);
    while (!condition.getAsBoolean()) {
      Assert.assertTrue(System.nanoTime() < deadline, "Timed out waiting for input storage");
      Thread.sleep(1);
    }
  }

  private static CfrModel model() {
    try (var source =
        new CfrModel(
            new CfrModelConfig(
                16,
                2,
                .001f,
                0,
                0,
                731,
                2,
                DecisionComputePrecision.FLOAT32,
                DecisionTensorTransfer.DIRECT_BUFFER,
                EpsilonUtilityProfile.TENHOU),
            CfrModel.Role.AVERAGE,
            Device.cpu())) {
      return source.copyForInference(Device.cpu());
    }
  }

  private static DecisionContext context() {
    var engine = new GameEngine(811L);
    var point = ((GameStepResult.AwaitingDecisions) engine.stepHanchan()).decisions().getFirst();
    return new DecisionContext(
        1, point.id(), point.player(), engine.observation(point.player()), point.legalActions());
  }
}
