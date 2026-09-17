package com.epsilon.runtime;

import static org.testng.Assert.*;

import com.epsilon.config.settings.InferenceBatchingSettings;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.testng.annotations.Test;

/** 複数モデル・デバイスの容量制限と投入順序、解放通知の競合、失敗時と終了時の資源回収を検証する。 */
public class InferenceDispatcherTest {
  @Test
  public void partialAdmissionWaitsForAnEmptySlotButDeadlineStillRescuesIt() {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      FakeDevice device = new FakeDevice("gpu:0", 2, 2);
      device.supplied = 2;
      try (var registration = dispatcher.register(List.of(new FakeBackend(device)))) {
        var admission =
            new InferenceAdmission<
                InferenceDispatcher.Registration<Integer, Integer>, String, Integer>(
                new InferenceBatchingSettings(1),
                (evaluator, key) -> 4,
                (evaluator, key) -> {
                  var attempt = evaluator.tryReserve();
                  if (!attempt.acquired())
                    return DecisionInferenceIngress.Attempt.blocked(attempt.available());
                  var reservation = attempt.reservation();
                  return DecisionInferenceIngress.Attempt.acquired(
                      new DecisionInferenceIngress(evaluator, reservation, reservation::close));
                },
                InferenceDispatcher.Registration::needsWork);
        admission.add(registration, "shape", 11, 0);
        var whileBusy = admission.startNextEncoding(100, false);
        if (whileBusy != null) admission.encodingFailed(whileBusy);
        assertNull(whileBusy);
        assertTrue(admission.hasQueuedRows());

        device.supplied = 1;
        var partial = admission.startNextEncoding(500, false);
        assertNotNull(partial);
        try {
          assertEquals(partial.reason(), BatchDispatchReason.SUPPLY);
          assertEquals(partial.rows(), List.of(11));
          device.supplied = 2;
          admission.add(registration, "rare", 22, 500);
          var deadline = admission.startNextEncoding(1500, false);
          if (deadline != null) admission.encodingFailed(deadline);
          assertNotNull(deadline);
          assertEquals(deadline.reason(), BatchDispatchReason.DEADLINE);
          assertEquals(deadline.rows(), List.of(22));
        } finally {
          admission.encodingFailed(partial);
        }
        assertFalse(admission.hasQueuedRows());
        assertFalse(admission.hasInFlightBatches());
      }
    }
  }

  @Test
  public void modelsShareOneHostBudgetAndReleaseOnlyAtSlotHandoff() throws Exception {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      FakeDevice device = new FakeDevice("gpu:0", 2);
      FakeBackend first = new FakeBackend(device);
      FakeBackend second = new FakeBackend(device);
      try (var a = dispatcher.register(List.of(first));
          var b = dispatcher.register(List.of(second))) {
        var one = a.tryReserve().reservation();
        var two = b.tryReserve().reservation();
        var blocked = a.tryReserve();
        assertFalse(blocked.acquired());
        assertEquals(dispatcher.metrics(), new InferenceDispatcher.Metrics(2, 2));
        CompletableFuture<Integer> result = one.submit(17);
        Flight flight = first.next();
        assertFalse(blocked.available().isDone());
        flight.stage();
        blocked.available().get(5, TimeUnit.SECONDS);
        a.tryReserve().reservation().close();
        flight.finish();
        assertEquals(result.get(5, TimeUnit.SECONDS), Integer.valueOf(17));
        two.close();
        dispatcher.awaitIdle();
        assertEquals(dispatcher.metrics().peakHostReservations(), 2);
        dispatcher.resetMetrics();
        a.tryReserve().reservation().close();
        assertEquals(dispatcher.metrics(), new InferenceDispatcher.Metrics(2, 1));
      }
    }
  }

  @Test
  public void saturatedGpuDoesNotBlockAnotherEligibleDevice() throws Exception {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      FakeDevice busy = new FakeDevice("gpu:0", 2);
      FakeDevice free = new FakeDevice("gpu:1", 2);
      busy.supplied = 1;
      FakeBackend blockedBackend = new FakeBackend(busy);
      FakeBackend runnableBackend = new FakeBackend(free);
      try (var blocked = dispatcher.register(List.of(blockedBackend));
          var runnable = dispatcher.register(List.of(runnableBackend))) {
        CompletableFuture<Integer> first = blocked.tryReserve().reservation().submit(11);
        CompletableFuture<Integer> second = runnable.tryReserve().reservation().submit(22);
        Flight next = runnableBackend.next();
        next.stage();
        next.finish();
        assertEquals(second.get(5, TimeUnit.SECONDS), Integer.valueOf(22));
        assertFalse(first.isDone());
        busy.free();
        next = blockedBackend.next();
        next.stage();
        next.finish();
        assertEquals(first.get(5, TimeUnit.SECONDS), Integer.valueOf(11));
      }
    }
  }

  @Test
  public void restrictedReservationsDoNotSupplyAnotherGpu() {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      FakeDevice first = new FakeDevice("gpu:0", 2);
      FakeDevice second = new FakeDevice("gpu:1", 2);
      first.supplied = 1;
      try (var restricted = dispatcher.register(List.of(new FakeBackend(first)));
          var other = dispatcher.register(List.of(new FakeBackend(second)))) {
        var reserved = restricted.tryReserve().reservation();
        assertFalse(restricted.needsWork());
        assertTrue(other.needsWork());
        reserved.close();
      }
    }
  }

  @Test
  public void removingADeviceDoesNotAdmitPastTheSmallerBudget() {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      var first = dispatcher.register(List.of(new FakeBackend(new FakeDevice("gpu:0", 2))));
      var second = dispatcher.register(List.of(new FakeBackend(new FakeDevice("gpu:1", 2))));
      try (first) {
        var one = first.tryReserve().reservation();
        var two = first.tryReserve().reservation();
        var three = first.tryReserve().reservation();
        var four = first.tryReserve().reservation();
        second.close();
        assertFalse(first.tryReserve().acquired());
        one.close();
        assertFalse(first.tryReserve().acquired());
        two.close();
        assertFalse(first.tryReserve().acquired());
        three.close();
        first.tryReserve().reservation().close();
        four.close();
      }
    }
  }

  @Test
  public void releaseBetweenNotificationCaptureAndFailedSubmitIsNotLost() throws Exception {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      FakeDevice device = new FakeDevice("gpu:0", 2);
      device.supplied = 1;
      FakeBackend backend = new FakeBackend(device);
      backend.releaseDuringAttempt = true;
      try (var registration = dispatcher.register(List.of(backend))) {
        var result = registration.tryReserve().reservation().submit(31);
        Flight flight = backend.next();
        flight.stage();
        flight.finish();
        assertEquals(result.get(5, TimeUnit.SECONDS), Integer.valueOf(31));
      }
    }
  }

  @Test
  public void failedCapacityNotificationFailsTheWaitingBatch() throws Exception {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      FakeDevice device = new FakeDevice("gpu:0", 2);
      device.supplied = 1;
      FakeBackend backend = new FakeBackend(device);
      try (var registration = dispatcher.register(List.of(backend))) {
        var result = registration.tryReserve().reservation().submit(41);
        backend.attempts.poll(5, TimeUnit.SECONDS);
        RuntimeException failure = new RuntimeException("device failed");
        device.available.completeExceptionally(failure);
        ExecutionException observed =
            expectThrows(ExecutionException.class, () -> result.get(5, TimeUnit.SECONDS));
        assertSame(observed.getCause(), failure);
        expectThrows(IllegalStateException.class, registration::tryReserve);
      }
    }
  }

  @Test
  public void synchronousSubmitFailureReturnsTheReservationOnce() throws Exception {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      FakeDevice device = new FakeDevice("gpu:0", 1);
      RuntimeException failure = new RuntimeException("submit failed");
      InferenceDispatcher.Backend<Integer, Integer> backend =
          new InferenceDispatcher.Backend<>() {
            @Override
            public InferenceDispatcher.Device device() {
              return device;
            }

            @Override
            public CompletableFuture<Integer> trySubmit(Integer batch, Runnable handoff) {
              handoff.run();
              throw failure;
            }
          };
      try (var registration = dispatcher.register(List.of(backend))) {
        var result = registration.tryReserve().reservation().submit(51);
        ExecutionException observed =
            expectThrows(ExecutionException.class, () -> result.get(5, TimeUnit.SECONDS));
        assertSame(observed.getCause(), failure);
        registration.tryReserve().reservation().close();
      }
    }
  }

  @Test
  public void cpuOnlyUsesTheSameFiniteAdmissionAndDrains() throws Exception {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher();
        CpuInferenceDevice cpu = new CpuInferenceDevice("cpu", 2)) {
      InferenceDispatcher.Backend<Integer, Integer> backend =
          new InferenceDispatcher.Backend<>() {
            @Override
            public InferenceDispatcher.Device device() {
              return cpu;
            }

            @Override
            public CompletableFuture<Integer> trySubmit(Integer batch, Runnable handoff) {
              return cpu.trySubmit(1, handoff, () -> batch * 2);
            }
          };
      try (var registration = dispatcher.register(List.of(backend))) {
        var one = registration.tryReserve().reservation();
        var two = registration.tryReserve().reservation();
        assertFalse(registration.tryReserve().acquired());
        var first = one.submit(4);
        var second = two.submit(5);
        assertEquals(first.get(5, TimeUnit.SECONDS), Integer.valueOf(8));
        assertEquals(second.get(5, TimeUnit.SECONDS), Integer.valueOf(10));
        registration.awaitIdle();
      }
    }
  }

  @Test
  public void resultCallbackCanWaitForTheNextInferenceAndCloseItsRegistration() throws Exception {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher();
        CpuInferenceDevice cpu = new CpuInferenceDevice("cpu", 2)) {
      InferenceDispatcher.Backend<Integer, Integer> backend =
          new InferenceDispatcher.Backend<>() {
            @Override
            public InferenceDispatcher.Device device() {
              return cpu;
            }

            @Override
            public CompletableFuture<Integer> trySubmit(Integer batch, Runnable handoff) {
              return cpu.trySubmit(1, handoff, () -> batch + 1);
            }
          };
      try (var registration = dispatcher.register(List.of(backend))) {
        var result =
            registration
                .tryReserve()
                .reservation()
                .submit(1)
                .thenApply(
                    first -> {
                      int second = registration.tryReserve().reservation().submit(first).join();
                      registration.close();
                      return second;
                    });
        assertEquals(result.get(5, TimeUnit.SECONDS), Integer.valueOf(3));
      }
    }
  }

  @Test
  public void closeDrainsDirectInputsAcceptedBeforeHostCapacityExists() throws Exception {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      FakeDevice device = new FakeDevice("gpu:0", 1);
      FakeBackend firstBackend = new FakeBackend(device);
      try (var first = dispatcher.register(List.of(firstBackend));
          var second = dispatcher.register(List.of(new FakeBackend(device)))) {
        var held = second.tryReserve().reservation();
        var firstResult = first.submit(61);
        var secondResult = first.submit(62);
        CompletableFuture<Void> closed = CompletableFuture.runAsync(first::close);
        try {
          expectThrows(TimeoutException.class, () -> closed.get(50, TimeUnit.MILLISECONDS));
        } finally {
          held.close();
        }
        Flight flight = firstBackend.next();
        flight.stage();
        flight.finish();
        flight = firstBackend.next();
        flight.stage();
        flight.finish();
        assertEquals(firstResult.get(5, TimeUnit.SECONDS), Integer.valueOf(61));
        assertEquals(secondResult.get(5, TimeUnit.SECONDS), Integer.valueOf(62));
        closed.get(5, TimeUnit.SECONDS);
        expectThrows(IllegalStateException.class, () -> first.submit(63));
      }
    }
  }

  private static final class FakeDevice implements InferenceDispatcher.Device {
    final String key;
    final int hostCapacity;
    final int slots;
    int supplied;
    CompletableFuture<Void> available = new CompletableFuture<>();

    FakeDevice(String key, int hostCapacity) {
      this(key, hostCapacity, 1);
    }

    FakeDevice(String key, int hostCapacity, int slots) {
      this.key = key;
      this.hostCapacity = hostCapacity;
      this.slots = slots;
    }

    @Override
    public Object key() {
      return key;
    }

    @Override
    public int slots() {
      return slots;
    }

    @Override
    public int hostCapacity() {
      return hostCapacity;
    }

    @Override
    public synchronized int suppliedBatches() {
      return supplied;
    }

    @Override
    public synchronized long pendingWork() {
      return supplied;
    }

    @Override
    public synchronized CompletableFuture<Void> capacityAvailable() {
      return available;
    }

    void free() {
      CompletableFuture<Void> released;
      synchronized (this) {
        supplied--;
        released = available;
        available = new CompletableFuture<>();
      }
      released.complete(null);
    }
  }

  private static final class FakeBackend implements InferenceDispatcher.Backend<Integer, Integer> {
    final FakeDevice device;
    final LinkedBlockingQueue<Flight> flights = new LinkedBlockingQueue<>();
    final LinkedBlockingQueue<Boolean> attempts = new LinkedBlockingQueue<>();
    boolean releaseDuringAttempt;

    FakeBackend(FakeDevice device) {
      this.device = device;
    }

    @Override
    public InferenceDispatcher.Device device() {
      return device;
    }

    @Override
    public CompletableFuture<Integer> trySubmit(Integer batch, Runnable handoff) {
      synchronized (device) {
        attempts.add(true);
        if (releaseDuringAttempt) {
          releaseDuringAttempt = false;
          device.free();
          return null;
        }
        if (device.supplied >= device.slots) return null;
        device.supplied++;
      }
      Flight flight = new Flight(this, batch, handoff);
      flights.add(flight);
      return flight.result;
    }

    Flight next() throws InterruptedException {
      Flight result = flights.poll(5, TimeUnit.SECONDS);
      assertNotNull(result, "Expected a runnable batch");
      return result;
    }
  }

  private record Flight(
      FakeBackend backend, int value, Runnable handoff, CompletableFuture<Integer> result) {
    Flight(FakeBackend backend, int value, Runnable handoff) {
      this(backend, value, handoff, new CompletableFuture<>());
    }

    void stage() {
      handoff.run();
    }

    void finish() {
      backend.device.free();
      result.complete(value);
    }
  }
}
