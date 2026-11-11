package com.epsilon.agent.runtime;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import org.testng.annotations.Test;

/** ホストの容量の予約と返却、実行先の選び方、デバイスが失敗したときと登録を解除したときの動きを確かめる。 */
public class InferenceDispatcherTest {

  @Test
  public void modelsOnOneDeviceShareItsHostCapacityUntilTheSlotHandoff() throws Exception {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      FakeDevice device = new FakeDevice("gpu0", 2, 1);
      FakeBackend first = new FakeBackend(device);
      try (var a = dispatcher.register(List.of(first));
          var b = dispatcher.register(List.of(new FakeBackend(device)))) {
        var one = a.tryReserve().reservation();
        var two = b.tryReserve().reservation();
        var blocked = a.tryReserve();
        assertFalse(blocked.acquired());
        assertEquals(dispatcher.metrics(), new InferenceDispatcher.Metrics(2, 2));

        CompletableFuture<Integer> result = one.submit(17);
        Flight flight = first.next();
        assertFalse(blocked.available().isDone());
        flight.handOff();
        blocked.available().get(5, TimeUnit.SECONDS);
        a.tryReserve().reservation().close();
        flight.finish();
        assertEquals(result.get(5, TimeUnit.SECONDS), Integer.valueOf(17));

        two.close();
        dispatcher.awaitIdle();
        dispatcher.resetMetrics();
        a.tryReserve().reservation().close();
        assertEquals(dispatcher.metrics(), new InferenceDispatcher.Metrics(2, 1));
      }
    }
  }

  @Test
  public void aFullDeviceDoesNotHoldBackAnotherDevice() throws Exception {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      FakeDevice busy = new FakeDevice("gpu0", 2, 1);
      FakeDevice free = new FakeDevice("gpu1", 2, 1);
      busy.supplied = 1;
      FakeBackend waiting = new FakeBackend(busy);
      FakeBackend running = new FakeBackend(free);
      try (var blocked = dispatcher.register(List.of(waiting));
          var runnable = dispatcher.register(List.of(running))) {
        CompletableFuture<Integer> first = blocked.tryReserve().reservation().submit(11);
        CompletableFuture<Integer> second = runnable.tryReserve().reservation().submit(22);
        Flight flight = running.next();
        flight.handOff();
        flight.finish();
        assertEquals(second.get(5, TimeUnit.SECONDS), Integer.valueOf(22));
        assertFalse(first.isDone());

        busy.free();
        flight = waiting.next();
        flight.handOff();
        flight.finish();
        assertEquals(first.get(5, TimeUnit.SECONDS), Integer.valueOf(11));
      }
    }
  }

  @Test
  public void reservationsCountTowardsTheDeviceThatCanRunThem() {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      FakeDevice first = new FakeDevice("gpu0", 2, 2);
      FakeDevice second = new FakeDevice("gpu1", 2, 2);
      first.supplied = 1;
      try (var restricted = dispatcher.register(List.of(new FakeBackend(first)));
          var other = dispatcher.register(List.of(new FakeBackend(second)))) {
        assertTrue(restricted.needsWork());
        var reserved = restricted.tryReserve().reservation();
        assertFalse(restricted.needsWork());
        assertTrue(other.needsWork());
        reserved.close();
        assertTrue(restricted.needsWork());
      }
    }
  }

  @Test
  public void removingADeviceShrinksTheHostCapacity() {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      var kept = dispatcher.register(List.of(new FakeBackend(new FakeDevice("gpu0", 2, 1))));
      var removed = dispatcher.register(List.of(new FakeBackend(new FakeDevice("gpu1", 2, 1))));
      try (kept) {
        var reservations = List.of(
            kept.tryReserve().reservation(),
            kept.tryReserve().reservation(),
            kept.tryReserve().reservation(),
            kept.tryReserve().reservation());
        removed.close();
        assertEquals(dispatcher.metrics().hostCapacity(), 2);
        for (int index = 0; index < 3; index++) {
          assertFalse(kept.tryReserve().acquired());
          reservations.get(index).close();
        }
        kept.tryReserve().reservation().close();
        reservations.get(3).close();
      }
    }
  }

  @Test
  public void aSubmitFailureReturnsTheReservationOnce() throws Exception {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      FakeDevice device = new FakeDevice("gpu0", 1, 1);
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
        ExecutionException observed = expectThrows(ExecutionException.class, () -> result.get(5, TimeUnit.SECONDS));
        assertSame(observed.getCause(), failure);
        registration.tryReserve().reservation().close();
      }
    }
  }

  @Test
  public void aFailedDeviceFailsTheWaitingBatchAndRefusesNewReservations() throws Exception {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      FakeDevice device = new FakeDevice("gpu0", 2, 1);
      device.supplied = 1;
      FakeBackend backend = new FakeBackend(device);
      try (var registration = dispatcher.register(List.of(backend))) {
        var result = registration.tryReserve().reservation().submit(41);
        assertNotNull(backend.attempts.poll(5, TimeUnit.SECONDS));
        RuntimeException failure = new RuntimeException("device failed");
        device.available.completeExceptionally(failure);
        ExecutionException observed = expectThrows(ExecutionException.class, () -> result.get(5, TimeUnit.SECONDS));
        assertSame(observed.getCause(), failure);
        expectThrows(IllegalStateException.class, registration::tryReserve);
      }
    }
  }

  @Test
  public void aReservationIsUsedOnce() throws Exception {
    try (InferenceDispatcher dispatcher = new InferenceDispatcher()) {
      FakeBackend backend = new FakeBackend(new FakeDevice("gpu0", 1, 1));
      try (var registration = dispatcher.register(List.of(backend))) {
        var reservation = registration.tryReserve().reservation();
        var result = reservation.submit(5);
        expectThrows(IllegalStateException.class, () -> reservation.submit(6));
        expectThrows(IllegalStateException.class, reservation::close);
        Flight flight = backend.next();
        flight.handOff();
        flight.finish();
        assertEquals(result.get(5, TimeUnit.SECONDS), Integer.valueOf(5));
      }
    }
  }

  @Test
  public void closingTheDispatcherNeedsClosedRegistrations() {
    InferenceDispatcher dispatcher = new InferenceDispatcher();
    var registration = dispatcher.register(List.of(new FakeBackend(new FakeDevice("gpu0", 1, 1))));
    expectThrows(IllegalStateException.class, dispatcher::close);
    registration.close();
    expectThrows(IllegalStateException.class, registration::tryReserve);
    dispatcher.close();
    FakeBackend late = new FakeBackend(new FakeDevice("gpu0", 1, 1));
    expectThrows(IllegalStateException.class, () -> dispatcher.register(List.of(late)));
  }

  /** スロットを数えるだけのデバイス。スロットを 1 つ空けるときに、待っている側へ知らせる。 */
  private static final class FakeDevice implements InferenceDispatcher.Device {
    final String key;
    final int hostCapacity;
    final int slots;
    int supplied;
    CompletableFuture<Void> available = new CompletableFuture<>();

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

  /** スロットが空いていればバッチを受け取るバックエンド。受け取ったバッチは、テストがスロットへの移動と完了を 1 つずつ進める。 */
  private static final class FakeBackend implements InferenceDispatcher.Backend<Integer, Integer> {
    final FakeDevice device;
    final LinkedBlockingQueue<Flight> flights = new LinkedBlockingQueue<>();
    final LinkedBlockingQueue<Boolean> attempts = new LinkedBlockingQueue<>();

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
        attempts.add(Boolean.TRUE);
        if (device.supplied >= device.slots) {
          return null;
        }
        device.supplied++;
      }
      Flight flight = new Flight(device, batch, handoff, new CompletableFuture<>());
      flights.add(flight);
      return flight.result();
    }

    Flight next() throws InterruptedException {
      Flight flight = flights.poll(5, TimeUnit.SECONDS);
      assertNotNull(flight, "Expected a submitted batch");
      return flight;
    }
  }

  private record Flight(FakeDevice device, int value, Runnable handoff, CompletableFuture<Integer> result) {
    void handOff() {
      handoff.run();
    }

    void finish() {
      device.free();
      result.complete(value);
    }
  }
}
