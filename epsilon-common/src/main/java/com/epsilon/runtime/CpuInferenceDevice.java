package com.epsilon.runtime;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Supplier;

/** 同じ実行環境の全モデルでCPUの実行枠を共有し、待機バッチ数に上限を設ける。 */
public final class CpuInferenceDevice implements InferenceDispatcher.Device, AutoCloseable {
  private final Object key;
  private final int hostCapacity;
  private final ExecutorService executor =
      Executors.newSingleThreadExecutor(
          Thread.ofPlatform().daemon().name("epsilon-inference-cpu").factory());
  private CompletableFuture<Void> availability = new CompletableFuture<>();
  private boolean busy;
  private long pendingWork;

  CpuInferenceDevice(Object key, int hostCapacity) {
    this.key = key;
    this.hostCapacity = hostCapacity;
  }

  @Override
  public Object key() {
    return key;
  }

  @Override
  public int slots() {
    return 1;
  }

  @Override
  public int hostCapacity() {
    return hostCapacity;
  }

  @Override
  public synchronized int suppliedBatches() {
    return busy ? 1 : 0;
  }

  @Override
  public synchronized long pendingWork() {
    return pendingWork;
  }

  @Override
  public synchronized CompletableFuture<Void> capacityAvailable() {
    return availability;
  }

  public synchronized <R> CompletableFuture<R> trySubmit(
      long work, Runnable slotHandoff, Supplier<R> evaluate) {
    if (busy) return null;
    busy = true;
    pendingWork = work;
    CompletableFuture<R> result = new CompletableFuture<>();
    executor.execute(
        () -> {
          R value = null;
          Throwable failure = null;
          try {
            slotHandoff.run();
            value = evaluate.get();
          } catch (Throwable problem) {
            failure = problem;
          }
          CompletableFuture<Void> released;
          synchronized (CpuInferenceDevice.this) {
            busy = false;
            pendingWork = 0;
            released = availability;
            availability = new CompletableFuture<>();
          }
          released.complete(null);
          if (failure == null) result.complete(value);
          else result.completeExceptionally(failure);
        });
    return result;
  }

  @Override
  public void close() {
    executor.close();
  }
}
