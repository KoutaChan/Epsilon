package com.epsilon.spi;

import ai.djl.engine.Engine;
import ai.djl.pytorch.engine.PtEngine;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.runtime.MeasurementStatus;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

/** 一回の対戦が所有する CPU エンコーダーと GPU 実行資源。系列間で共有します。 */
public final class PolicyExecutionContext implements AutoCloseable {
  private final DecisionExecutionContext devices = new DecisionExecutionContext();
  private final int encodingWorkers;
  private final int slotsPerDevice;
  private final int readyBatchesPerDevice;
  private ExecutorService encoder;

  public PolicyExecutionContext() {
    this(Math.min(32, Runtime.getRuntime().availableProcessors()));
  }

  public PolicyExecutionContext(int encodingWorkers) {
    this(encodingWorkers, 2, 4);
  }

  public PolicyExecutionContext(
      int encodingWorkers, int slotsPerDevice, int readyBatchesPerDevice) {
    if (encodingWorkers <= 0)
      throw new IllegalArgumentException("encodingWorkers must be positive");
    if (slotsPerDevice < 1 || slotsPerDevice > 4 || readyBatchesPerDevice < slotsPerDevice) {
      throw new IllegalArgumentException(
          "GPU slots must be 1..4 and ready capacity must cover slots");
    }
    this.encodingWorkers = encodingWorkers;
    this.slotsPerDevice = slotsPerDevice;
    this.readyBatchesPerDevice = readyBatchesPerDevice;
  }

  public DecisionExecutionContext devices() {
    return devices;
  }

  public int slotsPerDevice() {
    return slotsPerDevice;
  }

  public int readyBatchesPerDevice() {
    return readyBatchesPerDevice;
  }

  /** 使用GPUのメモリ割り当て機構統計だけを読む。GPU実行の成否と測定APIの可用性は分けます。 */
  public MemorySnapshot memorySnapshot() {
    List<DeviceMemory> memory = new ArrayList<>();
    devices.visitStreams(
        streams -> {
          try {
            var stats = ((PtEngine) Engine.getEngine("PyTorch")).getMemoryStats(streams.device());
            memory.add(
                new DeviceMemory(
                    streams.device().getDeviceId(),
                    MeasurementStatus.AVAILABLE,
                    stats.getAllocatedBytes(),
                    stats.getReservedBytes(),
                    stats.getPeakAllocatedBytes(),
                    null));
          } catch (RuntimeException | LinkageError failure) {
            memory.add(
                new DeviceMemory(
                    streams.device().getDeviceId(),
                    MeasurementStatus.UNAVAILABLE,
                    null,
                    null,
                    null,
                    failure.getClass().getSimpleName() + ": " + failure.getMessage()));
          }
        });
    long available =
        memory.stream().filter(value -> value.status() == MeasurementStatus.AVAILABLE).count();
    MeasurementStatus status =
        memory.isEmpty()
            ? MeasurementStatus.NOT_APPLICABLE
            : available == memory.size()
                ? MeasurementStatus.AVAILABLE
                : available == 0 ? MeasurementStatus.UNAVAILABLE : MeasurementStatus.PARTIAL;
    return new MemorySnapshot(
        status,
        "PyTorch allocator per device; peak is since its last reset, including warmup and other"
            + " sessions; arena does not reset it",
        memory);
  }

  public record MemorySnapshot(
      MeasurementStatus status, String scope, List<DeviceMemory> devices) {}

  public record DeviceMemory(
      int device,
      MeasurementStatus status,
      Long allocatedBytes,
      Long reservedBytes,
      Long peakAllocatedBytes,
      String reason) {}

  /** 互いに重ならない行範囲を並列に書き込み、全書き込み処理の終了後にだけ完了します。 */
  public CompletableFuture<Void> encodeRows(int rows, RangeEncoder encodeRange) {
    return encodeRows(rows, encodeRange, () -> {});
  }

  /** 最初のエンコーダータスクが実行された時刻を、バッチごとに一回だけ通知します。 */
  public CompletableFuture<Void> encodeRows(
      int rows, RangeEncoder encodeRange, Runnable encodingStarted) {
    int tasks = rows < 128 ? 1 : Math.min(encodingWorkers, rows / 64);
    if (tasks == 1) {
      encodingStarted.run();
      encodeRange.encode(0, rows);
      return CompletableFuture.completedFuture(null);
    }
    ExecutorService executor = encoder();
    CompletableFuture<?>[] complete = new CompletableFuture<?>[tasks];
    AtomicBoolean started = new AtomicBoolean();
    for (int task = 0; task < tasks; task++) {
      int from = task * rows / tasks;
      int to = (task + 1) * rows / tasks;
      complete[task] =
          CompletableFuture.runAsync(
              () -> {
                if (started.compareAndSet(false, true)) encodingStarted.run();
                encodeRange.encode(from, to);
              },
              executor);
    }
    return CompletableFuture.allOf(complete);
  }

  @FunctionalInterface
  public interface RangeEncoder {
    /** 指定範囲だけを書き込み、戻るまでに借用した作業領域を返却します。 */
    void encode(int fromInclusive, int toExclusive);
  }

  private synchronized ExecutorService encoder() {
    if (encoder == null) {
      encoder =
          Executors.newFixedThreadPool(
              encodingWorkers, Thread.ofPlatform().daemon().name("epsilon-encode-", 0).factory());
    }
    return encoder;
  }

  @Override
  public void close() {
    if (encoder != null) encoder.close();
    devices.close();
  }
}
