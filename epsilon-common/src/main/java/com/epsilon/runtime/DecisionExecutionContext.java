package com.epsilon.runtime;

import ai.djl.Device;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;

/** 一つの処理で、収集・学習・評価が再利用するGPUストリームを所有する。 */
public final class DecisionExecutionContext implements AutoCloseable {

  private final Map<Device, DecisionDeviceStreams> devices = new LinkedHashMap<>();
  private final Map<Device, CpuInferenceDevice> cpuDevices = new LinkedHashMap<>();
  private final InferenceDispatcher inferenceDispatcher = new InferenceDispatcher();
  private boolean closed;
  private Throwable drainFailure;

  /** CPUだけの処理ではGPUリソースを作らない。 */
  public DecisionExecutionContext() {}

  public InferenceDispatcher inferenceDispatcher() {
    return inferenceDispatcher;
  }

  /** CPUもモデルごとに実行サービスを増やさず、物理実行枠を共有する。 */
  public synchronized CpuInferenceDevice cpuInference(Device device, int hostCapacity) {
    if (closed) throw new IllegalStateException("Decision execution context is closed");
    CpuInferenceDevice runtime =
        cpuDevices.computeIfAbsent(device, key -> new CpuInferenceDevice(key, hostCapacity));
    if (runtime.hostCapacity() != hostCapacity)
      throw new IllegalArgumentException("CPU inference capacity differs");
    return runtime;
  }

  /** GPUの役割別ストリームを借りる。借用者はストリームを閉じない。 */
  public synchronized DecisionDeviceStreams streams(Device device) {
    Objects.requireNonNull(device, "device");
    if (closed) {
      throw new IllegalStateException("Decision execution context is closed");
    }
    if (drainFailure != null) {
      throw new IllegalStateException("Decision execution context failed to drain", drainFailure);
    }
    if (!device.isGpu()) {
      throw new IllegalArgumentException("Decision streams require a GPU device: " + device);
    }
    return devices.computeIfAbsent(device, DecisionDeviceStreams::new);
  }

  /** 要求を投入する処理を停止したフェーズ境界で、全役割のGPU処理完了を待つ。 */
  public synchronized void awaitIdle() {
    if (closed) {
      throw new IllegalStateException("Decision execution context is closed");
    }
    if (drainFailure != null) {
      throw new IllegalStateException("Decision execution context failed to drain", drainFailure);
    }
    try {
      for (DecisionDeviceStreams streams : devices.values()) {
        DecisionDevicePipeline.requireReleased(streams);
        streams.awaitIdle();
      }
    } catch (RuntimeException | Error failure) {
      drainFailure = failure;
      throw failure;
    }
  }

  public synchronized void visitStreams(Consumer<DecisionDeviceStreams> visitor) {
    devices.values().forEach(visitor);
  }

  /** モデルの利用権を保ったまま、ウォームアップで発行した推論を回収する。 */
  public synchronized void awaitInferenceIdle() {
    inferenceDispatcher.awaitIdle();
    for (DecisionDeviceStreams streams : devices.values()) {
      synchronized (streams) {
        if (streams.pipeline != null) streams.pipeline.awaitIdle();
      }
    }
  }

  /** 新規要求の投入を止めて推論を回収した後に、同じ実行環境の計測区間を切り替える。 */
  public synchronized void resetInferenceMetrics() {
    inferenceDispatcher.resetMetrics();
    for (DecisionDeviceStreams streams : devices.values()) {
      synchronized (streams) {
        if (streams.pipeline != null) streams.pipeline.resetMetrics();
      }
    }
  }

  public synchronized List<DecisionDevicePipeline.Metrics> inferenceMetrics() {
    ArrayList<DecisionDevicePipeline.Metrics> result = new ArrayList<>(devices.size());
    for (DecisionDeviceStreams streams : devices.values()) {
      synchronized (streams) {
        if (streams.pipeline != null) result.add(streams.pipeline.metrics());
      }
    }
    return result;
  }

  public InferenceDispatcher.Metrics inferenceDispatchMetrics() {
    return inferenceDispatcher.metrics();
  }

  /** 全利用者と未完了処理を閉じた後、処理の最後に呼ぶ。 */
  @Override
  public synchronized void close() {
    if (closed) {
      return;
    }
    try {
      inferenceDispatcher.close();
      for (CpuInferenceDevice runtime : cpuDevices.values()) runtime.close();
      cpuDevices.clear();
      awaitIdle();
    } finally {
      closed = true;
    }
    Throwable failure = null;
    for (DecisionDeviceStreams streams : devices.values()) {
      try {
        streams.closeOwned();
      } catch (RuntimeException | Error closeFailure) {
        if (failure == null) {
          failure = closeFailure;
        } else {
          failure.addSuppressed(closeFailure);
        }
      }
    }
    devices.clear();
    if (failure instanceof RuntimeException runtime) {
      throw runtime;
    }
    if (failure instanceof Error error) {
      throw error;
    }
  }
}
