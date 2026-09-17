package com.epsilon.pico.ai.decision.runtime;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.pico.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.pico.ai.decision.input.DecisionBucket;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import com.epsilon.pico.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.pico.ai.network.NetworkFactory;
import com.epsilon.pico.config.settings.DecisionInferenceSettings;
import com.epsilon.pico.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.spi.BatchedPolicy;
import com.epsilon.spi.DecisionRequest;
import com.epsilon.spi.PolicyExecutionContext;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/** 重みを固定した一つのモデルで、系列固有の入力生成から行動選択までを実行する。 */
public final class NativePolicy implements BatchedPolicy {
  private final PolicyExecutionContext execution;
  private final Model[] models;
  private final EpsilonDecisionInferenceCoordinator coordinator;
  private final DecisionBatchBuilder.EncodingWorkspacePool encodingWorkspaces;
  private boolean closed;
  private long timedBatches, encoderQueueNanos, encodingNanos, inferenceRoundTripNanos;

  public static NativePolicy open(
      Path checkpoint, Map<String, String> options, PolicyExecutionContext execution) {
    Map<String, String> values = new LinkedHashMap<>(options);
    String settingPath = values.remove("settings");
    String deviceText = values.remove("devices");
    String singleDevice = values.remove("device");
    if (deviceText == null) deviceText = singleDevice;
    SettingsLoader settings;
    try {
      settings = EpsilonSettings.load(settingPath == null ? null : Path.of(settingPath), values);
    } catch (IOException failure) {
      throw new UncheckedIOException("Cannot load policy settings", failure);
    }
    DecisionInferenceSettings inference = settings.bind(DecisionInferenceSettings.class);
    DecisionInferenceFusionSettings fusion = settings.bind(DecisionInferenceFusionSettings.class);
    DeviceSettings deviceSettings = settings.bind(DeviceSettings.class);
    if (deviceText != null) {
      String selected = deviceText.strip().equalsIgnoreCase("gpu") ? "gpu:0" : deviceText;
      deviceSettings =
          new DeviceSettings(
              deviceSettings.validateIndices(),
              deviceSettings.learner(),
              selected,
              deviceSettings.grp());
    }
    Device[] devices =
        NetworkFactory.getInferenceDevices(deviceSettings).asList().toArray(Device[]::new);
    Model[] models = new Model[devices.length];
    EpsilonDecisionInferenceServer[] servers = new EpsilonDecisionInferenceServer[devices.length];
    try {
      for (int i = 0; i < devices.length; i++) {
        models[i] =
            EpsilonDecisionCheckpointManager.loadForInference(checkpoint, devices[i], inference);
        servers[i] =
            EpsilonDecisionInferenceServer.forPolicySession(
                models[i], inference, fusion, execution.slotsPerDevice());
      }
      return new NativePolicy(models, servers, execution);
    } catch (IOException failure) {
      closeAfterFailure(models, servers, failure);
      throw new UncheckedIOException("Cannot open pico checkpoint: " + checkpoint, failure);
    } catch (RuntimeException | Error failure) {
      closeAfterFailure(models, servers, failure);
      throw failure;
    }
  }

  private NativePolicy(
      Model[] models, EpsilonDecisionInferenceServer[] servers, PolicyExecutionContext execution) {
    this.execution = execution;
    this.models = models;
    coordinator =
        new EpsilonDecisionInferenceCoordinator(
            servers,
            0,
            execution.devices(),
            execution.slotsPerDevice(),
            execution.readyBatchesPerDevice());
    encodingWorkspaces =
        new DecisionBatchBuilder.EncodingWorkspacePool(servers.length * execution.slotsPerDevice());
  }

  @Override
  public Object batchKey(DecisionRequest request) {
    return DecisionBatchBuilder.selectInferenceBucket(request.legalActions());
  }

  @Override
  public int maxBatchSize(Object key) {
    return coordinator.preferredStreamingBatchSize((DecisionBucket) key);
  }

  @Override
  public boolean needsInferenceWork() {
    return coordinator.needsInferenceWork();
  }

  /** ホスト容量と通知登録はCommonに委ね、ここでは入力型だけを保持する。 */
  @Override
  public synchronized Ingress tryAcquire(Object key, Runnable capacityAvailable) {
    if (closed) throw new IllegalStateException("Policy is closed");
    var attempt = coordinator.tryAcquireInferenceIngress();
    if (!attempt.acquired()) {
      attempt.available().whenComplete((ignored, failure) -> capacityAvailable.run());
      return null;
    }
    return new Reservation((DecisionBucket) key, attempt.ingress());
  }

  private final class Reservation implements Ingress {
    private final DecisionBucket bucket;
    private final DecisionInferenceIngress ingress;
    private boolean consumed;

    Reservation(DecisionBucket bucket, DecisionInferenceIngress ingress) {
      this.bucket = bucket;
      this.ingress = ingress;
    }

    @Override
    public CompletableFuture<int[]> submit(List<DecisionRequest> requests) {
      synchronized (this) {
        if (consumed) throw new IllegalStateException("Inference reservation was already consumed");
        consumed = true;
      }
      CompletableFuture<DecisionHostBatch> encoded;
      BatchTiming timing = new BatchTiming();
      try {
        if (requests.isEmpty() || requests.size() > maxBatchSize(bucket)) {
          throw new IllegalArgumentException(
              "Inference batch is empty or exceeds its admitted capacity");
        }
        var builder = DecisionBatchBuilder.inference(requests.size(), bucket, encodingWorkspaces);
        encoded =
            execution
                .encodeRows(
                    requests.size(),
                    (from, to) -> {
                      try (var session = builder.openEncoding()) {
                        for (int row = from; row < to; row++) {
                          DecisionRequest request = requests.get(row);
                          session.encodeObservedInferenceRow(
                              row, request.observation(), request.legalActions());
                        }
                      }
                    },
                    () -> timing.encodingStarted = System.nanoTime())
                .thenApply(ignored -> builder.buildEncodedInferenceRows());
      } catch (RuntimeException | Error failure) {
        ingress.abort();
        throw failure;
      }
      // 呼出元のcancelは、エンコーダー・ネイティブ処理の回収と予約の寿命を変更しない。
      CompletableFuture<int[]> result = new CompletableFuture<>();
      encoded.whenComplete(
          (batch, encodeFailure) -> {
            if (encodeFailure != null) {
              ingress.abort();
              result.completeExceptionally(encodeFailure);
              return;
            }
            timing.inferenceStarted = System.nanoTime();
            try {
              coordinator
                  .submitGreedyActionSlots(batch, ingress)
                  .whenComplete(
                      (selected, failure) -> {
                        if (failure == null) {
                          timing.completed = System.nanoTime();
                          recordTiming(timing);
                          result.complete(selected);
                        } else result.completeExceptionally(failure);
                      });
            } catch (RuntimeException | Error failure) {
              result.completeExceptionally(failure);
            }
          });
      return result;
    }

    @Override
    public void close() {
      synchronized (this) {
        if (consumed) return;
        consumed = true;
      }
      ingress.abort();
    }
  }

  private static final class BatchTiming {
    final long submitted = System.nanoTime();
    long encodingStarted, inferenceStarted, completed;
  }

  @Override
  public synchronized Timings timings() {
    return new Timings(
        true, timedBatches, encoderQueueNanos, encodingNanos, inferenceRoundTripNanos);
  }

  private synchronized void recordTiming(BatchTiming timing) {
    timedBatches++;
    encoderQueueNanos += timing.encodingStarted - timing.submitted;
    encodingNanos += timing.inferenceStarted - timing.encodingStarted;
    inferenceRoundTripNanos += timing.completed - timing.inferenceStarted;
  }

  @Override
  public void close() {
    synchronized (this) {
      if (closed) return;
      closed = true;
    }
    RuntimeException failure = null;
    try {
      coordinator.close();
    } catch (RuntimeException problem) {
      failure = problem;
    }
    // ネイティブ処理が資源を回収できない場合、参照中のモデルを閉じない。
    if (!coordinator.resourcesReleased()) {
      if (failure != null) throw failure;
      return;
    }
    for (int index = models.length - 1; index >= 0; index--) {
      try {
        models[index].close();
      } catch (RuntimeException problem) {
        if (failure == null) failure = problem;
        else failure.addSuppressed(problem);
      }
    }
    if (failure != null) throw failure;
  }

  private static void closeAfterFailure(
      Model[] models, EpsilonDecisionInferenceServer[] servers, Throwable failure) {
    for (int i = servers.length - 1; i >= 0; i--) {
      try {
        if (servers[i] != null) servers[i].close();
        if (models[i] != null) models[i].close();
      } catch (RuntimeException | Error closeFailure) {
        failure.addSuppressed(closeFailure);
      }
    }
  }
}
