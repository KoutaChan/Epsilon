package com.epsilon.nano.ai.decision.runtime;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.config.settings.DecisionInferenceReplicasSettings;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.nano.ai.network.NetworkDevices;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.nano.config.settings.DecisionInferenceSettings;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/** 学習・評価で使う Decision の推論器を構築し、単一モデルとデバイスごとの複製の管理をまとめる。 */
public final class EpsilonDecisionEvaluatorFactory {

  private static final AtomicInteger NEXT_REPLICA_PRIMARY = new AtomicInteger();

  private EpsilonDecisionEvaluatorFactory() {}

  public static Handle openTrainingActor(
      Model model, Path checkpoint, DecisionExecutionContext context) throws IOException {
    return openTrainingActor(model, checkpoint, context, EpsilonSettings.defaults());
  }

  public static Handle openCheckpointEvaluator(Path checkpoint, DecisionExecutionContext context)
      throws IOException {
    return openCheckpointEvaluator(checkpoint, context, EpsilonSettings.defaults());
  }

  public static Handle openPolicyCheckpointEvaluator(
      Path checkpoint, DecisionExecutionContext context) throws IOException {
    return openPolicyCheckpointEvaluator(checkpoint, context, EpsilonSettings.defaults());
  }

  public static GreedyHandle openGreedyPolicyCheckpointEvaluator(
      Path checkpoint, int maximumBatch, DecisionExecutionContext context) throws IOException {
    return openGreedyPolicyCheckpointEvaluator(
        checkpoint, maximumBatch, context, EpsilonSettings.defaults());
  }

  public static Handle openCheckpointEvaluator(
      Path checkpoint, NetworkDevices devices, DecisionExecutionContext context)
      throws IOException {
    return openCheckpointEvaluator(checkpoint, devices, context, EpsilonSettings.defaults());
  }

  static Handle openPolicyCheckpointEvaluator(
      Path checkpoint, NetworkDevices devices, DecisionExecutionContext context)
      throws IOException {
    return openPolicyCheckpointEvaluator(checkpoint, devices, context, EpsilonSettings.defaults());
  }

  static Handle openModelEvaluator(Model model, DecisionExecutionContext context) {
    return openModelEvaluator(model, context, EpsilonSettings.defaults());
  }

  static Handle openReplicas(
      NetworkDevices devices, ReplicaModelLoader loader, DecisionExecutionContext context)
      throws IOException {
    return openReplicas(devices, loader, context, EpsilonSettings.defaults());
  }

  public static Handle openTrainingActor(
      Model model,
      Path currentCheckpoint,
      DecisionExecutionContext executionContext,
      SettingsLoader config)
      throws IOException {
    if (!replicasEnabled(config)) {
      return openModelEvaluator(model, executionContext, config);
    }
    if (currentCheckpoint != null) {
      return openCheckpointEvaluator(currentCheckpoint, executionContext, config);
    }
    return openInMemoryReplicas(
        model,
        NetworkFactory.getInferenceDevices(config.bind(DeviceSettings.class)),
        executionContext,
        config);
  }

  public static Handle openCheckpointEvaluator(
      Path checkpoint, DecisionExecutionContext executionContext, SettingsLoader config)
      throws IOException {
    NetworkDevices devices =
        replicasEnabled(config)
            ? NetworkFactory.getInferenceDevices(config.bind(DeviceSettings.class))
            : NetworkDevices.of(NetworkFactory.getLearnerDevice(config.bind(DeviceSettings.class)));
    return openCheckpointEvaluator(checkpoint, devices, executionContext, config);
  }

  /** 価値を消費しない対戦相手対局生成用の方策のみ評価器を開く。 */
  public static Handle openPolicyCheckpointEvaluator(
      Path checkpoint, DecisionExecutionContext executionContext, SettingsLoader config)
      throws IOException {
    NetworkDevices devices =
        replicasEnabled(config)
            ? NetworkFactory.getInferenceDevices(config.bind(DeviceSettings.class))
            : NetworkDevices.of(NetworkFactory.getLearnerDevice(config.bind(DeviceSettings.class)));
    return openPolicyCheckpointEvaluator(checkpoint, devices, executionContext, config);
  }

  /** 固定対戦比較用に方策のみモデルの複製と最大確率の行動位置のみを返す出力境界を開く。 */
  public static GreedyHandle openGreedyPolicyCheckpointEvaluator(
      Path checkpoint,
      int maximumInferenceBatch,
      DecisionExecutionContext executionContext,
      SettingsLoader config)
      throws IOException {
    if (maximumInferenceBatch <= 0) {
      throw new IllegalArgumentException("maximumInferenceBatch must be positive");
    }
    NetworkDevices devices =
        replicasEnabled(config)
            ? NetworkFactory.getInferenceDevices(config.bind(DeviceSettings.class))
            : NetworkDevices.of(NetworkFactory.getLearnerDevice(config.bind(DeviceSettings.class)));
    Handle handle =
        openCheckpointEvaluator(
            checkpoint, devices, true, maximumInferenceBatch, executionContext, config);
    if (handle.evaluator() instanceof EpsilonDecisionGreedyEvaluator greedyEvaluator) {
      return new GreedyHandle(handle, greedyEvaluator);
    }
    handle.close();
    throw new IllegalStateException("Policy evaluator does not support compact greedy output");
  }

  public static Handle openCheckpointEvaluator(
      Path checkpoint,
      NetworkDevices devices,
      DecisionExecutionContext executionContext,
      SettingsLoader config)
      throws IOException {
    return openCheckpointEvaluator(checkpoint, devices, false, 0, executionContext, config);
  }

  static Handle openPolicyCheckpointEvaluator(
      Path checkpoint,
      NetworkDevices devices,
      DecisionExecutionContext executionContext,
      SettingsLoader config)
      throws IOException {
    return openCheckpointEvaluator(checkpoint, devices, true, 0, executionContext, config);
  }

  private static Handle openCheckpointEvaluator(
      Path checkpoint,
      NetworkDevices devices,
      boolean policyOnly,
      int duelMaximumBatch,
      DecisionExecutionContext executionContext,
      SettingsLoader config)
      throws IOException {
    Path resolved = EpsilonDecisionCheckpointManager.resolveExistingStrict(checkpoint);
    if (resolved == null) {
      throw new IOException("Decision checkpoint not found: " + checkpoint);
    }
    return openReplicas(
            devices,
            device -> EpsilonDecisionCheckpointManager.load(resolved, device, false),
            policyOnly,
            duelMaximumBatch,
            executionContext,
            config)
        .withCheckpointPath(resolved);
  }

  static Handle openModelEvaluator(
      Model model, DecisionExecutionContext executionContext, SettingsLoader config) {
    DecisionInferenceSettings settings = config.bind(DecisionInferenceSettings.class);
    EpsilonDecisionInferenceServer server =
        EpsilonDecisionInferenceServer.forModel(model, settings.maxBatch(), settings);
    try {
      EpsilonDecisionEvaluator evaluator =
          new EpsilonDecisionInferenceCoordinator(
              new EpsilonDecisionInferenceServer[] {server},
              0,
              executionContext,
              settings.slotsPerDevice(),
              settings.readyBatchesPerDevice());
      return new Handle(evaluator, evaluator, 1, "[" + model.getNDManager().getDevice() + "]");
    } catch (RuntimeException | Error failure) {
      try {
        server.close();
      } catch (RuntimeException | Error closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }
  }

  /** 新しく生成した方策モデルパラメーターを各デバイスが所有する独立した凍結モデルへ直接複製する。 */
  private static Handle openInMemoryReplicas(
      Model model,
      NetworkDevices devices,
      DecisionExecutionContext executionContext,
      SettingsLoader config)
      throws IOException {
    return openReplicas(
        devices,
        device -> EpsilonDecisionModelCopies.copyFrozenToDevice(model, device),
        executionContext,
        config);
  }

  static Handle openReplicas(
      NetworkDevices devices,
      ReplicaModelLoader loader,
      DecisionExecutionContext executionContext,
      SettingsLoader config)
      throws IOException {
    return openReplicas(devices, loader, false, 0, executionContext, config);
  }

  private static Handle openReplicas(
      NetworkDevices devices,
      ReplicaModelLoader loader,
      boolean policyOnly,
      int duelMaximumBatch,
      DecisionExecutionContext executionContext,
      SettingsLoader config)
      throws IOException {
    if (loader == null) {
      throw new IllegalArgumentException("Decision replica model loader must not be null");
    }
    int maxBatch = config.bind(DecisionInferenceSettings.class).maxBatch();
    DecisionInferenceSettings inferenceSettings = config.bind(DecisionInferenceSettings.class);
    int executionSlots = inferenceSettings.slotsPerDevice();
    ArrayList<Model> models = new ArrayList<>(devices.size());
    EpsilonDecisionInferenceServer[] servers = new EpsilonDecisionInferenceServer[devices.size()];
    EpsilonDecisionEvaluator evaluator = null;
    try {
      for (int i = 0; i < devices.size(); i++) {
        models.add(loadPreparedReplica(loader, devices.get(i), inferenceSettings));
      }
      for (int index = 0; index < models.size(); index++) {
        Model replica = models.get(index);
        servers[index] =
            duelMaximumBatch > 0
                ? EpsilonDecisionInferenceServer.forFrozenDuelPolicyModel(
                    replica,
                    maxBatch,
                    executionSlots,
                    duelMaximumBatch,
                    inferenceSettings,
                    config.bind(DecisionInferenceFusionSettings.class))
                : policyOnly
                    ? EpsilonDecisionInferenceServer.forPolicySession(
                        replica,
                        inferenceSettings,
                        config.bind(DecisionInferenceFusionSettings.class))
                    : EpsilonDecisionInferenceServer.forFrozenModel(
                        replica,
                        maxBatch,
                        inferenceSettings,
                        config.bind(DecisionInferenceFusionSettings.class));
      }
      evaluator =
          new EpsilonDecisionInferenceCoordinator(
              servers,
              NEXT_REPLICA_PRIMARY.getAndIncrement(),
              executionContext,
              inferenceSettings.slotsPerDevice(),
              inferenceSettings.readyBatchesPerDevice());
      return new Handle(
          evaluator, new ReplicaResources(evaluator, models), devices.size(), devices.toString());
    } catch (RuntimeException | IOException | Error failure) {
      boolean modelsCanClose =
          evaluator == null
              ? closeServersAfterFailure(servers, failure)
              : closeEvaluatorAfterFailure(evaluator, failure);
      if (modelsCanClose) {
        closeModelsAfterFailure(models, failure);
      }
      throw failure;
    }
  }

  public static boolean replicasEnabled() {
    return replicasEnabled(EpsilonSettings.defaults());
  }

  public static boolean replicasEnabled(SettingsLoader config) {
    return config.bind(DecisionInferenceReplicasSettings.class).enabled();
  }

  /** 読み込み処理所有権を受け取り、固定と型変換の完了後だけ呼び出し側へモデルの複製を移す。 */
  private static Model loadPreparedReplica(
      ReplicaModelLoader loader, Device device, DecisionInferenceSettings inferenceSettings)
      throws IOException {
    return EpsilonDecisionModelCopies.prepareOwnedInferenceModel(
        loader.load(device), inferenceSettings);
  }

  private static boolean closeServersAfterFailure(
      EpsilonDecisionInferenceServer[] servers, Throwable failure) {
    boolean allReleased = true;
    for (int i = servers.length - 1; i >= 0; i--) {
      EpsilonDecisionInferenceServer server = servers[i];
      if (server == null) {
        continue;
      }
      try {
        server.close();
      } catch (RuntimeException | Error cleanupFailure) {
        failure.addSuppressed(cleanupFailure);
      }
      allReleased &= server.resourcesReleased();
    }
    return allReleased;
  }

  private static boolean closeEvaluatorAfterFailure(
      EpsilonDecisionEvaluator evaluator, Throwable failure) {
    if (evaluator == null) {
      return true;
    }
    try {
      evaluator.close();
    } catch (RuntimeException | Error cleanupFailure) {
      failure.addSuppressed(cleanupFailure);
    }
    return replicaModelsCanClose(evaluator);
  }

  private static boolean replicaModelsCanClose(EpsilonDecisionEvaluator evaluator) {
    if (evaluator instanceof EpsilonDecisionInferenceServer server) {
      return server.resourcesReleased();
    }
    if (evaluator instanceof EpsilonDecisionInferenceCoordinator coordinator) {
      return coordinator.resourcesReleased();
    }
    return true;
  }

  private static void closeModelsAfterFailure(List<Model> models, Throwable failure) {
    for (int i = models.size() - 1; i >= 0; i--) {
      try {
        models.get(i).close();
      } catch (RuntimeException | Error cleanupFailure) {
        failure.addSuppressed(cleanupFailure);
      }
    }
  }

  @FunctionalInterface
  interface ReplicaModelLoader {
    Model load(Device device) throws IOException;
  }

  public static final class Handle implements AutoCloseable {
    private final EpsilonDecisionEvaluator evaluator;
    private final AutoCloseable resources;
    private final int shards;
    private final String devices;
    private final Path checkpointPath;

    private Handle(
        EpsilonDecisionEvaluator evaluator, AutoCloseable resources, int shards, String devices) {
      this(evaluator, resources, shards, devices, null);
    }

    private Handle(
        EpsilonDecisionEvaluator evaluator,
        AutoCloseable resources,
        int shards,
        String devices,
        Path checkpointPath) {
      this.evaluator = evaluator;
      this.resources = resources;
      this.shards = shards;
      this.devices = devices;
      this.checkpointPath = checkpointPath;
    }

    public EpsilonDecisionEvaluator evaluator() {
      return evaluator;
    }

    public int shards() {
      return shards;
    }

    public String devices() {
      return devices;
    }

    public Path checkpointPath() {
      return checkpointPath;
    }

    /** 単一の-デバイス診断を行うサーバーを返し、非同期循環バッファなら物理GPU全体を先に完了待ちと回収します。 */
    public EpsilonDecisionInferenceServer singleServerForDiagnostics() {
      if (evaluator instanceof EpsilonDecisionInferenceServer server) {
        return server;
      }
      if (evaluator instanceof EpsilonDecisionInferenceCoordinator coordinator) {
        return coordinator.singleServerForDiagnostics();
      }
      throw new IllegalStateException("Decision evaluator does not expose diagnostic execution");
    }

    /**
     * 固定ホスト側バッチ再生用に、この参照が所有する全デバイスサーバーを返します。
     *
     * <p>返却配列とサーバーは参照から借用します。呼び出し側は配列を書き換えず、サーバーを解放してはいけません。
     */
    public EpsilonDecisionInferenceServer[] serversForReplayBenchmark() {
      if (evaluator instanceof EpsilonDecisionInferenceServer server) {
        server.awaitDeviceIdle();
        return new EpsilonDecisionInferenceServer[] {server};
      }
      if (evaluator instanceof EpsilonDecisionInferenceCoordinator coordinator) {
        return coordinator.serversForReplayBenchmark();
      }
      throw new IllegalStateException("Decision evaluator does not expose replay execution");
    }

    private Handle withCheckpointPath(Path checkpointPath) {
      return new Handle(evaluator, resources, shards, devices, checkpointPath);
    }

    @Override
    public void close() {
      try {
        resources.close();
      } catch (Exception e) {
        throw new IllegalStateException("Failed to close Decision evaluator resources", e);
      }
    }
  }

  /** 方策のみ評価器の資源所有権を保った対戦比較用最大確率の行動位置のみを返す参照。 */
  public static final class GreedyHandle implements AutoCloseable {
    private final Handle delegate;
    private final EpsilonDecisionGreedyEvaluator evaluator;

    private GreedyHandle(Handle delegate, EpsilonDecisionGreedyEvaluator evaluator) {
      this.delegate = delegate;
      this.evaluator = evaluator;
    }

    public EpsilonDecisionGreedyEvaluator evaluator() {
      return evaluator;
    }

    public int shards() {
      return delegate.shards();
    }

    public String devices() {
      return delegate.devices();
    }

    public Path checkpointPath() {
      return delegate.checkpointPath();
    }

    @Override
    public void close() {
      delegate.close();
    }
  }

  private static final class ReplicaResources implements AutoCloseable {
    private final EpsilonDecisionEvaluator evaluator;
    private final List<Model> models;

    private ReplicaResources(EpsilonDecisionEvaluator evaluator, List<Model> models) {
      this.evaluator = evaluator;
      this.models = models;
    }

    @Override
    public void close() {
      Throwable failure = null;
      try {
        evaluator.close();
      } catch (RuntimeException | Error e) {
        failure = e;
      }
      if (replicaModelsCanClose(evaluator)) {
        for (int i = models.size() - 1; i >= 0; i--) {
          try {
            models.get(i).close();
          } catch (RuntimeException | Error e) {
            if (failure == null) {
              failure = e;
            } else {
              failure.addSuppressed(e);
            }
          }
        }
      }
      if (failure != null) {
        if (failure instanceof RuntimeException runtimeException) {
          throw runtimeException;
        }
        throw (Error) failure;
      }
    }
  }
}
