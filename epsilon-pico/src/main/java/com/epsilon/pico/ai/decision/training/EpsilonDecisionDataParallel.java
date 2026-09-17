package com.epsilon.pico.ai.decision.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.types.DataType;
import ai.djl.util.PairList;
import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.config.settings.DecisionTrainingDeviceTransferSchedule;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.pico.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.pico.ai.model.EpsilonDecisionNetwork;
import com.epsilon.pico.ai.model.EpsilonTileRelationEncoder;
import com.epsilon.pico.ai.network.NetworkDevices;
import com.epsilon.pico.ai.network.NetworkFactory;
import com.epsilon.pico.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.training.DataParallelGroup;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 系列固有のモデル構築、パラメーター選択、入力パイプラインを共通実行基盤へ接続する。 */
final class EpsilonDecisionDataParallel
    extends DataParallelGroup<EpsilonDecisionDataParallel.Lane> {
  private static final Logger log = LoggerFactory.getLogger(EpsilonDecisionDataParallel.class);

  private EpsilonDecisionDataParallel(
      List<Lane> lanes,
      ParameterScope scope,
      Model masterModel,
      DataType optimizerDataType,
      DecisionExecutionContext executionContext) {
    super(
        lanes,
        scope,
        masterModel,
        optimizerDataType,
        executionContext,
        EpsilonDecisionDataParallel::includes);
  }

  private static boolean includes(ParameterScope scope, String name) {
    return switch (scope) {
      case ONLINE -> EpsilonDecisionNetwork.isOnlineTrainingParameterName(name);
      case ONLINE_ACTOR -> EpsilonDecisionNetwork.isOnlineActorParameterName(name);
      case ONLINE_VALUE -> EpsilonDecisionNetwork.isValueParameterName(name);
      case PRETRAIN_ALL -> EpsilonDecisionNetwork.isOfflinePretrainingParameterName(name);
      case PRETRAIN_POLICY -> EpsilonDecisionNetwork.isOfflinePolicyParameterName(name);
      case PRETRAIN_VALUE -> EpsilonDecisionNetwork.isOfflineValueParameterName(name);
    };
  }

  private static void prepare(ParameterScope scope, EpsilonDecisionNetwork network) {
    switch (scope) {
      case ONLINE, ONLINE_ACTOR, ONLINE_VALUE -> network.prepareForOnlineTraining();
      case PRETRAIN_ALL -> network.prepareForOfflinePretraining(true, true);
      case PRETRAIN_POLICY -> network.prepareForOfflinePretraining(true, false);
      case PRETRAIN_VALUE -> network.prepareForOfflinePretraining(false, true);
    }
  }

  @FunctionalInterface
  interface LaneWork<T> {
    T run(Lane lane) throws Exception;
  }

  <T> List<T> invoke(List<? extends LaneWork<T>> work) throws IOException {
    return executeWork(
        work.stream().map(task -> (DataParallelGroup.LaneWork<Lane, T>) task::run).toList());
  }

  <T> List<T> invokeAssigned(int[] assignedLanes, List<? extends LaneWork<T>> work)
      throws IOException {
    return executeAssigned(
        assignedLanes,
        work.stream().map(task -> (DataParallelGroup.LaneWork<Lane, T>) task::run).toList());
  }

  /** 本番自己対局による学習器用の学習ワーカー集合を、有効デバイス設定から構築する。 */
  static EpsilonDecisionDataParallel openOnline(
      Model canonicalModel,
      DecisionTensorTransfer transferMode,
      DecisionTrainingDeviceTransferSchedule deviceTransferSchedule,
      DecisionExecutionContext executionContext) {
    return openOnline(
        canonicalModel,
        transferMode,
        deviceTransferSchedule,
        executionContext,
        EpsilonSettings.defaults().bind(DeviceSettings.class));
  }

  static EpsilonDecisionDataParallel openOnline(
      Model canonicalModel,
      DecisionTensorTransfer transferMode,
      DecisionTrainingDeviceTransferSchedule deviceTransferSchedule,
      DecisionExecutionContext executionContext,
      DeviceSettings deviceSettings) {
    NetworkDevices devices = NetworkFactory.getLearnerDevices(deviceSettings);
    if (deviceSettings.learner().isBlank()) {
      devices = canonicalFirst(canonicalModel.getNDManager().getDevice(), devices);
    }
    return openLanes(
        canonicalModel,
        devices,
        ParameterScope.ONLINE,
        transferMode,
        deviceTransferSchedule,
        DataType.FLOAT32,
        executionContext);
  }

  /** CPU 一致性テストなど、明示デバイス上へ単純な自己対局による学習ワーカー集合を構築する。 */
  static EpsilonDecisionDataParallel openOnline(
      Model canonicalModel, NetworkDevices devices, DecisionExecutionContext executionContext) {
    return openLanes(
        canonicalModel,
        devices,
        ParameterScope.ONLINE,
        DecisionTensorTransfer.DIRECT_BUFFER,
        DecisionTrainingDeviceTransferSchedule.ON_DEMAND,
        DataType.FLOAT32,
        executionContext);
  }

  /** FLOAT32・直接参照の転送の保存データによる事前学習ワーカー集合を構築する。 */
  static EpsilonDecisionDataParallel openPretraining(
      Model canonicalModel, NetworkDevices devices, DecisionExecutionContext executionContext) {
    return openPretraining(
        canonicalModel,
        devices,
        DecisionTensorTransfer.DIRECT_BUFFER,
        DataType.FLOAT32,
        executionContext);
  }

  /** 保存データによる事前学習ワーカー集合を、指定したホスト転送方式と実行パラメーター型で構築する。 */
  static EpsilonDecisionDataParallel openPretraining(
      Model canonicalModel,
      NetworkDevices devices,
      DecisionTensorTransfer transferMode,
      DataType modelParameterDataType,
      DecisionExecutionContext executionContext) {
    return openLanes(
        canonicalModel,
        devices,
        ParameterScope.PRETRAIN_ALL,
        transferMode,
        DecisionTrainingDeviceTransferSchedule.ON_DEMAND,
        modelParameterDataType,
        executionContext);
  }

  private static EpsilonDecisionDataParallel openLanes(
      Model canonicalModel,
      NetworkDevices devices,
      ParameterScope scope,
      DecisionTensorTransfer transferMode,
      DecisionTrainingDeviceTransferSchedule deviceTransferSchedule,
      DataType modelParameterDataType,
      DecisionExecutionContext executionContext) {
    if (modelParameterDataType != DataType.FLOAT32 && modelParameterDataType != DataType.BFLOAT16) {
      throw new IllegalArgumentException(
          "Unsupported Decision training parameter data type: " + modelParameterDataType);
    }
    boolean usesMasterWeights = modelParameterDataType != DataType.FLOAT32;
    if (usesMasterWeights && scope != ParameterScope.PRETRAIN_ALL) {
      throw new IllegalArgumentException(
          "Low-precision Decision parameters are only supported by offline pretraining");
    }
    if (usesMasterWeights) {
      requireFloatingParameterDataType(canonicalModel, DataType.FLOAT32, "master model");
    }
    Device canonicalDevice = canonicalModel.getNDManager().getDevice();
    if (!canonicalDevice.equals(devices.primary())) {
      throw new IllegalArgumentException(
          "Canonical training device must be the first learner device: canonical="
              + canonicalDevice
              + " devices="
              + devices);
    }

    ArrayList<Lane> lanes = new ArrayList<>(devices.size());
    try {
      if (usesMasterWeights) {
        for (int laneIndex = 0; laneIndex < devices.size(); laneIndex++) {
          lanes.add(
              copyOwnedTrainingLane(
                  canonicalModel,
                  devices.get(laneIndex),
                  scope,
                  transferMode,
                  deviceTransferSchedule,
                  modelParameterDataType,
                  laneIndex,
                  requiresAdditionalComputeStream(devices, laneIndex),
                  executionContext));
        }
      } else {
        prepare(scope, (EpsilonDecisionNetwork) canonicalModel.getBlock());
        lanes.add(
            new Lane(
                canonicalModel,
                false,
                scope,
                transferMode,
                deviceTransferSchedule,
                0,
                false,
                executionContext));
        for (int i = 1; i < devices.size(); i++) {
          lanes.add(
              copyOwnedTrainingLane(
                  canonicalModel,
                  devices.get(i),
                  scope,
                  transferMode,
                  deviceTransferSchedule,
                  modelParameterDataType,
                  i,
                  requiresAdditionalComputeStream(devices, i),
                  executionContext));
        }
      }
      EpsilonDecisionDataParallel session =
          new EpsilonDecisionDataParallel(
              lanes,
              scope,
              usesMasterWeights ? canonicalModel : null,
              usesMasterWeights ? DataType.FLOAT32 : modelParameterDataType,
              executionContext);
      session.recordParameterReadiness();
      log.info(
          "Decision data parallel: scope={} lanes={} devices={} parameterDataType={} "
              + "optimizerDataType={} tensorTransfer={} deviceTransferSchedule={} "
              + "dedicatedComputeStreams={}",
          scope,
          session.laneCount(),
          session.devices(),
          modelParameterDataType,
          session.optimizerDataType(),
          transferMode,
          deviceTransferSchedule,
          session.dedicatedComputeStreamCount());
      return session;
    } catch (RuntimeException | Error failure) {
      closeLanes(lanes, failure);
      throw failure;
    }
  }

  void finishTrainingInputPass() throws IOException {
    ArrayList<LaneWork<Boolean>> work = new ArrayList<>(lanes.size());
    for (int lane = 0; lane < lanes.size(); lane++) {
      work.add(
          activeLane -> {
            activeLane.trainingInput().finishInputPass();
            return true;
          });
    }
    invoke(work);
  }

  private static NetworkDevices canonicalFirst(Device canonical, NetworkDevices devices) {
    if (devices.contains(canonical)) {
      return devices.withPrimary(canonical);
    }
    throw new IllegalStateException(
        "Canonical training device is not present in learner devices: canonical="
            + canonical
            + " devices="
            + devices);
  }

  private static Lane copyOwnedTrainingLane(
      Model source,
      Device targetDevice,
      ParameterScope scope,
      DecisionTensorTransfer transferMode,
      DecisionTrainingDeviceTransferSchedule deviceTransferSchedule,
      DataType parameterDataType,
      int laneIndex,
      boolean additionalComputeStream,
      DecisionExecutionContext executionContext) {
    Model replica = copyTrainingReplica(source, targetDevice, scope);
    try {
      if (parameterDataType != DataType.FLOAT32) {
        castFloatingParameters(replica, parameterDataType);
      }
      return new Lane(
          replica,
          true,
          scope,
          transferMode,
          deviceTransferSchedule,
          laneIndex,
          additionalComputeStream,
          executionContext);
    } catch (RuntimeException | Error failure) {
      try {
        replica.close();
      } catch (RuntimeException closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }
  }

  private static Model copyTrainingReplica(
      Model source, Device targetDevice, ParameterScope scope) {
    EpsilonDecisionNetwork sourceBlock = (EpsilonDecisionNetwork) source.getBlock();
    Model target =
        NetworkFactory.createDecisionModel(
            targetDevice, false, sourceBlock.hiddenSize(), sourceBlock.utilityProfile());
    try {
      copyParameters(source, target);
      prepare(scope, (EpsilonDecisionNetwork) target.getBlock());
      return target;
    } catch (RuntimeException | Error failure) {
      target.close();
      throw failure;
    }
  }

  static boolean requiresAdditionalComputeStream(NetworkDevices devices, int laneIndex) {
    Device device = devices.get(laneIndex);
    if (!device.isGpu()) {
      return false;
    }
    for (int index = 0; index < laneIndex; index++) {
      if (device.equals(devices.get(index))) {
        return true;
      }
    }
    return false;
  }

  static final class TrainingInputSequence implements AutoCloseable {
    private final DecisionTrainingInputPipeline pipeline;
    private final List<DecisionTrainingMicroBatch> batches;
    private final DecisionPolicySignalMultipliers multipliers;
    private final ArrayDeque<CompletableFuture<DecisionTrainingReadyBatch>> futures =
        new ArrayDeque<>(DecisionTrainingInputPipeline.CAPACITY);
    private int submitted;
    private int consumed;
    private boolean closed;

    private TrainingInputSequence(
        DecisionTrainingInputPipeline pipeline,
        List<DecisionTrainingMicroBatch> batches,
        DecisionPolicySignalMultipliers multipliers) {
      this.pipeline = pipeline;
      this.batches = batches;
      this.multipliers = multipliers;
      try {
        fill();
      } catch (RuntimeException | Error failure) {
        try {
          close();
        } catch (IOException closeFailure) {
          failure.addSuppressed(closeFailure);
        }
        throw failure;
      }
    }

    /** 次バッチを返し、直前バッチが解放した位置へ次のlookaheadを補充する。 */
    PreparedTrainingInput next() throws IOException {
      requireOpen();
      try {
        fill();
      } catch (IllegalStateException terminalPipeline) {
        if (futures.isEmpty() || terminalPipeline.getCause() == null) {
          throw terminalPipeline;
        }
        // 先行prepareの失敗はキュー順に受け取り、元の例外型をそのまま返す。
      }
      if (futures.isEmpty()) {
        return null;
      }
      DecisionTrainingMicroBatch batch = batches.get(consumed);
      CompletableFuture<DecisionTrainingReadyBatch> future = futures.getFirst();
      try {
        DecisionTrainingReadyBatch ready = future.get();
        futures.removeFirst();
        consumed++;
        return new PreparedTrainingInput(batch, ready);
      } catch (InterruptedException failure) {
        Thread.currentThread().interrupt();
        throw new IOException("Interrupted while preparing a Decision training batch", failure);
      } catch (CancellationException failure) {
        futures.removeFirst();
        consumed++;
        throw new IOException("Decision training batch preparation was cancelled", failure);
      } catch (ExecutionException failure) {
        futures.removeFirst();
        consumed++;
        throw propagate(failure.getCause());
      }
    }

    private void fill() {
      while (futures.size() < DecisionTrainingInputPipeline.CAPACITY
          && submitted < batches.size()) {
        DecisionTrainingMicroBatch batch = batches.get(submitted++);
        futures.addLast(
            pipeline.submit(new DecisionTrainingBatchPlan(batch.samples(), multipliers)));
      }
    }

    /** 未使用lookaheadを完了まで待ち、ホスト側の実行枠と転送処理の識別情報をパイプラインへ返す。 */
    @Override
    public synchronized void close() throws IOException {
      if (closed) {
        return;
      }
      Throwable failure = null;
      boolean interrupted = Thread.interrupted();
      while (!futures.isEmpty()) {
        CompletableFuture<DecisionTrainingReadyBatch> future = futures.removeFirst();
        DecisionTrainingReadyBatch ready = null;
        boolean finished = false;
        while (!finished) {
          try {
            ready = future.get();
            finished = true;
          } catch (InterruptedException waitFailure) {
            interrupted = true;
          } catch (CancellationException cancelled) {
            failure = addFailure(failure, cancelled);
            finished = true;
          } catch (ExecutionException preparationFailure) {
            failure = addFailure(failure, preparationFailure.getCause());
            finished = true;
          }
        }
        if (ready != null) {
          try {
            ready.close();
          } catch (RuntimeException | Error closeFailure) {
            failure = addFailure(failure, closeFailure);
          }
        }
      }
      closed = true;
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
      if (failure != null) {
        throw propagate(failure);
      }
    }

    private static Throwable addFailure(Throwable primary, Throwable next) {
      if (primary == null) {
        return next;
      }
      if (primary != next) {
        primary.addSuppressed(next);
      }
      return primary;
    }

    private void requireOpen() {
      if (closed) {
        throw new IllegalStateException("Decision training input sequence is closed");
      }
    }
  }

  record PreparedTrainingInput(
      DecisionTrainingMicroBatch microBatch, DecisionTrainingReadyBatch ready) {}

  static final class Lane extends DataParallelGroup.Lane {
    private final DecisionTrainingInputPipeline trainingInput;
    private final DecisionBatchTransfer.Workspace pretrainingTransfer;
    private final DecisionTensorTransfer pretrainingTransferMode;
    private final PairList<String, Object> runtimeParameters;
    private final EpsilonDecisionHlGauss.DeviceConstants valueConstants;

    private Lane(
        Model model,
        boolean ownsModel,
        ParameterScope scope,
        DecisionTensorTransfer transferMode,
        DecisionTrainingDeviceTransferSchedule deviceTransferSchedule,
        int laneIndex,
        boolean additionalComputeStream,
        DecisionExecutionContext executionContext) {
      super(model, ownsModel, additionalComputeStream, executionContext);
      if (scope.usesOnlineTrainingInput()) {
        trainingInput =
            new DecisionTrainingInputPipeline(
                laneIndex,
                model.getNDManager(),
                transferMode,
                deviceTransferSchedule,
                executionContext,
                ((EpsilonDecisionNetwork) model.getBlock()).utilityProfile());
        pretrainingTransfer = null;
        pretrainingTransferMode = null;
      } else {
        trainingInput = null;
        pretrainingTransfer = new DecisionBatchTransfer.Workspace(model.getNDManager());
        pretrainingTransferMode = transferMode;
      }
      runtimeParameters = EpsilonTileRelationEncoder.createRuntimeParameters(model.getNDManager());
      valueConstants =
          new EpsilonDecisionHlGauss.DeviceConstants(
              model.getNDManager(), ((EpsilonDecisionNetwork) model.getBlock()).utilityProfile());
      runtimeParameters.add(EpsilonDecisionHlGauss.DEVICE_CONSTANTS, valueConstants);
    }

    DecisionTrainingInputPipeline trainingInput() {
      if (trainingInput == null) {
        throw new IllegalStateException("Pretraining lane has no online input pipeline");
      }
      return trainingInput;
    }

    TrainingInputSequence openTrainingInputs(
        List<DecisionTrainingMicroBatch> batches, DecisionPolicySignalMultipliers multipliers) {
      return new TrainingInputSequence(trainingInput(), batches, multipliers);
    }

    DecisionBatchTransfer.Workspace pretrainingTransfer() {
      if (pretrainingTransfer == null) {
        throw new IllegalStateException("Online lane has no pretraining transfer workspace");
      }
      return pretrainingTransfer;
    }

    DecisionTensorTransfer pretrainingTransferMode() {
      if (pretrainingTransferMode == null) {
        throw new IllegalStateException("Online lane has no pretraining transfer mode");
      }
      return pretrainingTransferMode;
    }

    PairList<String, Object> runtimeParameters() {
      return runtimeParameters;
    }

    EpsilonDecisionHlGauss.DeviceConstants valueConstants() {
      return valueConstants;
    }

    @Override
    protected void closeInputs() throws Exception {
      try {
        if (trainingInput != null) trainingInput.close();
      } finally {
        if (pretrainingTransfer != null) pretrainingTransfer.close();
      }
    }

    @Override
    protected void closeRuntime() {
      valueConstants.close();
    }
  }
}
