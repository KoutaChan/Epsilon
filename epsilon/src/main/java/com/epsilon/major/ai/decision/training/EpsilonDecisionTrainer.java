package com.epsilon.major.ai.decision.training;

import ai.djl.Model;
import ai.djl.engine.Autocast;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.config.settings.DecisionTrainSettings;
import com.epsilon.config.settings.DecisionTrainingDeviceTransferSchedule;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.major.ai.decision.data.EpsilonDecisionDataException;
import com.epsilon.major.ai.decision.data.EpsilonDecisionFragmentStore;
import com.epsilon.major.ai.decision.data.EpsilonDecisionSample;
import com.epsilon.major.ai.decision.data.EpsilonDecisionTrainingSampleDescriptor;
import com.epsilon.major.ai.decision.data.EpsilonDecisionTrainingSampleDescriptorReader;
import com.epsilon.major.ai.decision.input.DecisionDeviceBatch;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionAutocast;
import com.epsilon.major.ai.decision.training.DecisionTrainingMetrics.Accumulator;
import com.epsilon.major.ai.decision.training.DecisionTrainingMetrics.AdvantageSigns;
import com.epsilon.major.ai.decision.training.DecisionTrainingMetrics.Batch;
import com.epsilon.major.ai.decision.training.DecisionTrainingMetrics.BatchRead;
import com.epsilon.major.ai.decision.training.DecisionTrainingMetrics.PendingBatchRead;
import com.epsilon.major.ai.model.EpsilonDecisionLoss;
import com.epsilon.major.ai.model.EpsilonDecisionNetwork;
import com.epsilon.major.ai.model.EpsilonDecisionOutput;
import com.epsilon.major.ai.policy.EpsilonTrainingConstants;
import com.epsilon.major.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Decision の選択行動に対する方策勾配と価値損失を計算し、モデルを更新する。
 *
 * <p>自己対局の方策は選択行動へ探索分の学習への寄与 {@code alpha + (1-alpha) * piRollout/muBehavior} を残し、{@code
 * piCurrent/piRollout} だけをPPO クリップします。対局生成 KLは損失ではなく、各更新前の診断と棄却判定に使います。方策はFULL_POLICY（方策ネットワーク部分 +
 * 選択肢採点器 + 二択分岐ヘッド）を所有し、価値 ネットワーク部分/ヘッドとは独立したAdamW 状態と更新回数で更新します。
 */
public final class EpsilonDecisionTrainer implements AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(EpsilonDecisionTrainer.class);
  private static final long SHUFFLE_SEED = 0x5345_4c45_4354_5047L;
  private static final String GENERIC_LEARNER_CONTRACT = "decision-online-generic-v1";

  private final Model model;
  private final SettingsLoader config;
  private final com.epsilon.major.config.settings.DecisionSettings optimizerSettings;
  private final DecisionExecutionContext executionContext;
  private final Optimizer actorOptimizer;
  private final Optimizer valueOptimizer;
  private final Tracker actorLearningRate;
  private final float valueLearningRate;
  private final String learnerContractId;
  private final DecisionComputePrecision computePrecision;
  private final DecisionTensorTransfer tensorTransfer;
  private final DecisionTrainingDeviceTransferSchedule deviceTransferSchedule;
  private EpsilonDecisionDataParallel dataParallel;
  private boolean closed;

  /**
   * 標準学習率で方策・価値独立オプティマイザーを構築する。
   *
   * @param model 更新対象 Decision モデル
   */
  public EpsilonDecisionTrainer(Model model, DecisionExecutionContext executionContext) {
    this(
        model,
        EpsilonTrainingConstants.LEARNING_RATE,
        EpsilonTrainingConstants.LEARNING_RATE,
        executionContext);
  }

  /**
   * 方策・価値それぞれの学習率で独立オプティマイザーを構築する。
   *
   * @param model 更新対象 Decision モデル
   * @param actorLearningRate 方策 AdamWの初期学習率
   * @param valueLearningRate 価値 AdamWの固定学習率
   */
  public EpsilonDecisionTrainer(
      Model model,
      float actorLearningRate,
      float valueLearningRate,
      DecisionExecutionContext executionContext) {
    this(
        model,
        Tracker.fixed(actorLearningRate),
        valueLearningRate,
        GENERIC_LEARNER_CONTRACT,
        executionContext);
  }

  /**
   * 方策学習制御が所有するLRを直接参照し、固定価値 LRと独立したAdamWを構築する。
   *
   * @param model 更新対象 Decision モデル
   * @param actorLearningRate 方策 AdamWが直接参照する学習制御、または固定LR
   * @param valueLearningRate 価値 AdamWの固定学習率
   * @param learnerContractId 方策損失契約の互換性識別子
   */
  EpsilonDecisionTrainer(
      Model model,
      Tracker actorLearningRate,
      float valueLearningRate,
      String learnerContractId,
      DecisionExecutionContext executionContext) {
    this(
        model,
        actorLearningRate,
        valueLearningRate,
        learnerContractId,
        executionContext,
        EpsilonSettings.defaults());
  }

  EpsilonDecisionTrainer(
      Model model,
      Tracker actorLearningRate,
      float valueLearningRate,
      String learnerContractId,
      DecisionExecutionContext executionContext,
      SettingsLoader config) {
    this.config = config;
    this.optimizerSettings = config.bind(com.epsilon.major.config.settings.DecisionSettings.class);
    float initialActorLearningRate = actorLearningRate.getNewValue(0);
    if (!Float.isFinite(initialActorLearningRate) || initialActorLearningRate <= 0.0f) {
      throw new IllegalArgumentException("Decision Actor learning rate must be positive");
    }
    if (!Float.isFinite(valueLearningRate) || valueLearningRate <= 0.0f) {
      throw new IllegalArgumentException("Decision Value learning rate must be positive");
    }
    if (learnerContractId == null || learnerContractId.isBlank()) {
      throw new IllegalArgumentException("Decision learnerContractId must not be blank");
    }
    this.model = requireDecisionModel(model);
    if (((EpsilonDecisionNetwork) model.getBlock()).utilityProfile()
        != optimizerSettings.utilityProfile()) {
      throw new IllegalArgumentException(
          "Decision training utility profile differs from checkpoint");
    }
    this.executionContext = executionContext;
    this.actorLearningRate = actorLearningRate;
    this.valueLearningRate = valueLearningRate;
    this.learnerContractId = learnerContractId;
    actorOptimizer = createOptimizer(actorLearningRate);
    valueOptimizer = createOptimizer(Tracker.fixed(valueLearningRate));
    DecisionTrainSettings settings = config.bind(DecisionTrainSettings.class);
    computePrecision = settings.computePrecision();
    tensorTransfer = settings.tensorTransfer();
    deviceTransferSchedule = settings.deviceTransferSchedule();
  }

  public EpsilonDecisionTrainer(
      Model model, DecisionExecutionContext executionContext, SettingsLoader config) {
    this(
        model,
        Tracker.fixed(
            config.bind(com.epsilon.major.config.settings.DecisionSettings.class).learningRate()),
        config.bind(com.epsilon.major.config.settings.DecisionSettings.class).learningRate(),
        GENERIC_LEARNER_CONTRACT,
        executionContext,
        config);
  }

  float weightDecay() {
    return optimizerSettings.weightDecay();
  }

  float gradientClip() {
    return optimizerSettings.gradClip();
  }

  /** 方策・価値それぞれのAdamW 移動平均と更新回数を継続学習一組のデータへ保存する。 */
  void saveOptimizerStates(Path checkpoint) throws IOException {
    ensureOpen();
    actorOptimizer.saveState(DecisionLearnerCheckpoint.actorState(checkpoint));
    valueOptimizer.saveState(DecisionLearnerCheckpoint.valueState(checkpoint));
  }

  /** 検証済み更新中の一組のデータから方策・価値 AdamW 状態を復元する。 */
  void loadOptimizerStates(Path checkpoint) throws IOException {
    ensureOpen();
    actorOptimizer.loadState(
        model.getNDManager(), DecisionLearnerCheckpoint.actorState(checkpoint));
    valueOptimizer.loadState(
        model.getNDManager(), DecisionLearnerCheckpoint.valueState(checkpoint));
  }

  float actorLearningRate() {
    return actorLearningRate.getNewValue(0);
  }

  float valueLearningRate() {
    return valueLearningRate;
  }

  String learnerContractId() {
    return learnerContractId;
  }

  /**
   * 本番選択行動 FULL_POLICY と独立価値ネットワーク部分/ヘッドを1回の逐次処理順伝播で更新する。
   *
   * <p>逆伝播は一度だけ行うがパラメーター所有権は交差しない。正規化後に方策用AdamWと価値用AdamWへ別々に確定し、オプティマイザー 状態と更新回数を共有しない。
   *
   * @param fragmentPaths 入力元一巡の処理を構成する学習データ片パス
   * @param reader 学習データ片から軽量なサンプル記述情報を復元する読み取り処理
   * @param prefetchDepth 読み取り処理が先行保持する学習データ片数
   * @param prefetchWorkers 学習データ片を並列読込するワーカー数
   * @param configuredSamples 入力元一巡の処理に含まれることを事前確認したサンプル数
   * @param policyPlan 小バッチ、検証条件、エントロピー、パラメーター更新の実行計画
   * @return 同じ物理更新段階から集約した方策・価値指標
   * @throws IOException 学習データ片の読込に失敗した場合
   */
  public DecisionOnlineTrainingResult trainProductionSelectedPolicyAndValueStreaming(
      List<Path> fragmentPaths,
      EpsilonDecisionTrainingSampleDescriptorReader reader,
      int prefetchDepth,
      int prefetchWorkers,
      int configuredSamples,
      DecisionOnlineTrainingPlan policyPlan)
      throws IOException {
    ensureOpen();
    DecisionOnlineLossConfig config =
        DecisionOnlineLossConfig.create(
            policyPlan.policyUpdateClipRange(),
            policyPlan.explorationCreditMix(),
            policyPlan.entropyCoefficient());
    return trainAccumulatedActorAndValueStreaming(
        fragmentPaths,
        reader,
        prefetchDepth,
        prefetchWorkers,
        config,
        configuredSamples,
        TrainingExecution.actor(policyPlan),
        policyPlan.microBatchSize(),
        policyPlan.maximumDeviceTransitionCells(),
        policyPlan.ppoEpochs(),
        policyPlan.optimizerStepsPerEpoch());
  }

  private DecisionOnlineTrainingResult trainAccumulatedActorAndValueStreaming(
      List<Path> fragmentPaths,
      EpsilonDecisionTrainingSampleDescriptorReader reader,
      int prefetchDepth,
      int prefetchWorkers,
      DecisionOnlineLossConfig config,
      int configuredSamples,
      TrainingExecution actorExecution,
      int requestedBatchSize,
      int maximumDeviceTransitionCells,
      int ppoEpochs,
      int optimizerStepsPerEpoch)
      throws IOException {
    ensureOpen();
    if (configuredSamples <= 0) {
      throw new IllegalArgumentException(
          "configuredSamples must be positive for a fused Actor/Value macro");
    }
    if (optimizerStepsPerEpoch > configuredSamples) {
      throw new IllegalArgumentException(
          "optimizerStepsPerEpoch exceeds configuredSamples: steps="
              + optimizerStepsPerEpoch
              + " configuredSamples="
              + configuredSamples);
    }
    int totalOptimizerSteps = Math.multiplyExact(ppoEpochs, optimizerStepsPerEpoch);
    if (fragmentPaths.isEmpty()) {
      throw new IllegalArgumentException("fused Actor/Value macro requires at least one fragment");
    }
    if (prefetchDepth <= 0 || prefetchWorkers <= 0) {
      throw new IllegalArgumentException("prefetch depth and workers must be positive");
    }
    long fusedStartedAt = System.nanoTime();
    TrainingRun run =
        new TrainingRun(config, configuredSamples, prefetchDepth, prefetchWorkers, actorExecution);
    EpsilonDecisionDataParallel dataParallel = dataParallel();
    dataParallel.clearTrainingGradients();
    Throwable trainingFailure = null;
    float meanActorUpdateKl;
    try {
      DecisionActorUpdateKlProbe probe =
          DecisionActorUpdateKlProbe.select(
              fragmentPaths,
              reader,
              requestedBatchSize,
              maximumDeviceTransitionCells,
              dataParallel.laneCount(),
              actorExecution.policySignalMultipliers(),
              computePrecision);
      probe.captureBefore(dataParallel);
      log.info(
          "Decision online microbatch scheduler: globalRows={} lanes={} maximumRowsPerDevice={} "
              + "maximumDeviceTransitionCells={} tensorTransfer={} deviceTransferSchedule={} "
              + "computePrecision={} ppoEpochs={} "
              + "optimizerStepsPerEpoch={} totalOptimizerSteps={} "
              + "exactDecisionBuckets=true",
          requestedBatchSize,
          dataParallel.laneCount(),
          requestedBatchSize / dataParallel.laneCount(),
          maximumDeviceTransitionCells,
          tensorTransfer,
          deviceTransferSchedule,
          computePrecision,
          ppoEpochs,
          optimizerStepsPerEpoch,
          totalOptimizerSteps);
      for (int epoch = 0; epoch < ppoEpochs && !run.stopped(); epoch++) {
        ArrayList<Path> paths = new ArrayList<>(fragmentPaths);
        Collections.shuffle(paths, new Random(shuffleSeed(epoch, fragmentPaths.size())));
        try (ActorValueStepSource steps =
            new ActorValueStepSource(
                paths,
                reader,
                run,
                epoch,
                configuredSamples,
                optimizerStepsPerEpoch,
                requestedBatchSize,
                maximumDeviceTransitionCells,
                dataParallel.laneCount())) {
          executeActorValueEpoch(
              steps, dataParallel, run, epoch, ppoEpochs, optimizerStepsPerEpoch);
        }
      }
      meanActorUpdateKl = probe.measureAfter(dataParallel);
      log.info(
          "Decision Actor update KL probe: rows={} meanActorUpdateKl={}",
          probe.rows(),
          meanActorUpdateKl);
    } catch (IOException | RuntimeException | Error failure) {
      trainingFailure = failure;
      throw failure;
    } finally {
      finishTrainingPass(dataParallel, trainingFailure);
    }
    double fusedElapsedMillis = (System.nanoTime() - fusedStartedAt) / 1_000_000.0;
    // 方策と価値は順伝播・逆伝播を共有するため、価値だけの順伝播経過時間は計測しない。
    return new DecisionOnlineTrainingResult(
        finish(run, MetricGroup.ACTOR, fusedElapsedMillis, fusedElapsedMillis),
        finish(run, MetricGroup.VALUE, 0.0, fusedElapsedMillis),
        meanActorUpdateKl);
  }

  private static void finishTrainingPass(
      EpsilonDecisionDataParallel dataParallel, Throwable primary) throws IOException {
    Throwable failure = primary;
    try {
      dataParallel.clearTrainingGradients();
    } catch (RuntimeException | Error cleanupFailure) {
      if (failure == null) {
        failure = cleanupFailure;
      } else {
        failure.addSuppressed(cleanupFailure);
      }
    }
    try {
      dataParallel.finishTrainingInputPass();
    } catch (IOException | RuntimeException | Error cleanupFailure) {
      if (failure == null) {
        failure = cleanupFailure;
      } else {
        failure.addSuppressed(cleanupFailure);
      }
    }
    if (primary == null && failure != null) {
      propagateInputFailure(failure);
    }
  }

  private void executeActorValueEpoch(
      ActorValueStepSource steps,
      EpsilonDecisionDataParallel dataParallel,
      TrainingRun run,
      int epoch,
      int ppoEpochs,
      int optimizerStepsPerEpoch)
      throws IOException {
    int completedSteps = 0;
    PlannedActorValueStep step;
    while (!run.stopped() && (step = steps.nextStep()) != null) {
      EpsilonDecisionFusedGradientAccumulator gradientAccumulation =
          new EpsilonDecisionFusedGradientAccumulator(step.configuredStepSamples());
      Accumulator actorStepMetrics = new Accumulator();
      int globalStepIndex =
          Math.addExact(Math.multiplyExact(epoch, optimizerStepsPerEpoch), step.stepIndex());
      accumulateActorAndValueStep(
          step, dataParallel, run, gradientAccumulation, actorStepMetrics, globalStepIndex);
      if (run.stopped()) {
        break;
      }

      completeAccumulatedActorAndValueStep(
          run,
          gradientAccumulation,
          actorStepMetrics,
          epoch,
          ppoEpochs,
          step.stepIndex(),
          optimizerStepsPerEpoch,
          dataParallel);
      if (!run.stopped()) {
        completedSteps++;
      }
    }
    if (!run.stopped() && completedSteps != optimizerStepsPerEpoch) {
      throw new EpsilonDecisionDataException(
          "planned PPO epoch optimizer-step mismatch: completed="
              + completedSteps
              + " expected="
              + optimizerStepsPerEpoch);
    }
  }

  private static int optimizerStepSampleCount(int samples, int steps, int stepIndex) {
    return samples / steps + (stepIndex < samples % steps ? 1 : 0);
  }

  private void completeAccumulatedActorAndValueStep(
      TrainingRun run,
      EpsilonDecisionFusedGradientAccumulator gradientAccumulation,
      Accumulator actorStepMetrics,
      int epochIndex,
      int ppoEpochs,
      int epochStepIndex,
      int optimizerStepsPerEpoch,
      EpsilonDecisionDataParallel dataParallel)
      throws IOException {
    gradientAccumulation.requireCompleteAndValid();
    int totalOptimizerSteps = Math.multiplyExact(ppoEpochs, optimizerStepsPerEpoch);
    int totalStepIndex =
        Math.addExact(Math.multiplyExact(epochIndex, optimizerStepsPerEpoch), epochStepIndex);
    String stepPrefix =
        totalOptimizerSteps == 1
            ? ""
            : "ppoEpoch="
                + (epochIndex + 1)
                + "/"
                + ppoEpochs
                + ",optimizerStep="
                + (totalStepIndex + 1)
                + "/"
                + totalOptimizerSteps
                + ",";
    EpsilonDecisionTrainingRejection rejection =
        validateFusedStep(run, actorStepMetrics, stepPrefix);
    if (rejection.rejected()) {
      run.rejectAccumulatedGradient(rejection);
      return;
    }

    commitFusedStep(
        run,
        gradientAccumulation,
        dataParallel,
        epochIndex,
        ppoEpochs,
        epochStepIndex,
        optimizerStepsPerEpoch,
        actorStepMetrics);
  }

  private EpsilonDecisionTrainingRejection validateFusedStep(
      TrainingRun run, Accumulator actorStepMetrics, String stepPrefix) {
    EpsilonDecisionTrainingRejection rejection =
        run.rolloutPolicyKlRejectionAfterActorStep(actorStepMetrics).withPrefix(stepPrefix);
    if (rejection.rejected()) {
      return rejection;
    }
    return EpsilonDecisionTrainingRejection.PASSED;
  }

  private void commitFusedStep(
      TrainingRun run,
      EpsilonDecisionFusedGradientAccumulator gradientAccumulation,
      EpsilonDecisionDataParallel dataParallel,
      int epochIndex,
      int ppoEpochs,
      int epochStepIndex,
      int optimizerStepsPerEpoch,
      Accumulator actorStepMetrics)
      throws IOException {
    dataParallel.applyFlattenedOnlineOptimizerSteps(
        actorOptimizer,
        gradientAccumulation.actorGradientScale(),
        valueOptimizer,
        gradientAccumulation.valueGradientScale());
    run.recordOptimizerStep();
    int totalOptimizerSteps = Math.multiplyExact(ppoEpochs, optimizerStepsPerEpoch);
    int totalStepIndex =
        Math.addExact(Math.multiplyExact(epochIndex, optimizerStepsPerEpoch), epochStepIndex);
    log.info(
        "Decision independent Actor/Value PPO shard complete: epoch={}/{} epochStep={}/{} "
            + "totalStep={}/{} samples={} actorMass={} valueMass={} meanRolloutPolicyKl={} "
            + "maximumRolloutPolicyKl={}",
        epochIndex + 1,
        ppoEpochs,
        epochStepIndex + 1,
        optimizerStepsPerEpoch,
        totalStepIndex + 1,
        totalOptimizerSteps,
        gradientAccumulation.configuredSamples(),
        gradientAccumulation.actorMass(),
        gradientAccumulation.valueMass(),
        actorStepMetrics.meanRolloutPolicyKl(),
        actorStepMetrics.maximumRolloutPolicyKl());
  }

  private void accumulateActorAndValueStep(
      PlannedActorValueStep step,
      EpsilonDecisionDataParallel dataParallel,
      TrainingRun run,
      EpsilonDecisionFusedGradientAccumulator gradientAccumulation,
      Accumulator actorStepMetrics,
      int globalStepIndex)
      throws IOException {
    DecisionTrainingLanePlan lanePlan =
        DecisionTrainingLanePlan.create(step.batches(), dataParallel.laneCount(), globalStepIndex);
    ActorValueMicroBatchResult[] results =
        new ActorValueMicroBatchResult[lanePlan.microBatchCount()];
    int[] activeLanes = lanePlan.activeLanes();
    ArrayList<EpsilonDecisionDataParallel.LaneWork<Boolean>> work =
        new ArrayList<>(activeLanes.length);
    for (int laneIndex : activeLanes) {
      List<DecisionTrainingMicroBatch> laneBatches = lanePlan.batches(laneIndex);
      work.add(
          lane -> {
            runActorAndValueLane(
                laneBatches,
                results,
                lane,
                run.config(),
                run.execution(),
                step.configuredStepSamples());
            return true;
          });
    }
    dataParallel.invokeAssigned(activeLanes, work);
    for (ActorValueMicroBatchResult result : results) {
      if (run.stopped()) {
        break;
      }
      gradientAccumulation.observeMicroBatch(
          result.count(), result.actorMass(), result.valueMass());
      if (result.scalarAdvantageSigns() != null) {
        run.recordScalarAdvantageSigns(result.scalarAdvantageSigns());
      }
      if (result.rejection().rejected()) {
        EpsilonDecisionTrainingRejection rejection = result.rejection();
        run.rejectMicroBatch(result.count(), result.metrics(), rejection);
        log.warn(
            "Decision fused Actor/Value shard rejected before backward: reason={}",
            rejection.reason());
        break;
      }
      actorStepMetrics.add(result.count(), result.metrics());
      run.recordMicroBatch(result.count(), result.metrics());
    }
  }

  /** 一つの学習ワーカーが更新段階内の割当バッチ列を相手学習ワーカーの進捗を待たずに最後まで処理する。 */
  private void runActorAndValueLane(
      List<DecisionTrainingMicroBatch> batches,
      ActorValueMicroBatchResult[] results,
      EpsilonDecisionDataParallel.Lane lane,
      DecisionOnlineLossConfig config,
      TrainingExecution execution,
      int configuredStepSamples)
      throws IOException {
    try (EpsilonDecisionDataParallel.TrainingInputSequence inputs =
        lane.openTrainingInputs(batches, execution.policySignalMultipliers())) {
      EpsilonDecisionDataParallel.PreparedTrainingInput prepared;
      while ((prepared = inputs.next()) != null) {
        DecisionTrainingMicroBatch batch = prepared.microBatch();
        results[batch.sourceOrdinal()] =
            runActorAndValueMicroBatch(
                prepared.ready(), lane, config, execution, configuredStepSamples);
      }
    }
  }

  private ActorValueMicroBatchResult runActorAndValueMicroBatch(
      DecisionTrainingReadyBatch ready,
      EpsilonDecisionDataParallel.Lane lane,
      DecisionOnlineLossConfig config,
      TrainingExecution execution,
      int configuredStepSamples)
      throws IOException {
    try (ready;
        DecisionTrainingDeviceBatchLease input = lane.trainingInput().acquire(ready);
        GradientCollector collector = lane.newGradientCollector()) {
      DecisionTrainingBatchSummary summary = input.summary();
      int count = summary.rows();
      boolean debugActorValidation = execution.debugActorValidationEnabled();
      AdvantageSigns scalarAdvantageSigns = debugActorValidation ? summary.advantageSigns() : null;
      double actorMass = summary.actorMass();
      double valueMass = summary.valueMass();
      NDManager sub = input.manager();
      DecisionDeviceBatch deviceBatch = input.batch();
      EpsilonDecisionOutput output;
      try (Autocast ignored = EpsilonDecisionAutocast.open(lane.manager(), computePrecision)) {
        output =
            ((EpsilonDecisionNetwork) lane.model().getBlock())
                .forwardDecision(
                    new ParameterStore(sub, true), deviceBatch, true, lane.runtimeParameters());
        NDList rawOutput = output.toNDList();
        sub.attachAll(rawOutput);
      }
      EpsilonDecisionLoss.TrainingLossResult losses =
          EpsilonDecisionLoss.computeOnlineTrainingLoss(
              output,
              deviceBatch,
              config,
              summary.hasActorSignal(),
              debugActorValidation,
              ((EpsilonDecisionNetwork) lane.model().getBlock()).utilityProfile(),
              lane.valueConstants());

      try (PendingBatchRead pendingMetricRead = Batch.enqueueForPolicyGuards(losses, lane);
          NDArray actorObjective = losses.total().mul(actorMass / configuredStepSamples);
          NDArray valueObjective = losses.valueLoss().mul(valueMass / configuredStepSamples);
          NDArray objective = actorObjective.add(valueObjective)) {
        collector.backward(objective);
        BatchRead metricRead = pendingMetricRead.await();
        Batch metrics = metricRead.metrics();
        metricRead.requireFinite();
        return new ActorValueMicroBatchResult(
            count,
            actorMass,
            valueMass,
            metrics,
            scalarAdvantageSigns,
            EpsilonDecisionTrainingRejection.PASSED);
      }
    }
  }

  private static void propagateInputFailure(Throwable failure) throws IOException {
    if (failure instanceof IOException io) {
      throw io;
    }
    if (failure instanceof RuntimeException runtime) {
      throw runtime;
    }
    if (failure instanceof Error error) {
      throw error;
    }
    throw new IOException("Decision training batch preparation failed", failure);
  }

  /** 一つのパラメーター更新が所有する厳密に同じ容量区分小バッチ群。 */
  private record PlannedActorValueStep(
      int stepIndex, int configuredStepSamples, ArrayList<DecisionTrainingMicroBatch> batches) {

    private PlannedActorValueStep {
      if (stepIndex < 0 || configuredStepSamples <= 0 || batches.isEmpty()) {
        throw new IllegalArgumentException("invalid planned Decision Actor/Value step");
      }
    }
  }

  /** 上限付き学習データ片先読みから、一パラメーター更新分の記述情報実行計画を順に生成する。 */
  private static final class ActorValueStepSource implements AutoCloseable {
    private final EpsilonDecisionFragmentPrefetch prefetch;
    private final EpsilonDecisionTrainingMicroBatchQueue microBatches;
    private final TrainingRun run;
    private final int epoch;
    private final int configuredSamples;
    private final int optimizerStepsPerEpoch;
    private final DecisionPolicySignalMultipliers multipliers;

    private ArrayList<EpsilonDecisionTrainingSampleDescriptor> fragmentSamples = new ArrayList<>();
    private int fragmentOffset;
    private int fragmentOrdinal;
    private int stepIndex;
    private boolean sourceExhausted;
    private boolean complete;

    private ActorValueStepSource(
        List<Path> paths,
        EpsilonDecisionTrainingSampleDescriptorReader reader,
        TrainingRun run,
        int epoch,
        int configuredSamples,
        int optimizerStepsPerEpoch,
        int requestedBatchSize,
        int maximumDeviceTransitionCells,
        int laneCount) {
      this.run = run;
      this.epoch = epoch;
      this.configuredSamples = configuredSamples;
      this.optimizerStepsPerEpoch = optimizerStepsPerEpoch;
      multipliers = run.execution().policySignalMultipliers();
      microBatches =
          new EpsilonDecisionTrainingMicroBatchQueue(
              requestedBatchSize, laneCount, maximumDeviceTransitionCells);
      prefetch =
          new EpsilonDecisionFragmentPrefetch(
              paths, reader, run.prefetchDepth(), run.prefetchWorkers());
    }

    private PlannedActorValueStep nextStep() throws IOException {
      if (complete) {
        return null;
      }
      if (stepIndex == optimizerStepsPerEpoch) {
        if (nextDescriptor() != null) {
          throw new EpsilonDecisionDataException(
              "fused Actor/Value PPO epoch source exceeds configuredSamples: epoch="
                  + (epoch + 1)
                  + " configured="
                  + configuredSamples);
        }
        complete = true;
        return null;
      }

      int stepSamples =
          optimizerStepSampleCount(configuredSamples, optimizerStepsPerEpoch, stepIndex);
      ArrayList<DecisionTrainingMicroBatch> batches = new ArrayList<>();
      for (int sampleIndex = 0; sampleIndex < stepSamples; sampleIndex++) {
        EpsilonDecisionTrainingSampleDescriptor sample = nextDescriptor();
        if (sample == null) {
          throw new EpsilonDecisionDataException(
              "fused Actor/Value PPO epoch sample count mismatch: epoch="
                  + (epoch + 1)
                  + " step="
                  + (stepIndex + 1)
                  + " expected="
                  + stepSamples
                  + " actual="
                  + sampleIndex);
        }
        var completed = microBatches.add(sample);
        if (completed.isPresent()) {
          batches.add(
              DecisionTrainingMicroBatch.take(
                  batches.size(), completed.orElseThrow(), multipliers));
        }
      }
      for (ArrayList<EpsilonDecisionTrainingSampleDescriptor> tail : microBatches.drain()) {
        batches.add(DecisionTrainingMicroBatch.take(batches.size(), tail, multipliers));
      }
      return new PlannedActorValueStep(stepIndex++, stepSamples, batches);
    }

    private EpsilonDecisionTrainingSampleDescriptor nextDescriptor() throws IOException {
      while (true) {
        if (fragmentOffset < fragmentSamples.size()) {
          return fragmentSamples.set(fragmentOffset++, null);
        }
        if (sourceExhausted) {
          return null;
        }
        EpsilonDecisionFragmentPrefetch.Fragment fragment = prefetch.next();
        if (fragment == null) {
          sourceExhausted = true;
          return null;
        }
        fragmentSamples = fragment.samples();
        fragmentOffset = 0;
        run.fragmentDescriptorsRead(fragmentSamples.size());
        Collections.shuffle(
            fragmentSamples,
            new Random(
                shuffleSeed(
                    epoch,
                    fragmentOrdinal++,
                    EpsilonDecisionFragmentStore.stableShuffleSalt(fragment.path()))));
      }
    }

    @Override
    public void close() {
      fragmentSamples.clear();
      microBatches.drain();
      prefetch.close();
    }
  }

  private record ActorValueMicroBatchResult(
      int count,
      double actorMass,
      double valueMass,
      Batch metrics,
      AdvantageSigns scalarAdvantageSigns,
      EpsilonDecisionTrainingRejection rejection) {}

  /**
   * 方策重みだけを判断機会別に再重み付けする。
   *
   * <p>価値教師値とサンプル重みには触れません。
   */
  public static void applyProductionActorWeights(
      EpsilonDecisionTrainingBatch batch,
      List<EpsilonDecisionSample> samples,
      int start,
      int count,
      DecisionPolicySignalMultipliers multipliers) {
    for (int row = 0; row < count; row++) {
      EpsilonDecisionPointKind kind =
          EpsilonDecisionPointKind.of(samples.get(start + row).input(), 0);
      batch.multiplyActorWeight(row, multipliers.forKind(kind));
    }
  }

  static double actorWeightMass(EpsilonDecisionTrainingBatch batch) {
    double mass = 0.0;
    for (int row = 0; row < batch.size(); row++) {
      double weight = (double) batch.actorWeight(row) * batch.sampleWeight(row);
      if (!Double.isFinite(weight) || weight < 0.0) {
        throw new EpsilonDecisionDataException(
            "actor weight must be finite and non-negative at row=" + row + ": " + weight);
      }
      mass += weight;
      if (!Double.isFinite(mass)) {
        throw new EpsilonDecisionDataException("actor weight mass overflow at row=" + row);
      }
    }
    return mass;
  }

  static double valueWeightMass(EpsilonDecisionTrainingBatch batch) {
    double mass = 0.0;
    for (int row = 0; row < batch.size(); row++) {
      double weight = batch.sampleWeight(row);
      if (!Double.isFinite(weight) || weight < 0.0) {
        throw new EpsilonDecisionDataException(
            "value weight must be finite and non-negative at row=" + row + ": " + weight);
      }
      mass += weight;
      if (!Double.isFinite(mass)) {
        throw new EpsilonDecisionDataException("value weight mass overflow at row=" + row);
      }
    }
    return mass;
  }

  private DecisionTrainingResult finish(
      TrainingRun run,
      MetricGroup metricGroup,
      double elapsedMillis,
      double fusedTotalElapsedMillis) {
    DecisionTrainingResult metrics =
        run.finish(metricGroup, elapsedMillis, fusedTotalElapsedMillis);
    return logFinished(metricGroup, metrics);
  }

  private DecisionTrainingResult logFinished(
      MetricGroup metricGroup, DecisionTrainingResult metrics) {
    log.info(
        "Decision update: phase={} optimizerSteps={} microBatches={} loss={} actorLoss={}"
            + " entropyBonusLoss={} bcLoss={} valueLoss={} entropy={}"
            + " rolloutEntropy={} behaviorToRolloutPolicyKl={} chosenProb={} behaviorProbMean={}"
            + " behaviorProbMin={} meanRolloutPolicyKl={} maximumRolloutPolicyKl={}"
            + " actorWeight={} rawExplorationRatio={} explorationCreditWeight={}"
            + " policyUpdateRatio={} policyUpdateClipFraction={} effectiveActorRatio={}"
            + " effectiveActorRatioEffectiveSampleFraction={}"
            + " scalarAdvantageSigns=+{}/-{}/0:{} rejected={} reason={} {}",
        metricGroup,
        metrics.optimizerSteps(),
        metrics.microBatches(),
        metrics.loss(),
        metrics.actorLoss(),
        metrics.entropyBonusLoss(),
        metrics.behaviorCloningLoss(),
        metrics.valueLoss(),
        metrics.entropy(),
        metrics.rolloutEntropy(),
        metrics.behaviorToRolloutPolicyKl(),
        metrics.chosenProb(),
        metrics.behaviorProbMean(),
        metrics.behaviorProbMin(),
        metrics.meanRolloutPolicyKl(),
        metrics.maximumRolloutPolicyKl(),
        metrics.actorWeight(),
        metrics.policyRatios().rawExplorationRatio(),
        metrics.policyRatios().explorationCreditWeight(),
        metrics.policyRatios().policyUpdate(),
        metrics.policyRatios().policyUpdateClipFraction(),
        metrics.policyRatios().effectiveActorRatio(),
        metrics.policyRatios().effectiveActorRatioEffectiveSampleFraction(),
        metrics.positiveScalarAdvantageSamples(),
        metrics.negativeScalarAdvantageSamples(),
        metrics.zeroScalarAdvantageSamples(),
        metrics.rejected(),
        metrics.reason(),
        metrics.performance());
    return metrics;
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;
    if (dataParallel != null) {
      dataParallel.close();
      dataParallel = null;
    }
  }

  private record TrainingExecution(
      DecisionPolicyTrustRegion trustRegion,
      DecisionPolicySignalMultipliers policySignalMultipliers,
      boolean debugActorValidationEnabled) {

    private static TrainingExecution actor(DecisionOnlineTrainingPlan plan) {
      return new TrainingExecution(
          plan.trustRegion(), plan.policySignalMultipliers(), plan.debugActorValidationEnabled());
    }
  }

  private enum MetricGroup {
    ACTOR,
    VALUE
  }

  private static final class TrainingRun {
    private final DecisionOnlineLossConfig config;
    private final TrainingExecution execution;
    private final DecisionPolicyTrustRegion trustRegion;
    private final Accumulator actorMetrics = new Accumulator();
    private final Accumulator valueMetrics = new Accumulator();
    private final DecisionTrainingPerformance perf;
    private EpsilonDecisionTrainingRejection rejection = EpsilonDecisionTrainingRejection.PASSED;

    private TrainingRun(
        DecisionOnlineLossConfig config,
        int configuredSamples,
        int prefetchDepth,
        int prefetchWorkers,
        TrainingExecution execution) {
      this.config = config;
      this.execution = execution;
      trustRegion = execution.trustRegion();
      perf = new DecisionTrainingPerformance(configuredSamples, prefetchDepth, prefetchWorkers);
    }

    private DecisionOnlineLossConfig config() {
      return config;
    }

    private TrainingExecution execution() {
      return execution;
    }

    private int prefetchDepth() {
      return perf.prefetchDepth();
    }

    private int prefetchWorkers() {
      return perf.prefetchWorkers();
    }

    private void fragmentDescriptorsRead(int samples) {
      perf.fragmentDescriptorsRead(samples);
    }

    private void recordMicroBatch(int sampleCount, Batch batch) {
      actorMetrics.add(sampleCount, batch);
      valueMetrics.addValue(sampleCount, batch);
      perf.microBatch(sampleCount);
    }

    private void recordOptimizerStep() {
      actorMetrics.addOptimizerStep();
      valueMetrics.addOptimizerStep();
      perf.optimizerStep();
    }

    private void recordScalarAdvantageSigns(AdvantageSigns signs) {
      actorMetrics.addAdvantageSigns(signs);
    }

    private void rejectMicroBatch(
        int sampleCount, Batch batch, EpsilonDecisionTrainingRejection microBatchRejection) {
      recordMicroBatch(sampleCount, batch);
      reject(microBatchRejection);
    }

    private void rejectAccumulatedGradient(EpsilonDecisionTrainingRejection accumulatedRejection) {
      reject(accumulatedRejection);
      log.warn(
          "Decision PG macro rejected before optimizer step: {}", accumulatedRejection.reason());
    }

    private void reject(EpsilonDecisionTrainingRejection updateRejection) {
      if (!updateRejection.rejected()) {
        throw new IllegalArgumentException("cannot reject a training run with PASSED");
      }
      rejection = updateRejection;
    }

    private EpsilonDecisionTrainingRejection rolloutPolicyKlRejectionAfterActorStep(
        Accumulator stepMetrics) {
      return trustRegion.checkMeanRolloutPolicyKl(stepMetrics.meanRolloutPolicyKl());
    }

    private boolean stopped() {
      return rejection.rejected();
    }

    private DecisionTrainingResult finish(
        MetricGroup metricGroup, double elapsedMillis, double fusedTotalElapsedMillis) {
      Accumulator metrics = metricGroup == MetricGroup.ACTOR ? actorMetrics : valueMetrics;
      return metrics.toMetrics(rejection, perf.snapshot(elapsedMillis, fusedTotalElapsedMillis));
    }
  }

  private Optimizer createOptimizer(Tracker learningRateTracker) {
    return Optimizer.adamW()
        .optLearningRateTracker(learningRateTracker)
        .optWeightDecays(optimizerSettings.weightDecay())
        .optClipGrad(optimizerSettings.gradClip())
        .build();
  }

  private static Model requireDecisionModel(Model model) {
    if (!(model.getBlock() instanceof EpsilonDecisionNetwork)) {
      throw new IllegalArgumentException(
          "Decision trainer requires EpsilonDecisionNetwork, got " + model.getBlock());
    }
    return model;
  }

  private static long shuffleSeed(int epoch, int size) {
    return shuffleSeed(epoch, size, 0);
  }

  private static long shuffleSeed(int epoch, int size, int salt) {
    long value = SHUFFLE_SEED ^ ((long) epoch << 32) ^ size ^ ((long) salt << 1);
    value ^= value >>> 30;
    value *= 0xbf58_476d_1ce4_e5b9L;
    value ^= value >>> 27;
    value *= 0x94d0_49bb_1331_11ebL;
    return value ^ (value >>> 31);
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("Decision trainer is already closed");
    }
  }

  private EpsilonDecisionDataParallel dataParallel() {
    ensureOpen();
    if (dataParallel == null) {
      dataParallel =
          EpsilonDecisionDataParallel.openOnline(
              model,
              tensorTransfer,
              deviceTransferSchedule,
              executionContext,
              config.bind(com.epsilon.config.settings.DeviceSettings.class));
    }
    dataParallel.ensureUsable();
    return dataParallel;
  }
}
