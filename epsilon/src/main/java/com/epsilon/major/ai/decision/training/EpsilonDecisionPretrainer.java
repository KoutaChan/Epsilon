package com.epsilon.major.ai.decision.training;

import ai.djl.Model;
import ai.djl.engine.Autocast;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.major.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.major.ai.decision.input.DecisionDeviceBatch;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.major.ai.decision.input.DecisionValueDeviceBatch;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionAutocast;
import com.epsilon.major.ai.model.EpsilonDecisionLoss;
import com.epsilon.major.ai.model.EpsilonDecisionNetwork;
import com.epsilon.major.ai.model.EpsilonDecisionOutput;
import com.epsilon.major.ai.network.NetworkDevices;
import com.epsilon.major.ai.network.NetworkFactory;
import com.epsilon.major.config.settings.DecisionPretrainSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.function.IntSupplier;
import java.util.function.LongSupplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 変換済みの Decision バッチを、一台以上のデバイスで事前学習する。
 *
 * <p>方策/価値を各デバイスへ複製するモデル全体のデータ並列だけを使う。デバイス数は2の冪に限定せず、サンプル確率の重みで
 * 正規化した小バッチ勾配を標準形式の学習ワーカーへ集約する。共有状態の特徴表現は一度だけ順伝播し、価値損失はネットワーク内部で
 * 勾配を遮断した価値特徴量の集約以降にだけ流れる。オプティマイザーと保存対象チェックポイントは更新の基準となるモデルだけが所有し、対象パラメーターを
 * 連続バッファ上で一括更新してからモデルの複製へ同期する。
 *
 * <p>入力は{@link DecisionHostBatch}の型付き連続バッファであり、各学習ワーカーは転送バッファを再利用する。利用GPU数はオプティマイザーバッチから
 * 一学習ワーカー当たり十分な行数を保てる範囲で自動決定するため、2の冪にも方策/価値比にも依存しない。
 *
 * <p>合法行動が一つだけのバッチは方策教師情報を持たないものとして扱う。方策の順伝播/逆伝播を実行せず、独立した方策 オプティマイザーの更新回数、AdamW
 * モーメンタム、重み減衰も進めない。価値は状態のみの経路で通常どおり学習する。
 *
 * <p>このクラスは一つの学習ループが逐次所有する。{@link #trainEpoch}、{@link #setLearningRateScale(float)}、{@link
 * #close()} を複数スレッドから同時に呼び出してはならない。内部実行処理は、同じパラメーター更新のデータ並列学習ワーカーだけを並行実行する。
 */
public final class EpsilonDecisionPretrainer implements AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(EpsilonDecisionPretrainer.class);
  private static final int METRIC_COUNT = 5;
  private final Model checkpointModel;
  private final com.epsilon.major.config.settings.DecisionSettings optimizerSettings;
  private final DecisionPretrainSettings settings;
  private final NetworkDevices activeDevices;
  private volatile float learningRate;
  private final EpsilonDecisionDataParallel fullModelGroup;
  private final Optimizer policyOptimizer;
  private final Optimizer valueOptimizer;
  private final DataType inputNumericDataType;
  private boolean closed;

  /**
   * プロセス設定と検出済み学習器デバイスから事前学習器を開く。
   *
   * @param checkpointModel 更新対象の標準形式の Decision モデル。先頭学習器デバイス上に存在すること
   * @return 設定に応じてデバイス構成を構築した事前学習器
   */
  public static EpsilonDecisionPretrainer open(
      Model checkpointModel, DecisionExecutionContext executionContext) {
    return open(
        checkpointModel,
        DecisionPretrainSettings.defaults(),
        NetworkFactory.getLearnerDevices(),
        executionContext);
  }

  public static EpsilonDecisionPretrainer open(
      Model checkpointModel,
      DecisionPretrainSettings settings,
      NetworkDevices devices,
      DecisionExecutionContext executionContext) {
    return new EpsilonDecisionPretrainer(checkpointModel, settings, devices, executionContext);
  }

  private EpsilonDecisionPretrainer(
      Model checkpointModel,
      DecisionPretrainSettings settings,
      NetworkDevices devices,
      DecisionExecutionContext executionContext) {
    this(
        checkpointModel,
        settings,
        devices,
        executionContext,
        com.epsilon.major.config.settings.DecisionSettings.defaults());
  }

  public static EpsilonDecisionPretrainer open(
      Model model, SettingsLoader config, DecisionExecutionContext context) {
    return new EpsilonDecisionPretrainer(
        model,
        config.bind(DecisionPretrainSettings.class),
        NetworkFactory.getLearnerDevices(
            config.bind(com.epsilon.config.settings.DeviceSettings.class)),
        context,
        config.bind(com.epsilon.major.config.settings.DecisionSettings.class));
  }

  public static EpsilonDecisionPretrainer open(
      Model model,
      DecisionPretrainSettings settings,
      NetworkDevices devices,
      DecisionExecutionContext context,
      com.epsilon.major.config.settings.DecisionSettings optimizer) {
    return new EpsilonDecisionPretrainer(model, settings, devices, context, optimizer);
  }

  private EpsilonDecisionPretrainer(
      Model checkpointModel,
      DecisionPretrainSettings settings,
      NetworkDevices devices,
      DecisionExecutionContext executionContext,
      com.epsilon.major.config.settings.DecisionSettings optimizerSettings) {
    this.optimizerSettings = optimizerSettings;
    this.checkpointModel = requireDecisionModel(checkpointModel);
    if (((EpsilonDecisionNetwork) checkpointModel.getBlock()).utilityProfile()
        != optimizerSettings.utilityProfile()) {
      throw new IllegalArgumentException(
          "Decision pretraining utility profile differs from checkpoint");
    }
    this.settings = settings;
    if (!checkpointModel.getNDManager().getDevice().equals(devices.primary())) {
      throw new IllegalArgumentException(
          "Decision checkpoint model must be on the first pretrain device: model="
              + checkpointModel.getNDManager().getDevice()
              + " devices="
              + devices);
    }
    activeDevices =
        selectActiveDevices(
            devices, settings.optimizerBatchRows(), settings.minimumRowsPerDataParallelDevice());
    updateLearningRate(settings.learningRate());
    inputNumericDataType =
        switch (settings.computePrecision()) {
          case FLOAT32 -> DataType.FLOAT32;
          case FLOAT16 -> DataType.FLOAT16;
          case BFLOAT16 -> DataType.BFLOAT16;
        };
    DataType modelParameterDataType =
        inputNumericDataType == DataType.BFLOAT16 ? DataType.BFLOAT16 : DataType.FLOAT32;
    fullModelGroup =
        EpsilonDecisionDataParallel.openPretraining(
            checkpointModel,
            activeDevices,
            settings.tensorTransfer(),
            modelParameterDataType,
            executionContext);
    Tracker tracker = ignored -> learningRate;
    policyOptimizer = createOptimizer(tracker);
    valueOptimizer = createOptimizer(tracker);
    log.info(
        "Decision pretrainer opened: phase={} execution=FULL_MODEL_DATA_PARALLEL activeDevices={} "
            + "availableDevices={} optimizerBatchRows={} maxDeviceBatchRows={} tensorTransfer={} "
            + "computePrecision={} parameterPrecision={} optimizerPrecision=FLOAT32 "
            + "valueGradientIntoSharedMemory=false",
        "warm-start",
        activeDevices,
        devices,
        settings.optimizerBatchRows(),
        settings.maximumDeviceBatchRows(),
        settings.tensorTransfer(),
        settings.computePrecision(),
        modelParameterDataType);
  }

  /**
   * 基準学習率へ掛ける実行時倍率を変更する。
   *
   * <p>オプティマイザー状態は維持され、次のパラメーター更新から新しい倍率が使われる。
   *
   * @param scale 正かつ有限な倍率
   */
  public void setLearningRateScale(float scale) {
    requireOpen();
    if (!Float.isFinite(scale) || scale <= 0.0f) {
      throw new IllegalArgumentException("learning-rate scale must be positive and finite");
    }
    updateLearningRate(settings.learningRate() * scale);
  }

  private void updateLearningRate(float value) {
    if (!Float.isFinite(value) || value <= 0.0f) {
      throw new IllegalArgumentException("learning rate must be positive and finite");
    }
    learningRate = value;
  }

  public NetworkDevices activeDevices() {
    return activeDevices;
  }

  /**
   * 与えられた変換済みバッチ列を一エポックとして順番に学習する。
   *
   * <p>各バッチは学習教師値を持ち、設定されたオプティマイザーバッチ上限以下でなければならない。複数デバイスへの分割や末尾の不均等 小バッチは内部でサンプル確率の重みに応じて補正される。
   *
   * @param batches 同一入力スキーマで符号化済みの学習バッチ列
   * @return パラメーター更新平均の損失と実測処理速度
   * @throws IOException ワーカーまたはデータセット処理に失敗した場合
   */
  public EpochMetrics trainEpoch(List<DecisionHostBatch> batches) throws IOException {
    Iterator<DecisionHostBatch> iterator = batches.iterator();
    long totalRows = batches.stream().mapToLong(DecisionHostBatch::size).sum();
    return trainEpoch(
        () -> iterator.hasNext() ? iterator.next() : null,
        () -> batches.size(),
        () -> totalRows,
        this::trainFullModelStep);
  }

  EpochMetrics trainEpoch(EpsilonDecisionPretrainBatchCursor cursor) throws IOException {
    return trainEpoch(
        cursor::next, cursor::totalBatches, cursor::totalRows, this::trainFullModelStep);
  }

  private EpochMetrics trainEpoch(
      BatchSource source, IntSupplier totalSteps, LongSupplier totalRows, BatchTrainer batchTrainer)
      throws IOException {
    requireOpen();
    long started = System.nanoTime();
    long progressLogIntervalNanos = settings.progressLogIntervalNanos();
    long nextProgressLog = started + progressLogIntervalNanos;
    MetricsAccumulator metrics = new MetricsAccumulator();
    try {
      DecisionHostBatch batch;
      while ((batch = source.next()) != null) {
        if (!batch.hasTrainingTargets()) {
          throw new IllegalArgumentException("Decision pretrain batch has no targets");
        }
        if (batch.size() > settings.optimizerBatchRows()) {
          throw new IllegalArgumentException(
              "Compiled batch exceeds optimizerBatchRows: "
                  + batch.size()
                  + "/"
                  + settings.optimizerBatchRows());
        }
        StepMetrics step = batchTrainer.train(batch);
        metrics.add(step);
        long now = System.nanoTime();
        if (now >= nextProgressLog) {
          metrics.addDeferred(fullModelGroup.drainMetrics(METRIC_COUNT));
          now = System.nanoTime();
          logEpochProgress(metrics, totalSteps.getAsInt(), totalRows.getAsLong(), started, now);
          nextProgressLog = now + progressLogIntervalNanos;
        }
      }
      metrics.addDeferred(fullModelGroup.drainMetrics(METRIC_COUNT));
      if (metrics.isEmpty()) {
        return EpochMetrics.empty();
      }
      fullModelGroup.synchronizeMasterModel();
      return metrics.finish((System.nanoTime() - started) / 1_000_000.0);
    } catch (IOException | RuntimeException | Error failure) {
      try {
        fullModelGroup.clearMetrics();
      } catch (IOException | RuntimeException | Error cleanupFailure) {
        failure.addSuppressed(cleanupFailure);
      }
      throw failure;
    }
  }

  private static void logEpochProgress(
      MetricsAccumulator metrics, int totalSteps, long totalRows, long started, long now) {
    double elapsedSeconds = (now - started) / 1_000_000_000.0;
    double rowsPerSecond = metrics.rows / elapsedSeconds;
    double etaSeconds =
        totalRows < 0 ? -1.0 : Math.max(0L, totalRows - metrics.rows) / rowsPerSecond;
    log.info(
        "Decision pretrain warm-start progress: batches={}/{} policyBatches={} rows={}/{} "
            + "policyRows={} elapsedSeconds={} "
            + "etaSeconds={} rowsPerSecond={} loss={}",
        metrics.optimizerSteps,
        totalSteps,
        metrics.policyOptimizerSteps,
        metrics.rows,
        totalRows,
        metrics.policyRows,
        elapsedSeconds,
        etaSeconds,
        rowsPerSecond,
        metrics.loss / metrics.optimizerSteps);
  }

  private StepMetrics trainFullModelStep(DecisionHostBatch batch) throws IOException {
    if (!hasPolicyChoice(batch)) {
      return trainFullModelValueOnlyStep(batch);
    }
    boolean gradientsConsumed = false;
    try {
      double globalWeight = batch.sliceRows(0, batch.size()).sampleWeightMass();
      StepMetrics metrics =
          runWaves(fullModelGroup, batch, this::runFullModelMicroBatch, globalWeight);
      fullModelGroup.applyFlattenedPretrainOptimizerSteps(policyOptimizer, valueOptimizer);
      gradientsConsumed = true;
      return metrics;
    } finally {
      if (!gradientsConsumed) {
        fullModelGroup.clearTrainingGradients();
      }
    }
  }

  private StepMetrics trainFullModelValueOnlyStep(DecisionHostBatch batch) throws IOException {
    boolean gradientsConsumed = false;
    try {
      StepMetrics metrics = computeValueGradients(fullModelGroup, batch);
      fullModelGroup.applyFlattenedOptimizerStep(
          valueOptimizer, EpsilonDecisionDataParallel.ParameterScope.PRETRAIN_VALUE);
      gradientsConsumed = true;
      return metrics;
    } finally {
      if (!gradientsConsumed) {
        fullModelGroup.clearTrainingGradients();
      }
    }
  }

  private StepMetrics computeValueGradients(
      EpsilonDecisionDataParallel group, DecisionHostBatch batch) throws IOException {
    return runWaves(
        group,
        batch,
        this::runValueMicroBatch,
        batch.sliceRows(0, batch.size()).sampleWeightMass());
  }

  private static boolean hasPolicyChoice(DecisionHostBatch batch) {
    return batch.bucket().legalActionCapacity() > 1;
  }

  private StepMetrics runWaves(
      EpsilonDecisionDataParallel group,
      DecisionHostBatch batch,
      MicroBatchRunner runner,
      double globalWeight)
      throws IOException {
    ArrayList<ArrayList<AssignedMicroBatch>> batchesByLane = new ArrayList<>(group.laneCount());
    for (int laneIndex = 0; laneIndex < group.laneCount(); laneIndex++) {
      batchesByLane.add(new ArrayList<>());
    }
    int offset = 0;
    while (offset < batch.size()) {
      int remaining = batch.size() - offset;
      int activeLanes =
          Math.min(
              group.laneCount(),
              Math.max(
                  1,
                  (remaining + settings.maximumDeviceBatchRows() - 1)
                      / settings.maximumDeviceBatchRows()));
      int waveRemaining = remaining;
      int waveOffset = offset;
      for (int laneIndex = 0; laneIndex < activeLanes; laneIndex++) {
        int lanesRemaining = activeLanes - laneIndex;
        int count =
            Math.min(
                settings.maximumDeviceBatchRows(),
                (waveRemaining + lanesRemaining - 1) / lanesRemaining);
        DecisionHostBatch.RowSlice localRows = batch.sliceRows(waveOffset, count);
        double sampleScale = localRows.sampleWeightMass() / globalWeight;
        batchesByLane.get(laneIndex).add(new AssignedMicroBatch(localRows, sampleScale));
        waveOffset += count;
        waveRemaining -= count;
      }
      offset = waveOffset;
    }

    int activeLaneCount = 0;
    for (ArrayList<AssignedMicroBatch> laneBatches : batchesByLane) {
      if (!laneBatches.isEmpty()) {
        activeLaneCount++;
      }
    }
    int[] activeLanes = new int[activeLaneCount];
    ArrayList<EpsilonDecisionDataParallel.LaneWork<StepMetrics>> work =
        new ArrayList<>(activeLaneCount);
    int activeIndex = 0;
    for (int laneIndex = 0; laneIndex < batchesByLane.size(); laneIndex++) {
      ArrayList<AssignedMicroBatch> laneBatches = batchesByLane.get(laneIndex);
      if (laneBatches.isEmpty()) {
        continue;
      }
      activeLanes[activeIndex++] = laneIndex;
      work.add(
          lane -> {
            StepMetrics laneMetrics = StepMetrics.zero();
            for (AssignedMicroBatch assigned : laneBatches) {
              laneMetrics =
                  laneMetrics.combine(runner.run(lane, assigned.rows(), assigned.sampleScale()));
            }
            return laneMetrics;
          });
    }
    StepMetrics combined = StepMetrics.zero();
    group.setActiveGradientLanes(activeLanes);
    for (StepMetrics metrics : group.invokeAssigned(activeLanes, work)) {
      combined = combined.combine(metrics);
    }
    return combined;
  }

  private StepMetrics runFullModelMicroBatch(
      EpsilonDecisionDataParallel.Lane lane,
      DecisionHostBatch.RowSlice hostRows,
      double sampleScale) {
    try (NDManager sub = lane.manager().newSubManager();
        GradientCollector collector = lane.newGradientCollector()) {
      DecisionDeviceBatch input =
          DecisionBatchTransfer.transferTrainingToDevice(
              sub,
              hostRows,
              lane.pretrainingTransfer(),
              lane.pretrainingTransferMode(),
              inputNumericDataType);
      EpsilonDecisionOutput output;
      try (Autocast ignored = openAutocast(lane)) {
        output =
            ((EpsilonDecisionNetwork) lane.model().getBlock())
                .forwardDecision(
                    new ParameterStore(sub, true), input, true, lane.runtimeParameters());
        sub.attachAll(output.toNDList());
      }
      EpsilonDecisionLoss.PolicyPretrainLoss policyLoss =
          EpsilonDecisionLoss.computePolicyPretrainingLoss(output.policyScores(), input);
      NDArray valueLoss =
          EpsilonDecisionLoss.computeValueLoss(
              output.valueLogits(),
              input.trainingTargets().valueTarget(),
              input.trainingTargets().sampleWeight(),
              lane.valueConstants());
      try (NDArray decisionObjective =
              policyLoss
                  .total()
                  .mul(settings.behaviorCloningCoef())
                  .add(valueLoss.mul(settings.valueCoef()));
          NDArray scaled = decisionObjective.mul(sampleScale)) {
        collector.backward(scaled);
        accumulateMetrics(
            lane,
            sampleScale,
            decisionObjective,
            policyLoss.total(),
            valueLoss,
            policyLoss.entropy(),
            policyLoss.chosenProbability());
        return new StepMetrics(hostRows.size(), hostRows.size());
      }
    }
  }

  private StepMetrics runValueMicroBatch(
      EpsilonDecisionDataParallel.Lane lane,
      DecisionHostBatch.RowSlice hostRows,
      double sampleScale) {
    try (NDManager sub = lane.manager().newSubManager();
        GradientCollector collector = lane.newGradientCollector()) {
      DecisionValueDeviceBatch input =
          DecisionBatchTransfer.transferValueToDevice(
              sub,
              hostRows,
              lane.pretrainingTransfer(),
              lane.pretrainingTransferMode(),
              inputNumericDataType);
      EpsilonDecisionNetwork.ValueDiagnosticOutput output;
      try (Autocast ignored = openAutocast(lane)) {
        output =
            ((EpsilonDecisionNetwork) lane.model().getBlock())
                .forwardValueWithDiagnostics(
                    new ParameterStore(sub, true),
                    input.inputs(),
                    input.playerMemoryPresentIndices(),
                    true,
                    lane.runtimeParameters());
        sub.attachAll(new NDList(output.valueLogits(), output.valueStateEmbedding()));
      }
      NDArray valueLoss =
          EpsilonDecisionLoss.computeValueLoss(
              output.valueLogits(),
              input.valueTarget(),
              input.sampleWeight(),
              lane.valueConstants());
      try (NDArray scaled = valueLoss.mul(settings.valueCoef() * sampleScale)) {
        collector.backward(scaled);
      }
      try (NDArray zero = sub.zeros(new Shape(), valueLoss.getDataType());
          NDArray objective = valueLoss.mul(settings.valueCoef())) {
        accumulateMetrics(lane, sampleScale, objective, zero, valueLoss, zero, zero);
      }
      return new StepMetrics(hostRows.size(), 0);
    }
  }

  private static void accumulateMetrics(
      EpsilonDecisionDataParallel.Lane lane, double sampleScale, NDArray... arrays) {
    try (NDArray packed = NDArrays.stack(new NDList(arrays))) {
      lane.accumulateMetrics(packed, sampleScale);
    }
  }

  private Autocast openAutocast(EpsilonDecisionDataParallel.Lane lane) {
    return EpsilonDecisionAutocast.open(lane.manager(), settings.computePrecision());
  }

  /** オプティマイザーバッチを過度に細分化しない範囲で、先頭から任意台数のデバイスを選ぶ。 */
  static NetworkDevices selectActiveDevices(NetworkDevices devices, int optimizerBatchRows) {
    return selectActiveDevices(
        devices,
        optimizerBatchRows,
        DecisionPretrainSettings.defaults().minimumRowsPerDataParallelDevice());
  }

  private static NetworkDevices selectActiveDevices(
      NetworkDevices devices, int optimizerBatchRows, int minimumRowsPerDevice) {
    if (optimizerBatchRows < 1) {
      throw new IllegalArgumentException("optimizerBatchRows must be positive");
    }
    int usefulDeviceCount =
        Math.max(1, Math.min(devices.size(), optimizerBatchRows / minimumRowsPerDevice));
    return devices.first(usefulDeviceCount);
  }

  private Optimizer createOptimizer(Tracker learningRate) {
    return Optimizer.adamW()
        .optLearningRateTracker(learningRate)
        .optWeightDecays(optimizerSettings.weightDecay())
        .optClipGrad(optimizerSettings.gradClip())
        .build();
  }

  private static Model requireDecisionModel(Model model) {
    if (!(model.getBlock() instanceof EpsilonDecisionNetwork)) {
      throw new IllegalArgumentException("Pretrainer requires EpsilonDecisionNetwork");
    }
    return model;
  }

  private void requireOpen() {
    if (closed) {
      throw new IllegalStateException("Decision pretrainer is closed");
    }
  }

  /**
   * 学習済み標準形式のチェックポイントモデルを自己対局による学習状態へ戻し、全学習ワーカー資源を解放する。
   *
   * <p>事前学習中のパラメーター有効範囲は内部実装であり、呼び出し側へ持ち越さない。同期後のチェックポイントモデルは方策/価値の全パラメーターを
   * 学習可能へ戻すため、そのまま自己対局による一つに学習へ渡せる。
   */
  @Override
  public void close() {
    if (closed) {
      return;
    }
    ((EpsilonDecisionNetwork) checkpointModel.getBlock()).prepareForOnlineTraining();
    closed = true;
    fullModelGroup.close();
  }

  @FunctionalInterface
  private interface BatchSource {
    DecisionHostBatch next() throws IOException;
  }

  @FunctionalInterface
  private interface BatchTrainer {
    StepMetrics train(DecisionHostBatch batch) throws IOException;
  }

  @FunctionalInterface
  private interface MicroBatchRunner {
    StepMetrics run(
        EpsilonDecisionDataParallel.Lane lane,
        DecisionHostBatch.RowSlice hostRows,
        double sampleScale);
  }

  /** 一つの事前学習用ワーカーで同じパラメーター更新中に連続実行するホスト部分領域。 */
  private record AssignedMicroBatch(DecisionHostBatch.RowSlice rows, double sampleScale) {}

  private record StepMetrics(int rows, int policyRows) {
    private static StepMetrics zero() {
      return new StepMetrics(0, 0);
    }

    private StepMetrics combine(StepMetrics other) {
      return new StepMetrics(
          Math.addExact(rows, other.rows), Math.addExact(policyRows, other.policyRows));
    }
  }

  private static final class MetricsAccumulator {
    private int optimizerSteps;
    private int policyOptimizerSteps;
    private int rows;
    private int policyRows;
    private double loss;
    private double behaviorCloningLoss;
    private double valueLoss;
    private double entropy;
    private double chosenProbability;

    private boolean isEmpty() {
      return optimizerSteps == 0;
    }

    private void add(StepMetrics step) {
      optimizerSteps++;
      rows += step.rows();
      if (step.policyRows() > 0) {
        policyOptimizerSteps++;
        policyRows += step.policyRows();
      }
    }

    private void addDeferred(float[] deferred) {
      if (deferred.length != METRIC_COUNT) {
        throw new IllegalArgumentException(
            "Unexpected Decision pretrain metric count: " + deferred.length);
      }
      loss += deferred[0];
      behaviorCloningLoss += deferred[1];
      valueLoss += deferred[2];
      entropy += deferred[3];
      chosenProbability += deferred[4];
    }

    private EpochMetrics finish(double elapsedMillis) {
      return new EpochMetrics(
          optimizerSteps,
          policyOptimizerSteps,
          rows,
          policyRows,
          loss / optimizerSteps,
          policyOptimizerSteps == 0 ? 0.0 : behaviorCloningLoss / policyOptimizerSteps,
          valueLoss / optimizerSteps,
          policyOptimizerSteps == 0 ? 0.0 : entropy / policyOptimizerSteps,
          policyOptimizerSteps == 0 ? 0.0 : chosenProbability / policyOptimizerSteps,
          elapsedMillis,
          rows * 1000.0 / elapsedMillis);
    }
  }

  /**
   * 一エポックの集約結果。
   *
   * @param optimizerSteps 完了したオプティマイザー更新数
   * @param policyOptimizerSteps 方策オプティマイザーを実行したバッチ数。1候補バッチを含まない
   * @param rows 学習に利用したDecision行数
   * @param policyRows 方策事前学習へ利用したDecision行数。1候補行を含まない
   * @param loss パラメーター更新平均の総目的関数
   * @param behaviorCloningLoss 選択最終行動負の対数尤度
   * @param valueLoss Decision 価値の着順RPS
   * @param entropy 標準形式の最終的な行動分布のエントロピー
   * @param chosenProbability 教師行動へ割り当てた平均確率
   * @param elapsedMillis エポックの実測経過時間
   * @param rowsPerSecond 全デバイスを合算した処理行数/秒
   */
  public record EpochMetrics(
      int optimizerSteps,
      int policyOptimizerSteps,
      int rows,
      int policyRows,
      double loss,
      double behaviorCloningLoss,
      double valueLoss,
      double entropy,
      double chosenProbability,
      double elapsedMillis,
      double rowsPerSecond) {
    private static EpochMetrics empty() {
      return new EpochMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
  }
}
