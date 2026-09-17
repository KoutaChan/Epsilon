package com.epsilon.nano.ai.decision.runtime;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Autocast;
import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtStreamScope;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.nano.ai.decision.fusion.DecisionInferenceOutputPacker;
import com.epsilon.nano.ai.decision.fusion.DecisionPackedScores;
import com.epsilon.nano.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.input.DecisionInferenceDeviceBatch;
import com.epsilon.nano.ai.decision.input.DecisionNetworkInputs;
import com.epsilon.nano.ai.decision.input.DecisionStateInferenceBatch;
import com.epsilon.nano.ai.decision.input.DecisionStateInputs;
import com.epsilon.nano.ai.decision.policy.DecisionPolicyInferenceComposer;
import com.epsilon.nano.ai.decision.policy.DecisionPolicyScores;
import com.epsilon.nano.ai.model.DecisionInferenceExecution;
import com.epsilon.nano.ai.model.EpsilonDecisionNetwork;
import com.epsilon.nano.ai.model.EpsilonDecisionOutput;
import com.epsilon.nano.ai.model.EpsilonTileRelationEncoder;
import com.epsilon.nano.config.settings.DecisionInferenceSettings;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionDevicePipeline;
import com.epsilon.runtime.DecisionDeviceStreams;
import com.epsilon.runtime.InferenceDispatcher;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 一つのDecision ネットワークとその実行資源を所有する推論コンテキスト。
 *
 * <p>ホスト側バッチがデバイス上限を超える場合は非所有の連続 {@link DecisionHostBatch.RowSlice} に分割する。各部分ビューは {@link
 * DecisionBatchTransfer.Workspace}のページ固定の/直接参照のバッファへ直接詰められるため、中間のJava配列を作らない。GPU推論では複数の非所有行
 * ビューを{@link DecisionDevicePipeline}へ渡し、Fusion/方策利用権とバッチ
 * 管理元を最終D2H完了まで枠に保持する。返す方策は合法手候補の位置順の確率または自然対数確率、価値はHL-Gaussから復号した期待効用である。
 */
public final class EpsilonDecisionInferenceServer implements EpsilonDecisionGreedyEvaluator {

  private static final Logger log = LoggerFactory.getLogger(EpsilonDecisionInferenceServer.class);

  private DecisionBatchTransfer.Workspace inputWorkspace(DecisionDevicePipeline.Slot slot) {
    return slotWorkspaces.computeIfAbsent(
        slot, ignored -> new DecisionBatchTransfer.Workspace((PtNDManager) manager));
  }

  private static final ConcurrentMap<Device, ReentrantLock> DEVICE_INFERENCE_LOCKS =
      new ConcurrentHashMap<>();
  private static final ReentrantReadWriteLock ACCELERATOR_GRAPH_RUNTIME_LOCK =
      new ReentrantReadWriteLock();

  private final int maxBatch;
  private final int multiTransitionMaxBatch;
  private final int executionSlots;
  private final DecisionOperatorProfiler operatorProfiler;
  private final boolean lowPrecisionInputNumerics;
  private final boolean acceleratorGraphEnabled;
  private final ReentrantLock deviceInferenceLock;
  private final NDManager manager;
  private final EpsilonDecisionNetwork network;
  private final ParameterStore parameterStore;
  private final PairList<String, Object> inferenceRuntimeParameters;
  private final EpsilonDecisionHlGauss.DeviceConstants valueConstants;
  private final boolean ownsManager;
  private final DecisionBatchTransfer.Workspace transferWorkspace;
  private final ConcurrentMap<DecisionDevicePipeline.Slot, DecisionBatchTransfer.Workspace>
      slotWorkspaces = new ConcurrentHashMap<>();
  private final DecisionTensorTransfer tensorTransfer;
  private final DecisionComputePrecision computePrecision;
  private final DecisionInferenceExecution inferenceExecution;
  private final DecisionInferenceOutputPacker outputPacker;
  private final OutputKind outputKind;
  private final boolean acceleratorGraphEligible;
  private final int acceleratorGraphRows;
  private boolean acceleratorGraphWarmed;
  private DecisionPolicyInferenceGraph policyInferenceGraph;
  private boolean closed;
  private boolean outputPackerReleased;
  private boolean inferenceExecutionReleased;
  private boolean transferWorkspaceReleased;
  private boolean runtimeParametersReleased;
  private boolean parameterStoreReleased;
  private boolean managerReleased;
  private boolean resourcesReleased;
  private DecisionDevicePipeline.Lease pipelineLease;
  private volatile Consumer<DecisionHostBatch.RowBatch> physicalBatchObserver;
  private final ArrayList<FixedReplay> fixedReplays = new ArrayList<>();

  /**
   * 呼び出し側が所有するネットワークへ新しい管理元を割り当て、EAGER推論サーバーを作る。
   *
   * @param network 未初期化または同形状で利用可能なDecision ネットワーク
   * @param maxBatch 一回のデバイス順伝播へ入れる最大行数
   * @return 管理元を内部所有するサーバー
   */
  public static EpsilonDecisionInferenceServer forNetwork(
      EpsilonDecisionNetwork network, int maxBatch) {
    return forNetwork(network, maxBatch, DecisionInferenceFusionSettings.eager());
  }

  /** Fusion契約の検証用。明示設定はデバイスや凍結状態に合わせて書き換えない。 */
  static EpsilonDecisionInferenceServer forNetwork(
      EpsilonDecisionNetwork network,
      int maxBatch,
      DecisionInferenceFusionSettings fusionSettings) {
    NDManager manager = NDManager.newBaseManager();
    try {
      network.initialize(manager, DataType.FLOAT32, DecisionNetworkInputs.initializationShapes());
      return new EpsilonDecisionInferenceServer(
          network, manager, maxBatch, true, false, OutputKind.POLICY_AND_VALUE, 1, fusionSettings);
    } catch (RuntimeException | Error failure) {
      manager.close();
      throw failure;
    }
  }

  /**
   * 更新可能なモデルとその管理元を借用し、パラメーターを定数化しないEAGER サーバーを作る。
   * モデルを凍結・複製しない。チェックポイントの推論にはloadForInferenceとforFrozenModelを使う。
   *
   * @param model ブロックがDecision ネットワークである読み込み済みモデル
   * @param maxBatch 一回のデバイス順伝播へ入れる最大行数
   * @return モデル管理元を借用するサーバー。解放してもモデル管理元は閉じない
   */
  public static EpsilonDecisionInferenceServer forModel(Model model, int maxBatch) {
    return forModel(model, maxBatch, 1);
  }

  /** 借用モデルから指定数の未回収順伝播を保持できる実行コンテキストを作ります。 */
  static EpsilonDecisionInferenceServer forModel(Model model, int maxBatch, int executionSlots) {
    return new EpsilonDecisionInferenceServer(
        blockFrom(model),
        model.getNDManager(),
        maxBatch,
        false,
        false,
        OutputKind.POLICY_AND_VALUE,
        executionSlots,
        DecisionInferenceFusionSettings.eager());
  }

  /** 凍結済みモデルの複製から方策/価値推論サーバーを作る。 */
  public static EpsilonDecisionInferenceServer forFrozenModel(Model model, int maxBatch) {
    return forFrozenModel(model, maxBatch, 1);
  }

  /** 複数循環バッファ枠から同時に参照できる凍結方策/価値実行コンテキストを作ります。 */
  static EpsilonDecisionInferenceServer forFrozenModel(
      Model model, int maxBatch, int executionSlots) {
    return forFrozenModel(
        model, maxBatch, executionSlots, fusionForDevice(model.getNDManager().getDevice()));
  }

  /** GPU 一致性検証用に、構成要素別Fusion設定を指定して凍結方策/価値サーバーを作ります。 */
  static EpsilonDecisionInferenceServer forFrozenModel(
      Model model,
      int maxBatch,
      int executionSlots,
      DecisionInferenceFusionSettings fusionSettings) {
    return new EpsilonDecisionInferenceServer(
        blockFrom(model),
        model.getNDManager(),
        maxBatch,
        false,
        true,
        OutputKind.POLICY_AND_VALUE,
        executionSlots,
        fusionSettings);
  }

  /** 行動だけを消費する借用モデル用EAGER サーバー。モデルの学習可能状態は変更しない。 */
  static EpsilonDecisionInferenceServer forPolicyModel(Model model, int maxBatch) {
    return new EpsilonDecisionInferenceServer(
        blockFrom(model),
        model.getNDManager(),
        maxBatch,
        false,
        false,
        OutputKind.POLICY_ONLY,
        1,
        DecisionInferenceFusionSettings.eager());
  }

  /** 行動だけを消費する対局生成用に、凍結済み方策のみサーバーを作る。 */
  static EpsilonDecisionInferenceServer forFrozenPolicyModel(Model model, int maxBatch) {
    return forFrozenPolicyModel(model, maxBatch, 1);
  }

  /** 複数循環バッファ枠から同時に参照できる凍結方策のみ実行コンテキストを作ります。 */
  static EpsilonDecisionInferenceServer forFrozenPolicyModel(
      Model model, int maxBatch, int executionSlots) {
    return forFrozenPolicyModel(
        model, maxBatch, executionSlots, fusionForDevice(model.getNDManager().getDevice()));
  }

  /** 対戦比較の明示上限だけを適用し、元の計算グラフ行数設定の検証は維持する。 */
  static EpsilonDecisionInferenceServer forFrozenDuelPolicyModel(
      Model model, int maxBatch, int executionSlots, int maximumInferenceBatch) {
    if (maximumInferenceBatch <= 0) {
      throw new IllegalArgumentException("maximumInferenceBatch must be positive");
    }
    return new EpsilonDecisionInferenceServer(
        blockFrom(model),
        model.getNDManager(),
        maxBatch,
        false,
        true,
        OutputKind.POLICY_ONLY,
        executionSlots,
        fusionForDevice(model.getNDManager().getDevice()),
        maximumInferenceBatch);
  }

  static EpsilonDecisionInferenceServer forModel(
      Model model, int maximumBatch, DecisionInferenceSettings settings) {
    return new EpsilonDecisionInferenceServer(
        blockFrom(model),
        model.getNDManager(),
        maximumBatch,
        false,
        false,
        OutputKind.POLICY_AND_VALUE,
        settings.slotsPerDevice(),
        DecisionInferenceFusionSettings.eager(),
        0,
        settings);
  }

  static EpsilonDecisionInferenceServer forFrozenDuelPolicyModel(
      Model model,
      int maximumBatch,
      int executionSlots,
      int duelMaximumBatch,
      DecisionInferenceSettings settings,
      DecisionInferenceFusionSettings fusion) {
    return new EpsilonDecisionInferenceServer(
        blockFrom(model),
        model.getNDManager(),
        maximumBatch,
        false,
        true,
        OutputKind.POLICY_ONLY,
        executionSlots,
        model.getNDManager().getDevice().isGpu() ? fusion : DecisionInferenceFusionSettings.eager(),
        duelMaximumBatch,
        settings);
  }

  /** 学習検証に必要な方策/価値を明示的な推論設定で評価します。 */
  public static EpsilonDecisionInferenceServer forFrozenModel(
      Model model,
      int maximumBatch,
      DecisionInferenceSettings settings,
      DecisionInferenceFusionSettings fusion) {
    return new EpsilonDecisionInferenceServer(
        blockFrom(model),
        model.getNDManager(),
        maximumBatch,
        false,
        true,
        OutputKind.POLICY_AND_VALUE,
        settings.slotsPerDevice(),
        model.getNDManager().getDevice().isGpu() ? fusion : DecisionInferenceFusionSettings.eager(),
        0,
        settings);
  }

  /** 系列インスタンスが解決した設定だけから凍結方策実行資源を作ります。 */
  public static EpsilonDecisionInferenceServer forPolicySession(
      Model model, DecisionInferenceSettings settings, DecisionInferenceFusionSettings fusion) {
    return forPolicySession(model, settings, fusion, settings.slotsPerDevice());
  }

  /** 対局が所有する物理枠数で、モデル固有の実行作業領域を構築します。 */
  public static EpsilonDecisionInferenceServer forPolicySession(
      Model model,
      DecisionInferenceSettings settings,
      DecisionInferenceFusionSettings fusion,
      int executionSlots) {
    return new EpsilonDecisionInferenceServer(
        blockFrom(model),
        model.getNDManager(),
        settings.maxBatch(),
        false,
        true,
        OutputKind.POLICY_ONLY,
        executionSlots,
        model.getNDManager().getDevice().isGpu() ? fusion : DecisionInferenceFusionSettings.eager(),
        0,
        settings);
  }

  private static DecisionInferenceFusionSettings fusionForDevice(Device device) {
    DecisionInferenceFusionSettings configured =
        EpsilonSettings.defaults().bind(DecisionInferenceFusionSettings.class);
    return device.isGpu() ? configured : DecisionInferenceFusionSettings.eager();
  }

  /** GPU 一致性検証用に、構成要素別Fusion設定を指定して凍結方策のみサーバーを作ります。 */
  static EpsilonDecisionInferenceServer forFrozenPolicyModel(
      Model model,
      int maxBatch,
      int executionSlots,
      DecisionInferenceFusionSettings fusionSettings) {
    return new EpsilonDecisionInferenceServer(
        blockFrom(model),
        model.getNDManager(),
        maxBatch,
        false,
        true,
        OutputKind.POLICY_ONLY,
        executionSlots,
        fusionSettings);
  }

  private EpsilonDecisionInferenceServer(
      EpsilonDecisionNetwork network,
      NDManager manager,
      int maxBatch,
      boolean ownsManager,
      boolean frozenParameters,
      OutputKind outputKind,
      int executionSlots,
      DecisionInferenceFusionSettings fusionSettings) {
    this(
        network,
        manager,
        maxBatch,
        ownsManager,
        frozenParameters,
        outputKind,
        executionSlots,
        fusionSettings,
        0);
  }

  private EpsilonDecisionInferenceServer(
      EpsilonDecisionNetwork network,
      NDManager manager,
      int maxBatch,
      boolean ownsManager,
      boolean frozenParameters,
      OutputKind outputKind,
      int executionSlots,
      DecisionInferenceFusionSettings fusionSettings,
      int duelMaximumBatch) {
    this(
        network,
        manager,
        maxBatch,
        ownsManager,
        frozenParameters,
        outputKind,
        executionSlots,
        fusionSettings,
        duelMaximumBatch,
        DecisionInferenceSettings.defaults());
  }

  private EpsilonDecisionInferenceServer(
      EpsilonDecisionNetwork network,
      NDManager manager,
      int maxBatch,
      boolean ownsManager,
      boolean frozenParameters,
      OutputKind outputKind,
      int executionSlots,
      DecisionInferenceFusionSettings fusionSettings,
      int duelMaximumBatch,
      DecisionInferenceSettings settings) {
    if (maxBatch <= 0) {
      throw new IllegalArgumentException("maxBatch must be positive");
    }
    int originalMaxBatch = maxBatch;
    if (duelMaximumBatch > 0) {
      maxBatch = Math.min(maxBatch, duelMaximumBatch);
    }
    this.network = network;
    this.manager = manager;
    this.executionSlots = executionSlots;
    computePrecision =
        manager.getDevice().isGpu()
            ? settings.computePrecision()
            : DecisionComputePrecision.FLOAT32;
    acceleratorGraphEnabled = settings.acceleratorGraph();
    fusionSettings = Objects.requireNonNull(fusionSettings, "fusionSettings");
    validateFusionEligibility(
        fusionSettings,
        frozenParameters,
        acceleratorGraphEnabled,
        manager.getDevice(),
        computePrecision,
        settings.frozenBfloat16Parameters());
    this.maxBatch = maxBatch;
    this.ownsManager = ownsManager;
    this.outputKind = outputKind;
    deviceInferenceLock =
        DEVICE_INFERENCE_LOCKS.computeIfAbsent(manager.getDevice(), ignored -> new ReentrantLock());
    operatorProfiler = new DecisionOperatorProfiler(settings);
    lowPrecisionInputNumerics = manager.getDevice().isGpu() && settings.lowPrecisionInputNumerics();
    multiTransitionMaxBatch = Math.min(maxBatch, settings.multiTransitionMaxBatch());
    tensorTransfer =
        manager.getDevice().isGpu()
            ? DecisionTensorTransfer.PINNED_BUFFER
            : DecisionTensorTransfer.DIRECT_BUFFER;
    acceleratorGraphEligible =
        acceleratorGraphEnabled
            && outputKind == OutputKind.POLICY_ONLY
            && manager.getDevice().isGpu()
            && tensorTransfer == DecisionTensorTransfer.PINNED_BUFFER;
    int configuredGraphRows =
        settings.acceleratorGraphRows() > 0 ? settings.acceleratorGraphRows() : originalMaxBatch;
    if (configuredGraphRows > originalMaxBatch) {
      throw new IllegalArgumentException("acceleratorGraphRows must not exceed maxBatch");
    }
    acceleratorGraphRows = Math.min(configuredGraphRows, maxBatch);
    ParameterStore createdParameterStore = null;
    PairList<String, Object> createdRuntimeParameters = null;
    DecisionBatchTransfer.Workspace createdTransferWorkspace = null;
    DecisionInferenceExecution createdInferenceExecution = null;
    DecisionInferenceOutputPacker createdOutputPacker = null;
    try {
      createdParameterStore = new ParameterStore(manager, false);
      createdRuntimeParameters = EpsilonTileRelationEncoder.createRuntimeParameters(manager);
      if (outputKind == OutputKind.POLICY_AND_VALUE) {
        createdRuntimeParameters.add(
            EpsilonDecisionHlGauss.DEVICE_CONSTANTS,
            new EpsilonDecisionHlGauss.DeviceConstants(manager, network.utilityProfile()));
      }
      createdTransferWorkspace = new DecisionBatchTransfer.Workspace(manager);
      createdInferenceExecution =
          network.newInferenceExecution(
              manager,
              createdParameterStore,
              createdRuntimeParameters,
              fusionSettings,
              computeDataType(computePrecision),
              inputNumericDataType(),
              maxBatch,
              multiTransitionMaxBatch,
              executionSlots,
              frozenParameters,
              outputKind == OutputKind.POLICY_AND_VALUE);
      createdOutputPacker =
          new DecisionInferenceOutputPacker(
              manager, fusionSettings.outputPacking(), executionSlots);
    } catch (RuntimeException | Error constructionFailure) {
      Throwable rollbackFailure = constructionFailure;
      rollbackFailure = closeResource(rollbackFailure, createdOutputPacker);
      rollbackFailure = closeResource(rollbackFailure, createdInferenceExecution);
      rollbackFailure = closeResource(rollbackFailure, createdTransferWorkspace);
      rollbackFailure = closeRuntimeParameters(rollbackFailure, createdRuntimeParameters);
      closeResource(
          rollbackFailure, createdParameterStore == null ? null : createdParameterStore::close);
      throw constructionFailure;
    }
    parameterStore = createdParameterStore;
    inferenceRuntimeParameters = createdRuntimeParameters;
    valueConstants =
        (EpsilonDecisionHlGauss.DeviceConstants)
            inferenceRuntimeParameters.get(EpsilonDecisionHlGauss.DEVICE_CONSTANTS);
    transferWorkspace = createdTransferWorkspace;
    inferenceExecution = createdInferenceExecution;
    outputPacker = createdOutputPacker;
    log.info(
        "Decision inference opened: device={} frozen={} fusion={} computePrecision={}"
            + " inputType={} output={} maxBatch={}",
        manager.getDevice(),
        frozenParameters,
        fusionSettings.hasFusion(),
        computePrecision,
        inputNumericDataType(),
        outputKind,
        maxBatch);
  }

  /** 構成要素別Fusion設定が、この推論コンテキストのデバイス・パラメーター契約を満たすことを検証する。 */
  private static void validateFusionEligibility(
      DecisionInferenceFusionSettings fusionSettings,
      boolean frozenParameters,
      boolean acceleratorGraphEnabled,
      Device device,
      DecisionComputePrecision computePrecision,
      boolean frozenLowPrecisionParameters) {
    if (!fusionSettings.hasFusion()) {
      return;
    }
    if (acceleratorGraphEnabled) {
      throw new IllegalArgumentException("FUSION execution requires acceleratorGraph=false");
    }
    if (fusionSettings.requiresFrozenParameters() && !frozenParameters) {
      throw new IllegalArgumentException("FUSION execution requires a frozen inference replica");
    }
    if (!device.isGpu()) {
      throw new UnsupportedOperationException("FUSION execution requires a GPU inference device");
    }
    if (fusionSettings.requiresFrozenParameters()
        && computePrecision != DecisionComputePrecision.FLOAT32
        && !frozenLowPrecisionParameters) {
      throw new IllegalArgumentException(
          "low-precision FUSION execution requires frozen low-precision parameters");
    }
  }

  /**
   * 入力順を保ったまま全行を評価する。
   *
   * <p>{@code maxBatch} 超過時の分割は内部処理であり、返却順と各行の合法手候補の位置順は変わらない。
   *
   * @param batch 容量まで構築済みの推論バッチ
   * @return 入力行順の予測
   */
  @Override
  public List<Prediction> evaluateBatch(DecisionHostBatch batch) {
    if (batch.size() != batch.capacity()) {
      throw new IllegalArgumentException("inference batch must be exactly filled");
    }
    return evaluateRows(batch.sliceRows(0, batch.size()));
  }

  /**
   * 所有連続バッファを複製せず推論行ビューを評価する。
   *
   * <p>呼び出し元は完了までビューの所有権を保持する。
   */
  synchronized List<Prediction> evaluateRows(DecisionHostBatch.RowSlice rows) {
    requireOpen();
    int bucketMaxBatch = preferredBatchSize(rows.bucket());
    if (rows.size() <= bucketMaxBatch) {
      return evaluateDeviceBatch(rows, false).predictions();
    }
    ArrayList<Prediction> predictions = new ArrayList<>(rows.size());
    for (int rowStart = 0; rowStart < rows.size(); rowStart += bucketMaxBatch) {
      int rowCount = Math.min(bucketMaxBatch, rows.size() - rowStart);
      predictions.addAll(
          evaluateDeviceBatch(
                  rows.source().sliceRows(rows.fromInclusive() + rowStart, rowCount), false)
              .predictions());
    }
    return predictions;
  }

  /**
   * 方策のみモデルを評価し、全候補分布を保持せずに各行の最大確率の行動を選ぶ方式行動候補の位置だけを返す。
   *
   * <p>順伝播と方策グラフ合成は通常の方策のみ推論と同じであり、ホスト出力の表現だけを有効要素のみのにする。
   */
  @Override
  public int[] evaluateGreedyActionSlots(DecisionHostBatch batch) {
    if (batch.size() != batch.capacity()) {
      throw new IllegalArgumentException("inference batch must be exactly filled");
    }
    return evaluateGreedyRows(batch.sliceRows(0, batch.size()));
  }

  /**
   * 所有連続バッファを複製せず推論行ビューから最大確率の行動を選ぶ方式枠だけを返す。
   *
   * <p>呼び出し元は完了までビューの所有権を保持する。
   */
  synchronized int[] evaluateGreedyRows(DecisionHostBatch.RowSlice rows) {
    requireOpen();
    if (outputKind != OutputKind.POLICY_ONLY) {
      throw new IllegalStateException("Greedy-only evaluation requires a Policy-only model");
    }
    int bucketMaxBatch = preferredBatchSize(rows.bucket());
    if (rows.size() <= bucketMaxBatch) {
      return evaluateGreedyDeviceBatch(rows);
    }
    int[] selectedSlots = new int[rows.size()];
    for (int rowStart = 0; rowStart < rows.size(); rowStart += bucketMaxBatch) {
      int rowCount = Math.min(bucketMaxBatch, rows.size() - rowStart);
      int[] partial =
          evaluateGreedyDeviceBatch(
              rows.source().sliceRows(rows.fromInclusive() + rowStart, rowCount));
      System.arraycopy(partial, 0, selectedSlots, rowStart, rowCount);
    }
    return selectedSlots;
  }

  public synchronized DiagnosticBatchEvaluation evaluateDiagnosticBatch(DecisionHostBatch batch) {
    requireOpen();
    if (outputKind == OutputKind.POLICY_ONLY) {
      throw new IllegalStateException("Policy-only inference does not produce diagnostics");
    }
    if (batch.size() != batch.capacity()) {
      throw new IllegalArgumentException("inference batch must be exactly filled");
    }
    int bucketMaxBatch = preferredBatchSize(batch.bucket());
    if (batch.size() <= bucketMaxBatch) {
      return evaluateDeviceBatch(batch.sliceRows(0, batch.size()), true);
    }
    ArrayList<Prediction> predictions = new ArrayList<>(batch.size());
    ArrayList<float[]> policyStateEmbeddings = new ArrayList<>();
    ArrayList<float[]> valueStateEmbeddings = new ArrayList<>();
    for (int rowStart = 0; rowStart < batch.size(); rowStart += bucketMaxBatch) {
      int rowCount = Math.min(bucketMaxBatch, batch.size() - rowStart);
      DiagnosticBatchEvaluation partialEvaluation =
          evaluateDeviceBatch(batch.sliceRows(rowStart, rowCount), true);
      predictions.addAll(partialEvaluation.predictions());
      policyStateEmbeddings.add(partialEvaluation.policyStateEmbeddings());
      valueStateEmbeddings.add(partialEvaluation.valueStateEmbeddings());
    }
    return new DiagnosticBatchEvaluation(
        predictions,
        DecisionInferenceOutputCodec.concatenate(policyStateEmbeddings),
        DecisionInferenceOutputCodec.concatenate(valueStateEmbeddings));
  }

  private DiagnosticBatchEvaluation evaluateDeviceBatch(
      DecisionHostBatch.RowSlice hostRows, boolean includeDiagnostics) {
    if (hostRows.bucket().legalActionCapacity() == 1 && outputKind == OutputKind.POLICY_ONLY) {
      return forcedActionBatch(hostRows.size());
    }
    Lock graphRuntimeLock = acceleratorGraphRuntimeLock(hostRows);
    if (graphRuntimeLock != null) {
      graphRuntimeLock.lock();
    }
    try {
      deviceInferenceLock.lock();
      try (InferenceMode ignored = manager.getEngine().newInferenceMode()) {
        return evaluateSingleBatch(hostRows, includeDiagnostics);
      } finally {
        deviceInferenceLock.unlock();
      }
    } finally {
      if (graphRuntimeLock != null) {
        graphRuntimeLock.unlock();
      }
    }
  }

  private int[] evaluateGreedyDeviceBatch(DecisionHostBatch.RowSlice hostRows) {
    if (hostRows.bucket().legalActionCapacity() == 1) {
      return new int[hostRows.size()];
    }
    Lock graphRuntimeLock = acceleratorGraphRuntimeLock(hostRows);
    if (graphRuntimeLock != null) {
      graphRuntimeLock.lock();
    }
    try {
      deviceInferenceLock.lock();
      try (InferenceMode ignored = manager.getEngine().newInferenceMode()) {
        return evaluateGreedyPolicyBatch(hostRows);
      } finally {
        deviceInferenceLock.unlock();
      }
    } finally {
      if (graphRuntimeLock != null) {
        graphRuntimeLock.unlock();
      }
    }
  }

  private DecisionPackedScores forwardPackedPolicyScores(
      NDManager workingManager,
      DecisionInferenceDeviceBatch deviceBatch,
      DecisionHostBatch.RowSlice hostRows) {
    return forwardPackedPolicyScores(
        workingManager, deviceBatch, hostRows.size(), hostRows.bucket());
  }

  private DecisionPackedScores forwardPackedPolicyScores(
      NDManager workingManager,
      DecisionInferenceDeviceBatch deviceBatch,
      int rowCount,
      DecisionBucket bucket) {
    DecisionInferenceExecution.Forward inferenceForward =
        inferenceExecution.beginForward(workingManager);
    boolean handedOff = false;
    boolean profileBatch = operatorProfiler.claim(bucket, rowCount);
    try {
      boolean profileStarted = false;
      try {
        if (profileBatch) {
          operatorProfiler.start();
          profileStarted = true;
        }
        DecisionPolicyScores rawScores;
        try (Autocast ignored = EpsilonDecisionAutocast.open(workingManager, computePrecision)) {
          rawScores =
              network.forwardInferencePolicy(
                  parameterStore,
                  deviceBatch,
                  workingManager,
                  inferenceForward,
                  inferenceRuntimeParameters);
        }
        AutoCloseable dependencies = inferenceForward.seal();
        DecisionPackedScores packed =
            outputPacker.pack(
                workingManager,
                new NDList(
                    rawScores.alternativeScores(),
                    rawScores.actionCandidateScores(),
                    rawScores.riichiGateScores()),
                rowCount,
                preferredBatchSize(bucket),
                dependencies);
        handedOff = true;
        return packed;
      } finally {
        if (profileStarted) {
          operatorProfiler.stop();
        }
      }
    } catch (RuntimeException | Error failure) {
      if (!handedOff) {
        inferenceForward.poison(failure);
      }
      throw failure;
    }
  }

  private PackedDecisionScores forwardPackedDecisionScores(
      NDManager workingManager,
      DecisionInferenceDeviceBatch deviceBatch,
      DecisionHostBatch.RowSlice hostRows,
      boolean includeDiagnostics) {
    return forwardPackedDecisionScores(
        workingManager, deviceBatch, hostRows.size(), hostRows.bucket(), includeDiagnostics);
  }

  private PackedDecisionScores forwardPackedDecisionScores(
      NDManager workingManager,
      DecisionInferenceDeviceBatch deviceBatch,
      int rowCount,
      DecisionBucket bucket,
      boolean includeDiagnostics) {
    DecisionInferenceExecution.Forward inferenceForward =
        inferenceExecution.beginForward(workingManager);
    boolean handedOff = false;
    try {
      EpsilonDecisionNetwork.DiagnosticOutput diagnostic;
      EpsilonDecisionOutput rawOutput;
      try (Autocast ignored = EpsilonDecisionAutocast.open(workingManager, computePrecision)) {
        diagnostic =
            includeDiagnostics
                ? network.forwardInferenceWithDiagnostics(
                    parameterStore,
                    deviceBatch,
                    workingManager,
                    inferenceForward,
                    inferenceRuntimeParameters)
                : null;
        rawOutput =
            diagnostic == null
                ? network.forwardInferenceDecision(
                    parameterStore,
                    deviceBatch,
                    workingManager,
                    inferenceForward,
                    inferenceRuntimeParameters)
                : new EpsilonDecisionOutput(diagnostic.policyScores(), diagnostic.valueLogits());
      }
      DecisionPolicyScores scores = rawOutput.policyScores();
      AutoCloseable dependencies = inferenceForward.seal();
      DecisionPackedScores packedScores =
          outputPacker.pack(
              workingManager,
              new NDList(
                  scores.alternativeScores(),
                  scores.actionCandidateScores(),
                  scores.riichiGateScores(),
                  decodeValueUtility(rawOutput.valueLogits()).reshape(rowCount, 1)),
              rowCount,
              preferredBatchSize(bucket),
              dependencies);
      PackedDecisionScores result = new PackedDecisionScores(packedScores, diagnostic);
      handedOff = true;
      return result;
    } catch (RuntimeException | Error failure) {
      if (!handedOff) {
        inferenceForward.poison(failure);
      }
      throw failure;
    }
  }

  /**
   * 単一デバイス診断経路の連結した出力をホストへ読み、読み終えた後だけFusion/方策利用権を返します。
   *
   * <p>ホスト読み出しが失敗した場合も、スコアが表す最終デバイス処理の完了を確定してから利用権を閉じます。
   */
  private static float[] readPackedScoresSynchronously(DecisionPackedScores packedScores) {
    try {
      float[] hostScores = packedScores.activeRows().toFloatArray();
      packedScores.close();
      return hostScores;
    } catch (RuntimeException | Error failure) {
      closeFailedPackedScores(packedScores, failure);
      throw failure;
    }
  }

  private static void closeFailedPackedScores(
      DecisionPackedScores packedScores, Throwable failure) {
    if (packedScores == null) {
      return;
    }
    try {
      packedScores.synchronizeForFailure();
    } catch (Throwable synchronizationFailure) {
      failure.addSuppressed(synchronizationFailure);
    }
    try {
      packedScores.close();
    } catch (Throwable closeFailure) {
      failure.addSuppressed(closeFailure);
    }
  }

  /**
   * 同期順伝播の途中失敗で保持された実行計画利用権を、現在ストリーム末尾の完了後に解放します。
   *
   * <p>診断経路はサーバーごとに直列実行されるため、イベント完了時点でこの実行処理コンテキストに属する未完了順伝播はすべて安全に回収できます。
   */
  private void recoverFailedSynchronousForward(Throwable failure) {
    if (!outputPacker.hasIncompleteWork() && !inferenceExecution.hasIncompleteWork()) {
      return;
    }
    Throwable synchronizationFailure = synchronizeCurrentStream();
    if (synchronizationFailure != null) {
      failure.addSuppressed(synchronizationFailure);
      return;
    }
    releaseFailedPlansAfterCompletion(failure);
  }

  private Throwable synchronizeCurrentStream() {
    PtEvent completion = null;
    Throwable failure = null;
    try {
      completion = ((PtEngine) manager.getEngine()).newEvent(manager.getDevice());
      completion.record();
      completion.synchronize();
    } catch (Throwable synchronizationFailure) {
      failure = synchronizationFailure;
    } finally {
      if (completion != null) {
        try {
          completion.close();
        } catch (Throwable closeFailure) {
          failure = addFailure(failure, closeFailure);
        }
      }
    }
    return failure;
  }

  private void releaseFailedPlansAfterCompletion(Throwable failure) {
    try {
      outputPacker.releaseFailedSubmissionsAfterCompletion();
    } catch (Throwable releaseFailure) {
      failure.addSuppressed(releaseFailure);
    }
    try {
      inferenceExecution.releaseFailedForwardAfterCompletion();
    } catch (Throwable releaseFailure) {
      failure.addSuppressed(releaseFailure);
    }
  }

  private Lock acceleratorGraphRuntimeLock(DecisionHostBatch.RowSlice hostRows) {
    if (!acceleratorGraphEnabled) {
      return null;
    }
    boolean capture =
        isAcceleratorGraphBatch(hostRows) && acceleratorGraphWarmed && policyInferenceGraph == null;
    return capture
        ? ACCELERATOR_GRAPH_RUNTIME_LOCK.writeLock()
        : ACCELERATOR_GRAPH_RUNTIME_LOCK.readLock();
  }

  private DiagnosticBatchEvaluation evaluateSingleBatch(
      DecisionHostBatch.RowSlice hostRows, boolean includeDiagnostics) {
    if (hostRows.bucket().legalActionCapacity() == 1) {
      if (outputKind == OutputKind.POLICY_ONLY) {
        return forcedActionBatch(hostRows.size());
      }
      return evaluateForcedActionValueBatch(hostRows, includeDiagnostics);
    }
    if (outputKind == OutputKind.POLICY_ONLY) {
      return evaluatePolicyBatch(hostRows);
    }
    try (NDManager workingManager = manager.newSubManager()) {
      DecisionInferenceDeviceBatch deviceBatch =
          DecisionBatchTransfer.transferInferenceToDevice(
              workingManager, hostRows, transferWorkspace, tensorTransfer, inputNumericDataType());

      PackedDecisionScores packedDecision;
      try {
        packedDecision =
            forwardPackedDecisionScores(workingManager, deviceBatch, hostRows, includeDiagnostics);
      } catch (RuntimeException | Error failure) {
        recoverFailedSynchronousForward(failure);
        throw failure;
      }
      EpsilonDecisionNetwork.DiagnosticOutput diagnostic = packedDecision.diagnostic();
      DecisionPackedScores packedScores = packedDecision.packedScores();
      try {
        if (diagnostic != null) {
          workingManager.attachAll(
              new NDList(diagnostic.policyStateEmbedding(), diagnostic.valueStateEmbedding()));
        }

        float[] policyStateEmbeddings =
            diagnostic == null
                ? new float[0]
                : readDiagnosticFloatArray(diagnostic.policyStateEmbedding());
        float[] valueStateEmbeddings =
            diagnostic == null
                ? new float[0]
                : readDiagnosticFloatArray(diagnostic.valueStateEmbedding());
        float[] hostPackedScores = readPackedScoresSynchronously(packedScores);
        packedScores = null;
        float[] policyProbabilities =
            DecisionPolicyInferenceComposer.composeProbabilities(hostPackedScores, hostRows, true);
        float[] valueUtilities =
            DecisionPolicyInferenceComposer.composeValueUtilities(hostPackedScores, hostRows);
        List<Prediction> predictions =
            DecisionInferenceOutputCodec.decodePredictions(
                hostRows, policyProbabilities, true, valueUtilities);
        return new DiagnosticBatchEvaluation(
            predictions, policyStateEmbeddings, valueStateEmbeddings);
      } catch (RuntimeException | Error failure) {
        closeFailedPackedScores(packedScores, failure);
        throw failure;
      }
    }
  }

  private DiagnosticBatchEvaluation evaluatePolicyBatch(DecisionHostBatch.RowSlice hostRows) {
    return evaluatePolicyBatch(hostRows, this::decodePolicyPredictions);
  }

  private int[] evaluateGreedyPolicyBatch(DecisionHostBatch.RowSlice hostRows) {
    return evaluatePolicyBatch(hostRows, this::decodeGreedyActionSlots);
  }

  private <T> T evaluatePolicyBatch(
      DecisionHostBatch.RowSlice hostRows, PolicyBatchDecoder<T> decoder) {
    if (isAcceleratorGraphBatch(hostRows)) {
      if (acceleratorGraphWarmed) {
        return evaluatePolicyGraphBatch(hostRows, decoder);
      }
      acceleratorGraphWarmed = true;
    }
    return evaluatePolicyEagerBatch(hostRows, decoder);
  }

  private boolean isAcceleratorGraphBatch(DecisionHostBatch.RowSlice hostRows) {
    return acceleratorGraphEligible
        && hostRows.size() == acceleratorGraphRows
        && hostRows.bucket().legalActionCapacity() == 16
        && hostRows.bucket().actionTransitionCapacity() == 1;
  }

  private <T> T evaluatePolicyEagerBatch(
      DecisionHostBatch.RowSlice hostRows, PolicyBatchDecoder<T> decoder) {
    try (NDManager workingManager = manager.newSubManager()) {
      DecisionInferenceDeviceBatch deviceBatch =
          DecisionBatchTransfer.transferInferenceToDevice(
              workingManager, hostRows, transferWorkspace, tensorTransfer, inputNumericDataType());

      DecisionPackedScores packedScores;
      try {
        packedScores = forwardPackedPolicyScores(workingManager, deviceBatch, hostRows);
      } catch (RuntimeException | Error failure) {
        recoverFailedSynchronousForward(failure);
        throw failure;
      }

      try {
        float[] hostPackedScores = readPackedScoresSynchronously(packedScores);
        packedScores = null;
        T result = decoder.decode(hostPackedScores, hostRows);
        return result;
      } catch (RuntimeException | Error failure) {
        closeFailedPackedScores(packedScores, failure);
        throw failure;
      }
    }
  }

  private <T> T evaluatePolicyGraphBatch(
      DecisionHostBatch.RowSlice hostRows, PolicyBatchDecoder<T> decoder) {
    if (policyInferenceGraph == null) {
      policyInferenceGraph =
          DecisionPolicyInferenceGraph.capture(
              manager,
              network,
              parameterStore,
              inferenceRuntimeParameters,
              computePrecision,
              hostRows,
              inputNumericDataType());
    } else {
      policyInferenceGraph.refresh(hostRows);
    }

    policyInferenceGraph.replay();

    float[] packedScores = policyInferenceGraph.packedScores();
    T result = decoder.decode(packedScores, hostRows);
    return result;
  }

  private DiagnosticBatchEvaluation decodePolicyPredictions(
      float[] packedScores, DecisionHostBatch.RowSlice hostRows) {
    float[] policyProbabilities =
        DecisionPolicyInferenceComposer.composeProbabilities(packedScores, hostRows, false);
    List<Prediction> predictions =
        DecisionInferenceOutputCodec.decodePredictions(
            hostRows, policyProbabilities, true, new float[0]);
    return new DiagnosticBatchEvaluation(predictions, new float[0], new float[0]);
  }

  private int[] decodeGreedyActionSlots(float[] packedScores, DecisionHostBatch.RowSlice hostRows) {
    int[] selectedSlots =
        DecisionPolicyInferenceComposer.composeGreedyActionSlots(packedScores, hostRows, false);
    return selectedSlots;
  }

  private static DiagnosticBatchEvaluation forcedActionBatch(int rowCount) {
    return new DiagnosticBatchEvaluation(
        DecisionInferenceOutputCodec.forcedActions(rowCount), new float[0], new float[0]);
  }

  private DiagnosticBatchEvaluation evaluateForcedActionValueBatch(
      DecisionHostBatch.RowSlice hostRows, boolean includeDiagnostics) {
    int rowCount = hostRows.size();
    try (NDManager workingManager = manager.newSubManager()) {
      DecisionStateInputs stateInputs =
          DecisionBatchTransfer.transferStateToDevice(
              workingManager, hostRows, transferWorkspace, tensorTransfer, inputNumericDataType());

      EpsilonDecisionNetwork.ValueDiagnosticOutput diagnostic;
      NDArray valueLogits;
      try (Autocast ignored = EpsilonDecisionAutocast.open(workingManager, computePrecision)) {
        if (includeDiagnostics) {
          diagnostic =
              network.forwardInferenceValueWithDiagnostics(
                  parameterStore, stateInputs, workingManager, inferenceRuntimeParameters);
        } else {
          diagnostic = null;
        }
        valueLogits =
            diagnostic == null
                ? network.forwardInferenceValue(
                    parameterStore, stateInputs, workingManager, inferenceRuntimeParameters)
                : diagnostic.valueLogits();
      }
      valueLogits = valueLogits.toType(DataType.FLOAT32, false);
      workingManager.attachAll(
          diagnostic == null
              ? new NDList(valueLogits)
              : new NDList(valueLogits, diagnostic.valueStateEmbedding()));

      float[] valueUtilities = decodeValueUtility(valueLogits).toFloatArray();
      List<Prediction> predictions =
          DecisionInferenceOutputCodec.forcedActionsWithValues(valueUtilities, rowCount);
      return new DiagnosticBatchEvaluation(
          predictions,
          new float[0],
          diagnostic == null
              ? new float[0]
              : readDiagnosticFloatArray(diagnostic.valueStateEmbedding()));
    }
  }

  private DataType inputNumericDataType() {
    if (!lowPrecisionInputNumerics) {
      return DataType.FLOAT32;
    }
    return computeDataType(computePrecision);
  }

  private static DataType computeDataType(DecisionComputePrecision precision) {
    return switch (precision) {
      case FLOAT32 -> DataType.FLOAT32;
      case FLOAT16 -> DataType.FLOAT16;
      case BFLOAT16 -> DataType.BFLOAT16;
    };
  }

  /** このモデルコンテキストを物理GPU共有の非同期循環バッファへ登録します。 */
  synchronized void enableAsyncRing(
      int slotsPerDevice, int readyBatchesPerDevice, DecisionDeviceStreams streams) {
    requireOpen();
    if (!manager.getDevice().isGpu()) {
      throw new IllegalArgumentException("Decision inference ring requires a GPU device");
    }
    if (tensorTransfer != DecisionTensorTransfer.PINNED_BUFFER) {
      throw new IllegalArgumentException("Decision inference ring requires pinned input transfer");
    }
    if (executionSlots != slotsPerDevice) {
      throw new IllegalStateException(
          "model execution slots do not match device pipeline slots: "
              + executionSlots
              + " != "
              + slotsPerDevice);
    }
    if (pipelineLease != null) {
      throw new IllegalStateException("Decision model is already registered to an async ring");
    }
    pipelineLease =
        DecisionDevicePipeline.acquire(
            manager.getDevice(), slotsPerDevice, readyBatchesPerDevice, streams);
  }

  /** 本番循環バッファへ投入される物理バッチを同期観測する診断コールバックを設定します。 */
  public void setPhysicalBatchObserver(Consumer<DecisionHostBatch.RowBatch> observer) {
    physicalBatchObserver = observer;
  }

  /** 物理バッチの診断コールバックを解除します。通常推論では観測処理参照一回以外の処理を行いません。 */
  public void clearPhysicalBatchObserver() {
    physicalBatchObserver = null;
  }

  public DecisionDevicePipeline.HostReadyCell acquireHostReadyCell() {
    DecisionDevicePipeline.Lease lease = pipelineLease;
    if (lease == null) {
      throw new IllegalStateException("Decision model is not registered to an async ring");
    }
    return lease.acquireHostReadyCell();
  }

  /** 入力を受け付けられるバッチ枠が返却される次の世代を通知します。取得試行前にスナップショットして待機します。 */
  public CompletableFuture<Void> hostReadyCellAvailable() {
    DecisionDevicePipeline.Lease lease = pipelineLease;
    if (lease == null) {
      throw new IllegalStateException("Decision model is not registered to an async ring");
    }
    return lease.hostReadyCellAvailable();
  }

  /** 複数の非所有行ビューを一つの循環バッファバッチとして先行投入します。 */
  public CompletableFuture<DecisionInferenceResult> submitToRing(
      DecisionHostBatch.RowBatch rows,
      DecisionInferenceResult.Kind resultKind,
      DecisionDevicePipeline.HostReadyCell hostReadyCell) {
    if (hostReadyCell == null) {
      throw new NullPointerException("hostReadyCell");
    }
    DecisionDevicePipeline.Lease lease = pipelineLease;
    if (lease == null) {
      hostReadyCell.abortIfActive();
      throw new IllegalStateException("Decision model is not registered to an async ring");
    }
    try {
      lease.claim(hostReadyCell);
    } catch (RuntimeException | Error failure) {
      hostReadyCell.abortIfActive();
      throw failure;
    }
    boolean claimed = true;
    try {
      if (rows.size() > preferredBatchSize(rows.bucket())) {
        throw new IllegalArgumentException("async ring batch exceeds typed device capacity");
      }
      if (rows.bucket().legalActionCapacity() == 1 && outputKind == OutputKind.POLICY_ONLY) {
        lease.abortClaimed(hostReadyCell);
        claimed = false;
        return CompletableFuture.completedFuture(
            resultKind == DecisionInferenceResult.Kind.PREDICTIONS
                ? new DecisionInferenceResult.Predictions(
                    DecisionInferenceOutputCodec.forcedActions(rows.size()))
                : new DecisionInferenceResult.GreedyActionSlots(new int[rows.size()]));
      }
      Consumer<DecisionHostBatch.RowBatch> observer = physicalBatchObserver;
      if (observer != null) {
        observer.accept(rows);
      }
      RingWork work = new RingWork(rows, resultKind);
      claimed = false;
      lease.submitClaimed(work, hostReadyCell);
      return work.future;
    } catch (RuntimeException | Error failure) {
      if (claimed) {
        lease.abortClaimed(hostReadyCell);
      }
      throw failure;
    }
  }

  /** 一つの所有バッチを固定入力として、D2Hとホスト側の方策合成を除いた循環バッファパイプラインへ投入します。 */
  CompletableFuture<Void> submitPipelineOnly(DecisionHostBatch hostBatch) {
    return submitPipelineOnly(
        DecisionHostBatch.RowBatch.of(hostBatch.sliceRows(0, hostBatch.size())));
  }

  /** 複数行ビューを固定入力として、D2H完了までの循環バッファパイプラインだけを実行します。 */
  CompletableFuture<Void> submitPipelineOnly(DecisionHostBatch.RowBatch rows) {
    DecisionDevicePipeline.HostReadyCell hostReadyCell = acquireReplayHostReadyCell();
    return submitPipelineOnly(rows, hostReadyCell);
  }

  /** 呼び出し側が予約した格納枠を消費し、D2H完了までの循環バッファパイプラインだけを実行します。 */
  public CompletableFuture<Void> submitPipelineOnly(
      DecisionHostBatch.RowBatch rows, DecisionDevicePipeline.HostReadyCell hostReadyCell) {
    if (hostReadyCell == null) {
      throw new NullPointerException("hostReadyCell");
    }
    boolean consumed = false;
    try {
      requireReplayRows(rows);
      if (rows.bucket().legalActionCapacity() == 1 && outputKind == OutputKind.POLICY_ONLY) {
        hostReadyCell.abortIfActive();
        consumed = true;
        return CompletableFuture.completedFuture(null);
      }
      RingWork work = new RingWork(rows);
      submitReplayWork(work, hostReadyCell);
      consumed = true;
      return work.pipelineFuture;
    } finally {
      if (!consumed) {
        hostReadyCell.abortIfActive();
      }
    }
  }

  private DecisionDevicePipeline.HostReadyCell acquireReplayHostReadyCell() {
    DecisionDevicePipeline.Lease lease = pipelineLease;
    if (lease == null) {
      throw new IllegalStateException("fixed replay requires an enabled async ring");
    }
    return acquireReplayHostReadyCell(lease);
  }

  /**
   * 固定ホスト側バッチを一度だけデバイスへ転送し、演算のみ反復用セッションを開きます。
   *
   * <p>セッションは入力デバイステンソルを所有します。利用後は必ず閉じ、未完了反復が資源を返すまで入力を保持してください。
   */
  public FixedReplay openFixedReplay(DecisionHostBatch hostBatch) {
    return openFixedReplay(DecisionHostBatch.RowBatch.of(hostBatch.sliceRows(0, hostBatch.size())));
  }

  /** 複数行ビューを一度だけデバイスへ転送し、演算のみ反復用セッションを開きます。 */
  FixedReplay openFixedReplay(DecisionHostBatch.RowBatch rows) {
    requireReplayRows(rows);
    FixedReplay replay = new FixedReplay(rows);
    if (!replay.bypassed) {
      FixedReplayPrepareWork work = new FixedReplayPrepareWork(replay);
      try {
        submitReplayWork(work);
        work.future.join();
      } catch (RuntimeException | Error failure) {
        replay.closeUnregistered();
        throw failure;
      }
    }
    try {
      synchronized (this) {
        requireOpen();
        fixedReplays.add(replay);
        replay.registered = true;
      }
      return replay;
    } catch (RuntimeException | Error failure) {
      replay.closeUnregistered();
      throw failure;
    }
  }

  private void requireReplayRows(DecisionHostBatch.RowBatch rows) {
    requireOpen();
    if (pipelineLease == null) {
      throw new IllegalStateException("fixed replay requires an enabled async ring");
    }
    if (!manager.getDevice().isGpu()) {
      throw new IllegalStateException("fixed replay requires a GPU device");
    }
    if (rows.size() > preferredBatchSize(rows.bucket())) {
      throw new IllegalArgumentException("fixed replay batch exceeds typed device capacity");
    }
  }

  private void submitReplayWork(DecisionDevicePipeline.Work work) {
    DecisionDevicePipeline.Lease lease = pipelineLease;
    if (lease == null) {
      throw new IllegalStateException("fixed replay requires an enabled async ring");
    }
    DecisionDevicePipeline.HostReadyCell hostReadyCell = acquireReplayHostReadyCell(lease);
    submitReplayWork(work, hostReadyCell);
  }

  private void submitReplayWork(
      DecisionDevicePipeline.Work work, DecisionDevicePipeline.HostReadyCell hostReadyCell) {
    DecisionDevicePipeline.Lease lease = pipelineLease;
    if (lease == null) {
      hostReadyCell.abortIfActive();
      throw new IllegalStateException("fixed replay requires an enabled async ring");
    }
    try {
      lease.claim(hostReadyCell);
    } catch (RuntimeException | Error failure) {
      hostReadyCell.abortIfActive();
      throw failure;
    }
    lease.submitClaimed(work, hostReadyCell);
  }

  private static DecisionDevicePipeline.HostReadyCell acquireReplayHostReadyCell(
      DecisionDevicePipeline.Lease lease) {
    while (true) {
      CompletableFuture<Void> available = lease.hostReadyCellAvailable();
      DecisionDevicePipeline.HostReadyCell cell = lease.acquireHostReadyCell();
      if (cell != null) {
        return cell;
      }
      available.join();
    }
  }

  /** CPU テストまたは単一デバイス診断で、複数行ビューを中間連続バッファへ連結せず順に評価します。 */
  DecisionInferenceResult evaluateSynchronously(
      DecisionHostBatch.RowBatch rows, DecisionInferenceResult.Kind resultKind) {
    if (resultKind == DecisionInferenceResult.Kind.PREDICTIONS) {
      ArrayList<Prediction> predictions = new ArrayList<>(rows.size());
      for (int index = 0; index < rows.sliceCount(); index++) {
        predictions.addAll(evaluateRows(rows.slice(index)));
      }
      return new DecisionInferenceResult.Predictions(predictions);
    }
    int[] selected = new int[rows.size()];
    int offset = 0;
    for (int index = 0; index < rows.sliceCount(); index++) {
      int[] part = evaluateGreedyRows(rows.slice(index));
      System.arraycopy(part, 0, selected, offset, part.length);
      offset += part.length;
    }
    return new DecisionInferenceResult.GreedyActionSlots(selected);
  }

  /** このモデルコンテキストを含む物理GPU上の推定処理待ちの処理を返します。 */
  public long pendingEstimatedWork() {
    DecisionDevicePipeline.Lease lease = pipelineLease;
    return lease == null ? 0L : lease.pendingEstimatedWork();
  }

  public InferenceDispatcher.Device inferenceDevice() {
    return pipelineLease.inferenceDevice();
  }

  /** このモデルコンテキストが投入したすべての循環バッファ処理の回収を待ちます。 */
  void awaitDeviceIdle() {
    requireOpen();
    DecisionDevicePipeline.Lease lease = pipelineLease;
    if (lease != null) {
      lease.awaitDeviceIdle();
    }
  }

  /**
   * 単一遷移容量区分で一回の順伝播に入れる最大行数を返す。
   *
   * @return 推論サーバーの最大物理バッチ行数
   */
  public int maxBatch() {
    return maxBatch;
  }

  public Device device() {
    return manager.getDevice();
  }

  /** 解放がこのモデルコンテキストの全資源を解放し終えた場合だけ{@code true}を返します。 */
  boolean resourcesReleased() {
    return resourcesReleased;
  }

  @Override
  public int preferredBatchSize() {
    return maxBatch;
  }

  @Override
  public int preferredBatchSize(DecisionBucket bucket) {
    return bucket.actionTransitionCapacity() == 1 ? maxBatch : multiTransitionMaxBatch;
  }

  @Override
  public int preferredStreamingBatchSize(DecisionBucket bucket) {
    return preferredBatchSize(bucket);
  }

  /**
   * 未完了循環バッファ処理を待った後、出力から入力への依存順で実行資源を閉じます。
   *
   * <p>循環バッファの失敗バッチは各枠が最終イベントを待っています。パイプライン利用権の完了待ちと回収後に限り、途中失敗が保持したFusion出力枠と方策 利用権を一括回収します。
   */
  @Override
  public synchronized void close() {
    if (resourcesReleased) {
      return;
    }
    closed = true;
    physicalBatchObserver = null;
    Throwable failure = null;
    for (FixedReplay replay : List.copyOf(fixedReplays)) {
      try {
        replay.closeFromServer();
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    fixedReplays.clear();
    if (pipelineLease != null) {
      try {
        pipelineLease.close();
        pipelineLease = null;
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    if (failure != null) {
      rethrow(failure);
    }
    if (outputPacker.hasIncompleteWork() || inferenceExecution.hasIncompleteWork()) {
      failure = synchronizeCurrentStream();
      if (failure != null) {
        rethrow(failure);
      }
    }
    try {
      outputPacker.releaseFailedSubmissionsAfterCompletion();
    } catch (Throwable releaseFailure) {
      failure = releaseFailure;
    }
    try {
      inferenceExecution.releaseFailedForwardAfterCompletion();
    } catch (Throwable releaseFailure) {
      failure = addFailure(failure, releaseFailure);
    }
    if (failure != null) {
      rethrow(failure);
    }
    if (policyInferenceGraph != null) {
      try {
        policyInferenceGraph.close();
        policyInferenceGraph = null;
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    if (!outputPackerReleased) {
      try {
        outputPacker.close();
        outputPackerReleased = true;
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    if (!inferenceExecutionReleased) {
      try {
        inferenceExecution.close();
        inferenceExecutionReleased = true;
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    if (failure != null) {
      rethrow(failure);
    }
    if (!transferWorkspaceReleased) {
      try {
        for (DecisionBatchTransfer.Workspace workspace : slotWorkspaces.values()) workspace.close();
        slotWorkspaces.clear();
        transferWorkspace.close();
        transferWorkspaceReleased = true;
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    if (failure != null) {
      rethrow(failure);
    }
    if (!runtimeParametersReleased) {
      Throwable runtimeFailure = closeRuntimeParameters(null, inferenceRuntimeParameters);
      if (runtimeFailure == null) {
        runtimeParametersReleased = true;
      } else {
        failure = addFailure(failure, runtimeFailure);
      }
    }
    if (!parameterStoreReleased) {
      try {
        parameterStore.close();
        parameterStoreReleased = true;
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    if (failure != null) {
      rethrow(failure);
    }
    if (ownsManager && !managerReleased) {
      try {
        manager.close();
        managerReleased = true;
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    resourcesReleased = failure == null;
    rethrow(failure);
  }

  /**
   * 診断用の低精度テンソルを、Java側で扱えるFLOAT32配列として読み出す。
   *
   * <p>通常推論の連結したスコア転送には使わず、診断を要求した呼び出しだけで必要な型変換を行う。
   *
   * @param array デバイス上の診断テンソル
   * @return FLOAT32へ変換した診断値
   */
  static float[] readDiagnosticFloatArray(NDArray array) {
    if (array.getDataType() == DataType.FLOAT32) {
      return array.toFloatArray();
    }
    try (NDArray converted = array.toType(DataType.FLOAT32, false)) {
      return converted.toFloatArray();
    }
  }

  private void requireOpen() {
    if (closed) {
      throw new IllegalStateException("inference server is closed");
    }
    if (inferenceExecution.failure() != null) {
      throw new IllegalStateException(
          "inference server has a poisoned execution", inferenceExecution.failure());
    }
  }

  static Throwable closeResource(Throwable failure, AutoCloseable resource) {
    if (resource == null) {
      return failure;
    }
    try {
      resource.close();
    } catch (Throwable closeFailure) {
      return addFailure(failure, closeFailure);
    }
    return failure;
  }

  private static Throwable closeRuntimeParameters(
      Throwable failure, PairList<String, Object> runtimeParameters) {
    if (runtimeParameters == null) {
      return failure;
    }
    for (int index = runtimeParameters.size() - 1; index >= 0; index--) {
      Object value = runtimeParameters.valueAt(index);
      if (value instanceof AutoCloseable resource) {
        failure = closeResource(failure, resource);
      }
    }
    return failure;
  }

  private static Throwable addFailure(Throwable failure, Throwable additional) {
    if (failure == null) {
      return additional;
    }
    if (failure != additional) {
      failure.addSuppressed(additional);
    }
    return failure;
  }

  private static void rethrow(Throwable failure) {
    if (failure == null) {
      return;
    }
    if (failure instanceof RuntimeException runtimeException) {
      throw runtimeException;
    }
    if (failure instanceof Error error) {
      throw error;
    }
    throw new IllegalStateException(failure);
  }

  private static EpsilonDecisionNetwork blockFrom(Model model) {
    if (model.getBlock() instanceof EpsilonDecisionNetwork decisionNetwork) {
      return decisionNetwork;
    }
    throw new IllegalArgumentException("Model block is not EpsilonDecisionNetwork");
  }

  /** 一つの物理循環バッファ枠上で準備処理からホスト側の方策合成までを進めるバッチ所有権です。 */
  private final class RingWork implements DecisionDevicePipeline.Work {
    private final DecisionHostBatch.RowBatch rows;
    private final DecisionInferenceResult.Kind resultKind;
    private final boolean pipelineOnly;
    private final CompletableFuture<DecisionInferenceResult> future;
    private final CompletableFuture<Void> pipelineFuture;
    private DecisionBatchTransfer.StagedInference stagedInference;
    private DecisionBatchTransfer.StagedState stagedState;
    private NDManager workingManager;
    private DecisionPackedScores packedScores;
    private int elementCount;
    private DecisionInferenceResult result;
    private Throwable terminalFailure;
    private boolean h2dStarted;
    private boolean computeStarted;
    private boolean h2dRecorded;
    private boolean computeRecorded;

    private RingWork(DecisionHostBatch.RowBatch rows, DecisionInferenceResult.Kind resultKind) {
      this.rows = rows;
      this.resultKind = resultKind;
      pipelineOnly = false;
      future = new CompletableFuture<>();
      pipelineFuture = null;
      if (resultKind == DecisionInferenceResult.Kind.GREEDY_ACTION_SLOTS
          && outputKind != OutputKind.POLICY_ONLY) {
        throw new IllegalStateException("Greedy ring evaluation requires a Policy-only model");
      }
    }

    private RingWork(DecisionHostBatch.RowBatch rows) {
      this.rows = rows;
      resultKind = null;
      pipelineOnly = true;
      future = null;
      pipelineFuture = new CompletableFuture<>();
    }

    @Override
    public long estimatedWorkUnits() {
      return estimateWorkUnits(rows);
    }

    @Override
    public DecisionDevicePipeline.BatchMetadata batchMetadata() {
      return new DecisionDevicePipeline.BatchMetadata(
          EpsilonDecisionInferenceServer.this,
          rows.bucket(),
          pipelineOnly ? "PIPELINE_ONLY" : resultKind,
          rows.size(),
          preferredBatchSize(rows.bucket()));
    }

    @Override
    public void stage(DecisionDevicePipeline.Slot slot) {
      if (rows.bucket().legalActionCapacity() == 1) {
        stagedState = inputWorkspace(slot).stageState(rows);
      } else {
        stagedInference = inputWorkspace(slot).stageInference(rows);
      }
    }

    @Override
    public void submit(DecisionDevicePipeline.Slot slot) {
      workingManager = manager.newSubManager();
      try {
        Object deviceInput;
        h2dStarted = true;
        try (PtStreamScope ignored = slot.h2dStream().openScope()) {
          deviceInput =
              stagedState == null
                  ? inputWorkspace(slot)
                      .enqueueStagedInference(
                          (PtNDManager) workingManager, stagedInference, inputNumericDataType())
                  : inputWorkspace(slot)
                      .enqueueStagedState(
                          (PtNDManager) workingManager, stagedState, inputNumericDataType());
          slot.h2dReady().record();
          h2dRecorded = true;
        }

        computeStarted = true;
        try (PtStreamScope ignored = slot.computeStream().openScope();
            InferenceMode inferenceMode = manager.getEngine().newInferenceMode()) {
          slot.h2dReady().waitOnStream();
          packedScores =
              stagedState == null
                  ? forwardRingDecision((DecisionInferenceDeviceBatch) deviceInput)
                  : forwardRingForcedValue((DecisionStateInferenceBatch) deviceInput);
          slot.computeReady().record();
          computeRecorded = true;
        }

        elementCount = packedScores.elementCount();
        slot.prepareOutput((PtNDArray) packedScores.activeRows());
      } catch (RuntimeException | Error failure) {
        recordLatestCompletion(slot, failure);
        throw failure;
      }
    }

    private DecisionPackedScores forwardRingDecision(DecisionInferenceDeviceBatch deviceBatch) {
      return outputKind == OutputKind.POLICY_ONLY
          ? forwardPackedPolicyScores(workingManager, deviceBatch, rows.size(), rows.bucket())
          : forwardPackedDecisionScores(
                  workingManager, deviceBatch, rows.size(), rows.bucket(), false)
              .packedScores();
    }

    private DecisionPackedScores forwardRingForcedValue(DecisionStateInferenceBatch stateBatch) {
      NDArray valueLogits;
      try (Autocast ignored = EpsilonDecisionAutocast.open(workingManager, computePrecision)) {
        valueLogits =
            network.forwardInferenceValue(
                parameterStore,
                stateBatch.inputs(),
                stateBatch.playerMemoryPresentIndices(),
                workingManager,
                inferenceRuntimeParameters);
      }
      return outputPacker.pack(
          workingManager,
          new NDList(decodeValueUtility(valueLogits).reshape(rows.size(), 1)),
          rows.size(),
          preferredBatchSize(rows.bucket()));
    }

    private void recordLatestCompletion(DecisionDevicePipeline.Slot slot, Throwable failure) {
      if (!h2dRecorded) {
        try (PtStreamScope ignored = slot.h2dStream().openScope()) {
          slot.h2dReady().record();
          h2dRecorded = true;
        } catch (Throwable completionFailure) {
          failure.addSuppressed(completionFailure);
        }
      }
      if (h2dRecorded && !computeRecorded) {
        try (PtStreamScope ignored = slot.computeStream().openScope()) {
          slot.h2dReady().waitOnStream();
          slot.computeReady().record();
          computeRecorded = true;
        } catch (Throwable completionFailure) {
          failure.addSuppressed(completionFailure);
        }
      }
    }

    @Override
    public boolean isComplete(DecisionDevicePipeline.Slot slot) {
      return slot.computeReady().isComplete();
    }

    @Override
    public void complete(DecisionDevicePipeline.Slot slot) {
      if (pipelineOnly) {
        return;
      }
      float[] hostScores = new float[elementCount];
      slot.outputFloats().get(hostScores);

      if (stagedState != null) {
        float[] values = hostScores;
        result =
            new DecisionInferenceResult.Predictions(
                DecisionInferenceOutputCodec.forcedActionsWithValues(values, rows.size()));
      } else if (resultKind == DecisionInferenceResult.Kind.GREEDY_ACTION_SLOTS) {
        result =
            new DecisionInferenceResult.GreedyActionSlots(
                DecisionPolicyInferenceComposer.composeGreedyActionSlots(hostScores, rows, false));
      } else {
        boolean includesValue = outputKind == OutputKind.POLICY_AND_VALUE;
        float[] policyValues =
            DecisionPolicyInferenceComposer.composeProbabilities(hostScores, rows, includesValue);
        float[] values =
            includesValue
                ? DecisionPolicyInferenceComposer.composeValueUtilities(hostScores, rows)
                : new float[0];
        result =
            new DecisionInferenceResult.Predictions(
                DecisionInferenceOutputCodec.decodePredictions(rows, policyValues, true, values));
      }
    }

    @Override
    public void release(DecisionDevicePipeline.Slot slot) {
      Throwable failure = null;
      if (packedScores != null) {
        try {
          packedScores.close();
          packedScores = null;
        } catch (Throwable closeFailure) {
          failure = closeFailure;
        }
      }
      if (workingManager != null) {
        try {
          workingManager.close();
          workingManager = null;
        } catch (Throwable closeFailure) {
          failure = addFailure(failure, closeFailure);
        }
      }
      rethrow(failure);
    }

    @Override
    public boolean abort(DecisionDevicePipeline.Slot slot, Throwable failure) {
      boolean completionProven =
          (!h2dStarted || h2dRecorded) && (!computeStarted || computeRecorded);
      if (!completionProven) {
        failure.addSuppressed(
            new IllegalStateException("Decision stream completion could not be recorded"));
        fail(failure);
        return false;
      }
      try {
        if (computeRecorded) {
          slot.computeReady().synchronize();
        } else if (h2dRecorded) {
          slot.h2dReady().synchronize();
        }
      } catch (Throwable synchronizationFailure) {
        failure.addSuppressed(synchronizationFailure);
        fail(failure);
        return false;
      }
      // 順伝播開始後の例外では、方策/Fusion側が未完了投入済み処理を保持している可能性がある。
      // バッチ単位で安全に回収できないため、枠を再利用せずパイプライン全体を直ちに例外を送出させる。
      if (computeStarted) {
        fail(failure);
        return false;
      }
      try {
        release(slot);
      } catch (Throwable closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      fail(failure);
      return true;
    }

    @Override
    public void fail(Throwable failure) {
      terminalFailure = addFailure(terminalFailure, failure);
    }

    @Override
    public void publish() {
      if (pipelineOnly) {
        if (terminalFailure == null) {
          pipelineFuture.complete(null);
        } else {
          pipelineFuture.completeExceptionally(terminalFailure);
        }
      } else if (terminalFailure == null) {
        future.complete(result);
      } else {
        future.completeExceptionally(terminalFailure);
      }
    }
  }

  /** 一度だけデバイスへ移した固定入力を共有し、順伝播と出力連結だけを反復するセッションです。 */
  public final class FixedReplay implements AutoCloseable {
    private final DecisionHostBatch.RowBatch rows;
    private final boolean bypassed;
    private NDManager deviceManager;
    private Object deviceInput;
    private int pending;
    private boolean closing;
    private boolean closed;
    private volatile boolean registered;
    private Throwable closeFailure;

    private FixedReplay(DecisionHostBatch.RowBatch rows) {
      this.rows = rows;
      bypassed = rows.bucket().legalActionCapacity() == 1 && outputKind == OutputKind.POLICY_ONLY;
    }

    /** 固定バッチの行数を返します。 */
    int rows() {
      return rows.size();
    }

    /** 固定バッチの型付き容量区分を返します。 */
    DecisionBucket bucket() {
      return rows.bucket();
    }

    /** H2DとD2Hを行わず、本番と同じ順伝播と出力連結を一回投入します。 */
    CompletableFuture<Void> submitCompute() {
      return submitCompute(acquireReplayHostReadyCell());
    }

    /** 呼び出し側が予約した格納枠を消費し、演算のみ反復を一回投入します。 */
    public CompletableFuture<Void> submitCompute(
        DecisionDevicePipeline.HostReadyCell hostReadyCell) {
      if (hostReadyCell == null) {
        throw new NullPointerException("hostReadyCell");
      }
      synchronized (this) {
        if (closing || closed) {
          hostReadyCell.abortIfActive();
          throw new IllegalStateException("fixed replay is closed");
        }
        if (bypassed) {
          hostReadyCell.abortIfActive();
          return CompletableFuture.completedFuture(null);
        }
        pending++;
      }
      FixedReplayComputeWork work = new FixedReplayComputeWork(this);
      try {
        submitReplayWork(work, hostReadyCell);
      } catch (RuntimeException | Error failure) {
        finishSubmission();
        throw failure;
      }
      work.future.whenComplete((ignored, failure) -> finishSubmission());
      return work.future;
    }

    private synchronized Object deviceInput() {
      if (deviceInput == null) {
        throw new IllegalStateException("fixed replay input is not prepared");
      }
      return deviceInput;
    }

    private synchronized void installDeviceInput(NDManager deviceManager, Object deviceInput) {
      this.deviceManager = deviceManager;
      this.deviceInput = deviceInput;
    }

    private synchronized void finishSubmission() {
      pending--;
      notifyAll();
    }

    private void closeFromServer() {
      close(false);
    }

    private void closeUnregistered() {
      close(false);
    }

    @Override
    public void close() {
      close(true);
    }

    private void close(boolean unregister) {
      NDManager managerToClose;
      boolean interrupted = false;
      synchronized (this) {
        if (closed) {
          rethrow(closeFailure);
          return;
        }
        if (closing) {
          while (!closed) {
            try {
              wait();
            } catch (InterruptedException ignored) {
              interrupted = true;
            }
          }
          if (interrupted) {
            Thread.currentThread().interrupt();
          }
          rethrow(closeFailure);
          return;
        }
        closing = true;
        while (pending != 0) {
          try {
            wait();
          } catch (InterruptedException ignored) {
            interrupted = true;
          }
        }
        managerToClose = deviceManager;
        deviceManager = null;
        deviceInput = null;
      }
      Throwable failure = null;
      if (managerToClose != null) {
        try {
          managerToClose.close();
        } catch (Throwable closeFailure) {
          failure = closeFailure;
        }
      }
      synchronized (this) {
        this.closeFailure = failure;
        closed = true;
        notifyAll();
      }
      if (unregister && registered) {
        synchronized (EpsilonDecisionInferenceServer.this) {
          fixedReplays.remove(this);
          registered = false;
        }
      }
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
      rethrow(failure);
    }
  }

  /** 固定入力を一度だけ枠のページ固定のバッファから長寿命デバイステンソルへ移す処理です。 */
  private final class FixedReplayPrepareWork implements DecisionDevicePipeline.Work {
    private final FixedReplay replay;
    private final CompletableFuture<Void> future = new CompletableFuture<>();
    private DecisionBatchTransfer.StagedInference stagedInference;
    private DecisionBatchTransfer.StagedState stagedState;
    private NDManager workingManager;
    private Object deviceInput;
    private Throwable terminalFailure;
    private boolean h2dStarted;
    private boolean h2dRecorded;

    private FixedReplayPrepareWork(FixedReplay replay) {
      this.replay = replay;
    }

    @Override
    public long estimatedWorkUnits() {
      return estimateWorkUnits(replay.rows);
    }

    @Override
    public void stage(DecisionDevicePipeline.Slot slot) {
      if (replay.rows.bucket().legalActionCapacity() == 1) {
        stagedState = inputWorkspace(slot).stageState(replay.rows);
      } else {
        stagedInference = inputWorkspace(slot).stageInference(replay.rows);
      }
    }

    @Override
    public void submit(DecisionDevicePipeline.Slot slot) {
      workingManager = manager.newSubManager();
      try {
        h2dStarted = true;
        try (PtStreamScope ignored = slot.h2dStream().openScope()) {
          deviceInput =
              stagedState == null
                  ? inputWorkspace(slot)
                      .enqueueStagedInference(
                          (PtNDManager) workingManager, stagedInference, inputNumericDataType())
                  : inputWorkspace(slot)
                      .enqueueStagedState(
                          (PtNDManager) workingManager, stagedState, inputNumericDataType());
          slot.h2dReady().record();
          h2dRecorded = true;
        }
      } catch (RuntimeException | Error failure) {
        if (!h2dRecorded) {
          try (PtStreamScope ignored = slot.h2dStream().openScope()) {
            slot.h2dReady().record();
            h2dRecorded = true;
          } catch (Throwable completionFailure) {
            failure.addSuppressed(completionFailure);
          }
        }
        throw failure;
      }
    }

    @Override
    public boolean isComplete(DecisionDevicePipeline.Slot slot) {
      return slot.h2dReady().isComplete();
    }

    @Override
    public void complete(DecisionDevicePipeline.Slot slot) {
      replay.installDeviceInput(workingManager, deviceInput);
      workingManager = null;
      deviceInput = null;
    }

    @Override
    public void release(DecisionDevicePipeline.Slot slot) {
      if (workingManager != null) {
        workingManager.close();
        workingManager = null;
        deviceInput = null;
      }
    }

    @Override
    public boolean abort(DecisionDevicePipeline.Slot slot, Throwable failure) {
      if (h2dStarted && !h2dRecorded) {
        failure.addSuppressed(
            new IllegalStateException("fixed replay H2D completion could not be recorded"));
        fail(failure);
        return false;
      }
      try {
        if (h2dRecorded) {
          slot.h2dReady().synchronize();
        }
        release(slot);
      } catch (Throwable releaseFailure) {
        failure.addSuppressed(releaseFailure);
      }
      fail(failure);
      return true;
    }

    @Override
    public void fail(Throwable failure) {
      terminalFailure = addFailure(terminalFailure, failure);
    }

    @Override
    public void publish() {
      if (terminalFailure == null) {
        future.complete(null);
      } else {
        future.completeExceptionally(terminalFailure);
      }
    }
  }

  /** 長寿命デバイス入力へ本番と同じ順伝播と出力連結を適用する処理です。 */
  private final class FixedReplayComputeWork implements DecisionDevicePipeline.Work {
    private final FixedReplay replay;
    private final CompletableFuture<Void> future = new CompletableFuture<>();
    private NDManager workingManager;
    private DecisionPackedScores packedScores;
    private Throwable terminalFailure;
    private boolean computeStarted;
    private boolean computeRecorded;

    private FixedReplayComputeWork(FixedReplay replay) {
      this.replay = replay;
    }

    @Override
    public long estimatedWorkUnits() {
      return estimateWorkUnits(replay.rows);
    }

    @Override
    public void stage(DecisionDevicePipeline.Slot slot) {}

    @Override
    public void submit(DecisionDevicePipeline.Slot slot) {
      workingManager = manager.newSubManager();
      try {
        computeStarted = true;
        try (PtStreamScope ignored = slot.computeStream().openScope();
            InferenceMode inferenceMode = manager.getEngine().newInferenceMode()) {
          Object input = replay.deviceInput();
          packedScores =
              input instanceof DecisionInferenceDeviceBatch inferenceInput
                  ? (outputKind == OutputKind.POLICY_ONLY
                      ? forwardPackedPolicyScores(
                          workingManager, inferenceInput, replay.rows.size(), replay.rows.bucket())
                      : forwardPackedDecisionScores(
                              workingManager,
                              inferenceInput,
                              replay.rows.size(),
                              replay.rows.bucket(),
                              false)
                          .packedScores())
                  : forwardFixedReplayValue((DecisionStateInferenceBatch) input, replay.rows);
          slot.computeReady().record();
          computeRecorded = true;
        }
      } catch (RuntimeException | Error failure) {
        if (!computeRecorded) {
          try (PtStreamScope ignored = slot.computeStream().openScope()) {
            slot.computeReady().record();
            computeRecorded = true;
          } catch (Throwable completionFailure) {
            failure.addSuppressed(completionFailure);
          }
        }
        throw failure;
      }
    }

    private DecisionPackedScores forwardFixedReplayValue(
        DecisionStateInferenceBatch stateBatch, DecisionHostBatch.RowBatch rows) {
      NDArray valueLogits;
      try (Autocast ignored = EpsilonDecisionAutocast.open(workingManager, computePrecision)) {
        valueLogits =
            network.forwardInferenceValue(
                parameterStore,
                stateBatch.inputs(),
                stateBatch.playerMemoryPresentIndices(),
                workingManager,
                inferenceRuntimeParameters);
      }
      return outputPacker.pack(
          workingManager,
          new NDList(decodeValueUtility(valueLogits).reshape(rows.size(), 1)),
          rows.size(),
          preferredBatchSize(rows.bucket()));
    }

    @Override
    public boolean isComplete(DecisionDevicePipeline.Slot slot) {
      return slot.computeReady().isComplete();
    }

    @Override
    public void complete(DecisionDevicePipeline.Slot slot) {}

    @Override
    public void release(DecisionDevicePipeline.Slot slot) {
      Throwable failure = null;
      if (packedScores != null) {
        try {
          packedScores.close();
          packedScores = null;
        } catch (Throwable closeFailure) {
          failure = closeFailure;
        }
      }
      if (workingManager != null) {
        try {
          workingManager.close();
          workingManager = null;
        } catch (Throwable closeFailure) {
          failure = addFailure(failure, closeFailure);
        }
      }
      rethrow(failure);
    }

    @Override
    public boolean abort(DecisionDevicePipeline.Slot slot, Throwable failure) {
      if (computeStarted && !computeRecorded) {
        failure.addSuppressed(
            new IllegalStateException("fixed replay compute completion could not be recorded"));
        fail(failure);
        return false;
      }
      try {
        if (computeRecorded) {
          slot.computeReady().synchronize();
        }
      } catch (Throwable synchronizationFailure) {
        failure.addSuppressed(synchronizationFailure);
        fail(failure);
        return false;
      }
      if (computeStarted) {
        fail(failure);
        return false;
      }
      try {
        release(slot);
      } catch (Throwable closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      fail(failure);
      return true;
    }

    @Override
    public void fail(Throwable failure) {
      terminalFailure = addFailure(terminalFailure, failure);
    }

    @Override
    public void publish() {
      if (terminalFailure == null) {
        future.complete(null);
      } else {
        future.completeExceptionally(terminalFailure);
      }
    }
  }

  private static long estimateWorkUnits(DecisionHostBatch.RowBatch rows) {
    long transitionSlots =
        (long) rows.bucket().legalActionCapacity() * rows.bucket().actionTransitionCapacity();
    return (128L + 2L * transitionSlots) * rows.size();
  }

  private NDArray decodeValueUtility(NDArray valueLogits) {
    return valueConstants.decode(valueLogits);
  }

  /**
   * 一つの意思決定行に対する推論結果。
   *
   * <p>通常の自己対局推論では合法行動確率の行ビューを保持する。監査・検証が自然対数確率を要求した場合だけ {@link
   * #policyLogProbabilities()}で変換し、無作為抽出直前の不要な対数と指数の相互変換を避ける。公開コンストラクターは従来どおり自然対数確率を受け取る。
   */
  public static final class Prediction {

    private static final float[] FORCED_POLICY = {1.0f};
    private static final Prediction FORCED_ACTION =
        new Prediction(FORCED_POLICY, 0, 1, true, Float.NaN);

    private final float[] policyValues;
    private final int policyOffset;
    private final int policyLength;
    private final boolean policyValuesAreProbabilities;
    private final float valueUtility;

    /** 方策配列を借用し、復号済み期待効用を値として保持する。 */
    public Prediction(float[] policyLogProbabilities, float valueUtility) {
      this(policyLogProbabilities, 0, policyLogProbabilities.length, false, valueUtility);
    }

    private Prediction(
        float[] policyValues,
        int policyOffset,
        int policyLength,
        boolean policyValuesAreProbabilities,
        float valueUtility) {
      this.policyValues = policyValues;
      this.policyOffset = policyOffset;
      this.policyLength = policyLength;
      this.policyValuesAreProbabilities = policyValuesAreProbabilities;
      this.valueUtility = valueUtility;
    }

    static Prediction view(
        float[] policyLogProbabilities, int policyOffset, int policyLength, float valueUtility) {
      return new Prediction(
          policyLogProbabilities, policyOffset, policyLength, false, valueUtility);
    }

    static Prediction probabilityView(
        float[] policyValues, int policyOffset, int policyLength, float valueUtility) {
      return new Prediction(policyValues, policyOffset, policyLength, true, valueUtility);
    }

    /** 自家の期待効用。方策のみの結果は価値を持たず NaN。 */
    public float valueUtility() {
      return valueUtility;
    }

    public boolean hasValue() {
      return !Float.isNaN(valueUtility);
    }

    /** 変換が必要な場合だけ、合法候補の自然対数確率を作る。 */
    public float[] policyLogProbabilities() {
      if (!policyValuesAreProbabilities
          && policyOffset == 0
          && policyLength == policyValues.length) {
        return policyValues;
      }
      float[] result = Arrays.copyOfRange(policyValues, policyOffset, policyOffset + policyLength);
      if (policyValuesAreProbabilities) {
        for (int slot = 0; slot < result.length; slot++) {
          result[slot] = (float) Math.log(result[slot]);
        }
      }
      return result;
    }

    public int policySize() {
      return policyLength;
    }

    float policyLogProbability(int slot) {
      float value = policyValues[policyOffset + slot];
      return policyValuesAreProbabilities ? (float) Math.log(value) : value;
    }

    /** 方策の格納表現を変換せず最大確率の合法手候補の位置を返す。 */
    public int greedyActionSlot() {
      int best = 0;
      for (int slot = 1; slot < policyLength; slot++) {
        if (policyValues[policyOffset + slot] > policyValues[policyOffset + best]) {
          best = slot;
        }
      }
      return best;
    }

    public float[] policyProbabilities() {
      float[] result = Arrays.copyOfRange(policyValues, policyOffset, policyOffset + policyLength);
      if (policyValuesAreProbabilities) {
        return result;
      }
      float max = Float.NEGATIVE_INFINITY;
      for (float value : result) {
        max = Math.max(max, value);
      }
      float sum = 0.0f;
      for (int slot = 0; slot < result.length; slot++) {
        result[slot] = (float) Math.exp(result[slot] - max);
        sum += result[slot];
      }
      for (int slot = 0; slot < result.length; slot++) {
        result[slot] /= sum;
      }
      return result;
    }

    public static Prediction forcedAction() {
      return FORCED_ACTION;
    }

    static Prediction forcedActionWithValue(float valueUtility) {
      return probabilityView(FORCED_POLICY, 0, 1, valueUtility);
    }
  }

  public record DiagnosticBatchEvaluation(
      List<Prediction> predictions, float[] policyStateEmbeddings, float[] valueStateEmbeddings) {}

  @FunctionalInterface
  private interface PolicyBatchDecoder<T> {
    T decode(float[] packedScores, DecisionHostBatch.RowSlice hostRows);
  }

  private enum OutputKind {
    POLICY_AND_VALUE,
    POLICY_ONLY
  }

  private record PackedDecisionScores(
      DecisionPackedScores packedScores, EpsilonDecisionNetwork.DiagnosticOutput diagnostic) {}
}
