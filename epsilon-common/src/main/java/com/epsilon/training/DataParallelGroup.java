package com.epsilon.training;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.pytorch.engine.PtCopyEvent;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtEvent;
import ai.djl.pytorch.engine.PtFlatGradientPacker;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtPinnedBuffer;
import ai.djl.pytorch.engine.PtStream;
import ai.djl.pytorch.engine.PtStreamScope;
import ai.djl.pytorch.engine.PtTensorCopyPlan;
import ai.djl.training.GradientCollector;
import ai.djl.training.optimizer.Optimizer;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.util.ArrayList;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;

/**
 * モデルと更新対象のパラメーターを受け取り、複数の学習ワーカーでデータ並列学習を実行する。
 *
 * <p>ワーカーごとのGPUストリーム、診断値の転送、勾配の集約、オプティマイザーによる更新を管理する。 損失の計算、モデルの構築、系列固有の学習入力は、各モデル系列が実装する。
 */
public class DataParallelGroup<L extends DataParallelGroup.Lane> implements AutoCloseable {

  private static final AtomicInteger THREAD_IDS = new AtomicInteger();

  protected final List<L> lanes;
  private final java.util.function.BiPredicate<ParameterScope, String> parameterSelection;
  private final DecisionExecutionContext executionContext;
  private final ParameterScope scope;
  private final Model masterModel;
  private final DataType optimizerDataType;
  private final Map<ParameterScope, ParameterLayout> layouts = new EnumMap<>(ParameterScope.class);
  private final Map<ParameterScope, FlatOptimizerState> flatOptimizerStates =
      new EnumMap<>(ParameterScope.class);
  private final ExecutorService executor;
  private int[] activeGradientLanes;
  private Throwable terminalCommitFailure;
  private boolean closed;

  protected DataParallelGroup(
      List<L> lanes,
      ParameterScope scope,
      Model masterModel,
      DataType optimizerDataType,
      DecisionExecutionContext executionContext,
      java.util.function.BiPredicate<ParameterScope, String> parameterSelection) {
    this.parameterSelection = parameterSelection;
    this.lanes = lanes;
    this.executionContext = executionContext;
    this.scope = scope;
    this.masterModel = masterModel;
    this.optimizerDataType = optimizerDataType;
    activeGradientLanes = new int[lanes.size()];
    for (int laneIndex = 0; laneIndex < lanes.size(); laneIndex++) {
      activeGradientLanes[laneIndex] = laneIndex;
    }
    layouts.put(
        scope,
        ParameterLayout.from(
            lanes, name -> parameterSelection.test(scope, name), scope.label(), optimizerDataType));
    executor =
        lanes.size() == 1
            ? null
            : Executors.newFixedThreadPool(
                lanes.size() - 1,
                task -> {
                  Thread thread =
                      new Thread(task, "decision-data-parallel-" + THREAD_IDS.incrementAndGet());
                  thread.setDaemon(true);
                  return thread;
                });
  }

  public DataType optimizerDataType() {
    return optimizerDataType;
  }

  public int laneCount() {
    return lanes.size();
  }

  /**
   * 対象のバッチを処理した学習ワーカーだけを勾配集約の対象にする。
   *
   * <p>渡された配列を保持するため、呼び出し元は次の設定までその要素を変更しない。
   */
  public void setActiveGradientLanes(int[] activeLanes) {
    activeGradientLanes = activeLanes;
  }

  protected long dedicatedComputeStreamCount() {
    return lanes.stream().filter(Lane::hasDedicatedComputeStream).count();
  }

  /** 初回順伝播がモデル複製より先行しないよう、各専用ストリームへ渡す初期パラメーターイベントを記録する。 */
  protected void recordParameterReadiness() {
    lanes.forEach(Lane::recordParametersReady);
  }

  public String devices() {
    return lanes.stream().map(lane -> lane.manager().getDevice().toString()).toList().toString();
  }

  public void ensureUsable() {
    requireOpen();
  }

  public <T> List<T> executeWork(List<? extends LaneWork<L, T>> work) throws IOException {
    return executeWork(null, work);
  }

  /**
   * 処理ごとに指定された学習ワーカーで並列実行し、入力と同じ順序で結果を返す。
   *
   * <p>先頭のワーカーから順に割り当てる必要はなく、最後に残ったバッチを複製モデル側だけで処理することもできる。同じ呼び出し内でワーカーを重複指定してはならない。
   */
  public <T> List<T> executeAssigned(int[] laneIndices, List<? extends LaneWork<L, T>> work)
      throws IOException {
    return executeWork(laneIndices, work);
  }

  private <T> List<T> executeWork(int[] assignedLanes, List<? extends LaneWork<L, T>> work)
      throws IOException {
    requireOpen();
    if (work.isEmpty() || work.size() > lanes.size()) {
      throw new IllegalArgumentException(
          "Decision data parallel work size must be in [1," + lanes.size() + "]: " + work.size());
    }
    if (assignedLanes != null && assignedLanes.length != work.size()) {
      throw new IllegalArgumentException("Decision lane assignment size differs from work size");
    }
    boolean[] used = assignedLanes == null ? null : new boolean[lanes.size()];
    for (int index = 0; assignedLanes != null && index < assignedLanes.length; index++) {
      int lane = assignedLanes[index];
      if (lane < 0 || lane >= lanes.size() || used[lane]) {
        throw new IllegalArgumentException(
            "Invalid or duplicate Decision lane assignment: " + lane);
      }
      used[lane] = true;
    }
    int singleLane = assignedLanes == null ? 0 : assignedLanes[0];
    if (work.size() == 1 && singleLane == 0) {
      try {
        return List.of(executeLane(lanes.getFirst(), work.getFirst()));
      } catch (Exception failure) {
        throw propagate(failure);
      }
    }

    ArrayList<Future<T>> futures = new ArrayList<>(work.size());
    int[] futureResultIndices = new int[work.size()];
    int futureCount = 0;
    int canonicalWorkIndex = -1;
    for (int workIndex = 0; workIndex < work.size(); workIndex++) {
      int laneIndex = assignedLanes == null ? workIndex : assignedLanes[workIndex];
      if (laneIndex == 0) {
        canonicalWorkIndex = workIndex;
        continue;
      }
      L lane = lanes.get(laneIndex);
      LaneWork<L, T> laneWork = work.get(workIndex);
      futures.add(executor.submit(() -> executeLane(lane, laneWork)));
      futureResultIndices[futureCount++] = workIndex;
    }
    ArrayList<T> results = new ArrayList<>(work.size());
    for (int i = 0; i < work.size(); i++) {
      results.add(null);
    }
    Throwable failure;
    failure = null;
    if (canonicalWorkIndex >= 0) {
      try {
        results.set(
            canonicalWorkIndex, executeLane(lanes.getFirst(), work.get(canonicalWorkIndex)));
      } catch (Throwable laneFailure) {
        failure = laneFailure;
      }
    }
    boolean interrupted = false;
    for (int index = 0; index < futures.size(); index++) {
      Future<T> future = futures.get(index);
      boolean complete = false;
      while (!complete) {
        try {
          T result = future.get();
          if (failure == null) {
            results.set(futureResultIndices[index], result);
          }
          complete = true;
        } catch (InterruptedException waitFailure) {
          interrupted = true;
          if (failure == null) {
            failure = waitFailure;
          } else {
            failure.addSuppressed(waitFailure);
          }
        } catch (ExecutionException workerFailure) {
          Throwable cause = workerFailure.getCause();
          if (failure == null) {
            failure = cause;
          } else {
            failure.addSuppressed(cause);
          }
          complete = true;
        }
      }
    }
    if (interrupted) {
      Thread.currentThread().interrupt();
    }
    if (failure == null) {
      return results;
    }
    if (failure instanceof InterruptedException) {
      throw new IOException("Interrupted while waiting for data-parallel work", failure);
    }
    throw propagate(failure);
  }

  public void clearTrainingGradients() {
    requireNotClosed();
    try (ComputeScopes ignored = openComputeScopes()) {
      lanes.forEach(
          lane -> clearGradients(lane.model(), name -> parameterSelection.test(scope, name)));
    }
  }

  /** 学習ワーカー上で遅延集約した小さな診断値をホストへ回収し、次の集約区間へ向けて0へ戻す。 */
  public float[] drainMetrics(int metricCount) throws IOException {
    if (metricCount < 1) {
      throw new IllegalArgumentException("metricCount must be positive: " + metricCount);
    }
    ArrayList<LaneWork<L, float[]>> work = new ArrayList<>(lanes.size());
    for (int laneIndex = 0; laneIndex < lanes.size(); laneIndex++) {
      work.add(lane -> lane.drainMetrics(metricCount));
    }
    float[] combined = new float[metricCount];
    for (float[] laneMetrics : executeWork(work)) {
      for (int metricIndex = 0; metricIndex < metricCount; metricIndex++) {
        combined[metricIndex] += laneMetrics[metricIndex];
      }
    }
    return combined;
  }

  /** 失敗したエポックの未回収診断値を破棄する。 */
  public void clearMetrics() throws IOException {
    ArrayList<LaneWork<L, Boolean>> work = new ArrayList<>(lanes.size());
    for (int laneIndex = 0; laneIndex < lanes.size(); laneIndex++) {
      work.add(
          lane -> {
            lane.clearMetrics();
            return true;
          });
    }
    executeWork(work);
  }

  /** GPU間コピーが参照するコピー元・コピー先のストリームを、各デバイスの共有演算ストリームにそろえる。 */
  private ComputeScopes openComputeScopes() {
    return new ComputeScopes();
  }

  private final class ComputeScopes implements AutoCloseable {
    private final ArrayList<PtStreamScope> scopes = new ArrayList<>();

    private ComputeScopes() {
      HashSet<Device> opened = new HashSet<>();
      try {
        for (L lane : lanes) {
          Device device = lane.manager().getDevice();
          if (device.isGpu() && opened.add(device)) {
            scopes.add(executionContext.streams(device).compute().openScope());
          }
        }
        for (L lane : lanes) {
          if (lane.stepCompletion != null) {
            lane.stepCompletion.waitOnStream();
            lane.parametersReady.waitOnStream();
          }
        }
      } catch (RuntimeException | Error failure) {
        for (int index = scopes.size() - 1; index >= 0; index--) scopes.get(index).close();
        throw failure;
      }
    }

    @Override
    public void close() {
      try {
        for (L lane : lanes) {
          if (lane.parametersReady != null) lane.parametersReady.record();
        }
      } finally {
        for (int index = scopes.size() - 1; index >= 0; index--) scopes.get(index).close();
      }
      for (L lane : lanes) {
        if (lane.parametersReady != null) lane.parametersReady.waitOnStream();
      }
    }
  }

  /** 更新の基準となる重みを、呼び出し元が所有するFP32モデルへ反映する。初期化済みのパラメーター範囲だけを対象にする。 */
  public void synchronizeMasterModel() {
    try (ComputeScopes ignored = openComputeScopes()) {
      requireOpen();
      if (masterModel == null) {
        return;
      }
      L canonical = lanes.getFirst();
      for (FlatOptimizerState optimizerState : flatOptimizerStates.values()) {
        if (!optimizerState.parametersInitialized) {
          continue;
        }
        TransferBuffers buffers =
            canonical.transferBuffers(optimizerState.scope(), optimizerState.layout());
        unpackMasterParameters(masterModel, buffers, optimizerState.layout());
      }
    }
  }

  /**
   * 更新対象のパラメーターを一つの連続テンソルとしてAdamWで更新し、基準モデルと全複製モデルへ反映する。
   *
   * <p>全対象パラメーターが毎回勾配を持つ事前学習では、同じハイパーパラメーターと更新回数を使う個別更新と同じ数式になる。勾配の連結・デバイス間の加算・更新・配布をまとめ、オプティマイザーのカーネル起動を1回にする。元の勾配は連結時に消去する。
   *
   * @param optimizer 連続バッファにまとめたパラメーターと同じ更新回数を所有するオプティマイザー
   * @throws IOException 学習ワーカー上の勾配連結に失敗した場合
   */
  public void applyFlattenedOptimizerStep(Optimizer optimizer) throws IOException {
    applyFlattenedOptimizerStep(optimizer, scope, 1.0);
  }

  /**
   * 指定した範囲のパラメーターだけを一括更新する。
   *
   * <p>{@link
   * ParameterScope#PRETRAIN_ALL}では方策と価値が独立したAdamWの状態と更新回数を持つ。合法手が一つしかないバッチで価値だけを更新しても、方策側のモーメント・重み減衰・更新回数は変化しない。
   */
  public void applyFlattenedOptimizerStep(Optimizer optimizer, ParameterScope optimizerScope)
      throws IOException {
    applyFlattenedOptimizerStep(optimizer, optimizerScope, 1.0);
  }

  /**
   * 指定した範囲の勾配を集約し、正の係数を掛けてパラメーターを一括更新する。
   *
   * <p>各ワーカーの勾配を連続バッファへ連結して元の勾配を消去する。基準デバイスで勾配の集約・検証・倍率の適用・AdamW更新を行い、更新結果を複製モデルへ配布する。
   *
   * @param optimizer 範囲専用のオプティマイザー状態/更新回数
   * @param optimizerScope 更新対象パラメーター範囲
   * @param gradientScale 集約後の勾配へ掛ける有限かつ正の係数
   */
  public void applyFlattenedOptimizerStep(
      Optimizer optimizer, ParameterScope optimizerScope, double gradientScale) throws IOException {
    PreparedFlatOptimizerStep prepared =
        prepareFlattenedOptimizerStep(optimizerScope, gradientScale);
    validateAndScalePreparedGradients(prepared);
    try {
      commitPreparedOptimizerSteps(new PreparedOptimizerUpdate(optimizer, prepared));
    } catch (IOException | RuntimeException | Error failure) {
      poisonAfterCommitFailure(failure);
      throw failure;
    }
  }

  /** 方策と価値の勾配を両方検証してから更新し、検証エラーによって片方だけが更新されることを防ぐ。 */
  public void applyFlattenedOnlineOptimizerSteps(
      Optimizer actorOptimizer,
      double actorGradientScale,
      Optimizer valueOptimizer,
      double valueGradientScale)
      throws IOException {
    if (scope != ParameterScope.ONLINE) {
      throw new IllegalStateException("fused online optimizer steps require an ONLINE session");
    }
    PreparedFlatOptimizerStep actor =
        prepareFlattenedOptimizerStep(ParameterScope.ONLINE_ACTOR, actorGradientScale);
    PreparedFlatOptimizerStep value =
        prepareFlattenedOptimizerStep(ParameterScope.ONLINE_VALUE, valueGradientScale);
    validateAndScalePreparedGradients(actor, value);
    try {
      commitPreparedOptimizerSteps(
          new PreparedOptimizerUpdate(actorOptimizer, actor),
          new PreparedOptimizerUpdate(valueOptimizer, value));
    } catch (IOException | RuntimeException | Error failure) {
      poisonAfterCommitFailure(failure);
      throw failure;
    }
  }

  /** 方策と価値の勾配を1回のホスト同期で検証し、それぞれ独立したAdamWの状態で更新する。 */
  public void applyFlattenedPretrainOptimizerSteps(
      Optimizer policyOptimizer, Optimizer valueOptimizer) throws IOException {
    if (scope != ParameterScope.PRETRAIN_ALL) {
      throw new IllegalStateException(
          "fused pretrain optimizer steps require a PRETRAIN_ALL session");
    }
    PreparedFlatOptimizerStep policy =
        prepareFlattenedOptimizerStep(ParameterScope.PRETRAIN_POLICY, 1.0);
    PreparedFlatOptimizerStep value =
        prepareFlattenedOptimizerStep(ParameterScope.PRETRAIN_VALUE, 1.0);
    validateAndScalePreparedGradients(policy, value);
    try {
      commitPreparedOptimizerSteps(
          new PreparedOptimizerUpdate(policyOptimizer, policy),
          new PreparedOptimizerUpdate(valueOptimizer, value));
    } catch (IOException | RuntimeException | Error failure) {
      poisonAfterCommitFailure(failure);
      throw failure;
    }
  }

  private PreparedFlatOptimizerStep prepareFlattenedOptimizerStep(
      ParameterScope optimizerScope, double gradientScale) throws IOException {
    requireOpen();
    if (!Double.isFinite(gradientScale)
        || gradientScale <= 0.0
        || gradientScale > Float.MAX_VALUE) {
      throw new IllegalArgumentException(
          optimizerScope + " gradientScale must be finite and positive: " + gradientScale);
    }
    if (!scope.contains(optimizerScope)) {
      throw new IllegalArgumentException(
          "Decision data-parallel scope " + scope + " does not contain " + optimizerScope);
    }
    FlatOptimizerState optimizerState = flatOptimizerState(optimizerScope);
    initializeFlatOptimizerBuffers(optimizerState);
    ParameterLayout layout = optimizerState.layout();

    ArrayList<LaneWork<L, Boolean>> packWork = new ArrayList<>(activeGradientLanes.length);
    boolean canonicalContributes = false;
    for (int laneIndex : activeGradientLanes) {
      canonicalContributes |= laneIndex == 0;
      packWork.add(
          lane -> {
            TransferBuffers buffers = lane.transferBuffers(optimizerScope, layout);
            buffers.packAndClearGradients(optimizerScope.allowsMissingLaneGradients());
            return true;
          });
    }
    executeAssigned(activeGradientLanes, packWork);

    try (ComputeScopes ignored = openComputeScopes()) {
      L canonical = lanes.getFirst();
      TransferBuffers canonicalBuffers = canonical.transferBuffers(optimizerScope, layout);
      if (canonicalContributes) {
        canonicalBuffers.prepareOptimizerGradient();
      } else {
        canonicalBuffers.gradient().fillI(0.0f);
      }
      for (int laneIndex : activeGradientLanes) {
        if (laneIndex == 0) continue;
        lanes
            .get(laneIndex)
            .transferBuffers(optimizerScope, layout)
            .packedGradient()
            .copyTo(optimizerState.gradientReductionBuffer());
        canonicalBuffers.gradient().addi(optimizerState.gradientReductionBuffer());
      }

      return new PreparedFlatOptimizerStep(optimizerScope, canonicalBuffers, layout, gradientScale);
    }
  }

  private void commitPreparedOptimizerSteps(PreparedOptimizerUpdate... updates) throws IOException {
    try (ComputeScopes ignored = openComputeScopes()) {
      for (PreparedOptimizerUpdate update : updates) {
        PreparedFlatOptimizerStep prepared = update.prepared();
        if (prepared.layout().parameterDataType() == prepared.layout().optimizerDataType()) {
          update
              .optimizer()
              .update(
                  prepared.scope().label(),
                  prepared.canonicalBuffers().parameter(),
                  prepared.canonicalBuffers().gradient());
        } else {
          update
              .optimizer()
              .updateWithMasterWeight(
                  prepared.scope().label(),
                  prepared.canonicalBuffers().modelParameter(),
                  prepared.canonicalBuffers().parameter(),
                  prepared.canonicalBuffers().gradient());
        }
        for (int laneIndex = 1; laneIndex < lanes.size(); laneIndex++) {
          TransferBuffers destination =
              lanes.get(laneIndex).transferBuffers(prepared.scope(), prepared.layout());
          prepared.canonicalBuffers().modelParameter().copyTo(destination.modelParameter());
        }
      }
    }
    ArrayList<LaneWork<L, Boolean>> unpackWork = new ArrayList<>(lanes.size());
    for (int laneIndex = 0; laneIndex < lanes.size(); laneIndex++) {
      unpackWork.add(
          lane -> {
            for (PreparedOptimizerUpdate update : updates) {
              PreparedFlatOptimizerStep prepared = update.prepared();
              lane.transferBuffers(prepared.scope(), prepared.layout())
                  .synchronizeModelParameters();
            }
            return true;
          });
    }
    executeWork(unpackWork);
  }

  private void poisonAfterCommitFailure(Throwable failure) {
    if (terminalCommitFailure == null) {
      terminalCommitFailure = failure;
    }
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;
    if (executor != null) {
      executor.shutdown();
    }
    flatOptimizerStates.values().forEach(FlatOptimizerState::close);
    RuntimeException failure = null;
    for (int i = lanes.size() - 1; i >= 0; i--) {
      try {
        lanes.get(i).close();
      } catch (RuntimeException closeFailure) {
        if (failure == null) {
          failure = closeFailure;
        } else {
          failure.addSuppressed(closeFailure);
        }
      }
    }
    if (failure != null) {
      throw failure;
    }
  }

  private void validateAndScalePreparedGradients(PreparedFlatOptimizerStep... preparedSteps) {
    try (ComputeScopes ignored = openComputeScopes()) {
      if (preparedSteps.length == 0) {
        throw new IllegalArgumentException("At least one prepared gradient is required");
      }
      ArrayList<NDArray> temporaries = new ArrayList<>(preparedSteps.length * 2);
      ArrayList<NDArray> maximumArrays = new ArrayList<>(preparedSteps.length);
      float[] maximums;
      try {
        for (PreparedFlatOptimizerStep prepared : preparedSteps) {
          NDArray absolute = prepared.canonicalBuffers().gradient().abs();
          NDArray maximum = absolute.max();
          temporaries.add(absolute);
          temporaries.add(maximum);
          maximumArrays.add(maximum);
        }
        if (maximumArrays.size() == 1) {
          maximums = new float[] {maximumArrays.getFirst().getFloat()};
        } else {
          try (NDArray packedMaximums = NDArrays.stack(new NDList(maximumArrays))) {
            maximums = packedMaximums.toFloatArray();
          }
        }
      } finally {
        for (int index = temporaries.size() - 1; index >= 0; index--) {
          temporaries.get(index).close();
        }
      }

      for (int index = 0; index < preparedSteps.length; index++) {
        PreparedFlatOptimizerStep prepared = preparedSteps[index];
        float maximum = maximums[index];
        if (!Float.isFinite(maximum)) {
          throw new IllegalStateException(
              prepared.scope().label() + " gradient contains a non-finite value");
        }
        if (!(maximum > 0.0f)) {
          throw new IllegalStateException(prepared.scope().label() + " gradient is zero");
        }
        double scaledMaximum = maximum * prepared.gradientScale();
        if (!Double.isFinite(scaledMaximum) || scaledMaximum > Float.MAX_VALUE) {
          throw new IllegalStateException(
              prepared.scope().label()
                  + " gradient normalization would overflow: maxAbs="
                  + maximum
                  + " scale="
                  + prepared.gradientScale());
        }
      }
      for (PreparedFlatOptimizerStep prepared : preparedSteps) {
        if (prepared.gradientScale() != 1.0) {
          prepared.canonicalBuffers().gradient().muli((float) prepared.gradientScale());
        }
      }
    }
  }

  private FlatOptimizerState flatOptimizerState(ParameterScope updateScope) {
    return flatOptimizerStates.computeIfAbsent(
        updateScope,
        key ->
            new FlatOptimizerState(
                key,
                layouts.computeIfAbsent(
                    key,
                    ignored ->
                        ParameterLayout.from(
                            lanes,
                            name -> parameterSelection.test(key, name),
                            key.label(),
                            optimizerDataType))));
  }

  private void initializeFlatOptimizerBuffers(FlatOptimizerState optimizerState) {
    try (ComputeScopes ignored = openComputeScopes()) {
      ParameterLayout layout = optimizerState.layout();
      L canonical = lanes.getFirst();
      lanes.forEach(lane -> lane.transferBuffers(optimizerState.scope(), layout));
      if (lanes.size() > 1 && optimizerState.gradientReductionBuffer() == null) {
        optimizerState.gradientReductionBuffer =
            canonical.manager().zeros(new Shape(layout.elements()), layout.parameterDataType());
      }
      if (!optimizerState.parametersInitialized) {
        packParameters(
            masterModel == null ? canonical.model() : masterModel,
            canonical.transferBuffers(optimizerState.scope(), layout).parameter(),
            layout);
        optimizerState.parametersInitialized = true;
      }
    }
  }

  private static void packParameters(Model model, NDArray packed, ParameterLayout layout) {
    ParameterList parameters = model.getBlock().getParameters();
    for (ParameterSlice slice : layout.slices()) {
      try (NDArray flattened = parameters.get(slice.name()).getArray().reshape(slice.flatShape())) {
        packed.set(slice.index(), flattened);
      }
    }
  }

  private static void unpackMasterParameters(
      Model model, TransferBuffers buffers, ParameterLayout layout) {
    ParameterList parameters = model.getBlock().getParameters();
    for (int ordinal = 0; ordinal < layout.slices().size(); ordinal++) {
      ParameterSlice slice = layout.slices().get(ordinal);
      buffers.parameterSlice(ordinal).copyTo(parameters.get(slice.name()).getArray());
    }
  }

  public static void copyParameters(Model source, Model target) {
    ParameterList sourceParameters = source.getBlock().getParameters();
    ParameterList targetParameters = target.getBlock().getParameters();
    for (String name : sourceParameters.keys()) {
      Parameter targetParameter = targetParameters.get(name);
      if (targetParameter == null) {
        throw new IllegalArgumentException("Training replica parameter is missing: " + name);
      }
      sourceParameters.get(name).getArray().copyTo(targetParameter.getArray());
    }
  }

  protected static void castFloatingParameters(Model model, DataType dataType) {
    ParameterList parameters = model.getBlock().getParameters();
    for (String name : parameters.keys()) {
      Parameter parameter = parameters.get(name);
      if (parameter.getArray().getDataType().isFloating()) {
        parameter.castArray(dataType);
      }
    }
  }

  protected static void requireFloatingParameterDataType(
      Model model, DataType expected, String label) {
    ParameterList parameters = model.getBlock().getParameters();
    for (String name : parameters.keys()) {
      DataType actual = parameters.get(name).getArray().getDataType();
      if (actual.isFloating() && actual != expected) {
        throw new IllegalArgumentException(
            "Decision "
                + label
                + " floating parameter dtype differs: "
                + name
                + " expected="
                + expected
                + " actual="
                + actual);
      }
    }
  }

  public static void copyParameters(Model source, Model target, Predicate<String> included) {
    ParameterList sourceParameters = source.getBlock().getParameters();
    ParameterList targetParameters = target.getBlock().getParameters();
    for (String name : sourceParameters.keys()) {
      if (!included.test(name)) {
        continue;
      }
      Parameter targetParameter = targetParameters.get(name);
      if (targetParameter == null) {
        throw new IllegalArgumentException("Training target parameter is missing: " + name);
      }
      sourceParameters.get(name).getArray().copyTo(targetParameter.getArray());
    }
  }

  private static void clearGradients(Model model, Predicate<String> included) {
    ParameterList parameters = model.getBlock().getParameters();
    for (String name : parameters.keys()) {
      if (!included.test(name)) {
        continue;
      }
      NDArray array = parameters.get(name).getArray();
      if (array.hasGradient()) {
        try (NDArray gradient = array.getGradient()) {
          gradient.fillI(0.0f);
        }
      }
    }
  }

  protected static IOException propagate(Throwable failure) {
    if (failure instanceof IOException ioFailure) {
      return ioFailure;
    }
    if (failure instanceof RuntimeException runtimeFailure) {
      throw runtimeFailure;
    }
    if (failure instanceof Error error) {
      throw error;
    }
    return new IOException("Decision data-parallel worker failed", failure);
  }

  protected static void closeLanes(List<? extends Lane> lanes, Throwable failure) {
    for (int i = lanes.size() - 1; i >= 0; i--) {
      try {
        lanes.get(i).close();
      } catch (RuntimeException closeFailure) {
        failure.addSuppressed(closeFailure);
      }
    }
  }

  private void requireOpen() {
    requireNotClosed();
    if (terminalCommitFailure != null) {
      throw new IllegalStateException(
          "Decision data-parallel session cannot be reused after a failed optimizer" + " commit",
          terminalCommitFailure);
    }
  }

  private void requireNotClosed() {
    if (closed) {
      throw new IllegalStateException("Decision data-parallel session is already closed");
    }
  }

  private <T> T executeLane(L lane, LaneWork<L, T> work) throws Exception {
    try (LaneExecution ignored = lane.openExecution()) {
      return work.run(lane);
    }
  }

  @FunctionalInterface
  public interface LaneWork<L extends Lane, T> {
    T run(L lane) throws Exception;
  }

  /** 学習段階と、更新対象が方策・価値・その両方のどれかを表す。 */
  public enum ParameterScope {
    ONLINE("online"),
    ONLINE_ACTOR("online-actor"),
    ONLINE_VALUE("online-value"),
    PRETRAIN_ALL("offline-pretrain"),
    PRETRAIN_POLICY("offline-policy"),
    PRETRAIN_VALUE("offline-value");

    private final String label;

    ParameterScope(String label) {
      this.label = label;
    }

    public String label() {
      return label;
    }

    public boolean contains(ParameterScope requested) {
      return this == requested
          || (this == ONLINE && (requested == ONLINE_ACTOR || requested == ONLINE_VALUE))
          || (this == PRETRAIN_ALL
              && (requested == PRETRAIN_POLICY || requested == PRETRAIN_VALUE));
    }

    public boolean allowsMissingLaneGradients() {
      return this == ONLINE_ACTOR || this == ONLINE_VALUE;
    }

    public boolean usesOnlineTrainingInput() {
      return this == ONLINE || this == ONLINE_ACTOR || this == ONLINE_VALUE;
    }
  }

  private static final class FlatOptimizerState implements AutoCloseable {
    private final ParameterLayout layout;
    private final ParameterScope scope;
    private NDArray gradientReductionBuffer;
    private boolean parametersInitialized;

    private FlatOptimizerState(ParameterScope scope, ParameterLayout layout) {
      this.layout = layout;
      this.scope = scope;
    }

    private ParameterLayout layout() {
      return layout;
    }

    private ParameterScope scope() {
      return scope;
    }

    private NDArray gradientReductionBuffer() {
      return gradientReductionBuffer;
    }

    @Override
    public void close() {
      if (gradientReductionBuffer != null) {
        gradientReductionBuffer.close();
      }
    }
  }

  private record ParameterLayout(
      List<ParameterSlice> slices,
      long elements,
      DataType parameterDataType,
      DataType optimizerDataType,
      String label) {
    private static ParameterLayout from(
        List<? extends Lane> lanes,
        Predicate<String> included,
        String label,
        DataType optimizerDataType) {
      ParameterList canonical = lanes.getFirst().model().getBlock().getParameters();
      ArrayList<ParameterSlice> slices = new ArrayList<>();
      DataType parameterDataType = null;
      long offset = 0L;
      for (String name : canonical.keys()) {
        if (!included.test(name)) {
          continue;
        }
        Parameter parameter = canonical.get(name);
        if (!parameter.requiresGradient()) {
          throw new IllegalStateException(label + " parameter is frozen: " + name);
        }
        NDArray array = parameter.getArray();
        if (parameterDataType == null) {
          parameterDataType = array.getDataType();
        } else if (array.getDataType() != parameterDataType) {
          throw new IllegalArgumentException(
              label
                  + " data-parallel transfer requires one dtype: "
                  + name
                  + " expected="
                  + parameterDataType
                  + " actual="
                  + array.getDataType());
        }
        long elements = array.getShape().size();
        long end = Math.addExact(offset, elements);
        slices.add(new ParameterSlice(name, array.getShape(), offset, end));
        offset = end;
      }
      if (slices.isEmpty()) {
        throw new IllegalArgumentException("Decision model has no online " + label + " parameters");
      }
      for (int laneIndex = 1; laneIndex < lanes.size(); laneIndex++) {
        ParameterList replica = lanes.get(laneIndex).model().getBlock().getParameters();
        for (ParameterSlice slice : slices) {
          Parameter parameter = replica.get(slice.name());
          if (parameter == null) {
            throw new IllegalArgumentException(
                label + " replica parameter is missing: " + slice.name());
          }
          NDArray array = parameter.getArray();
          if (!array.getShape().equals(slice.shape()) || array.getDataType() != parameterDataType) {
            throw new IllegalArgumentException(
                label
                    + " replica parameter layout differs: "
                    + slice.name()
                    + " expected="
                    + slice.shape()
                    + "/"
                    + parameterDataType
                    + " actual="
                    + array.getShape()
                    + "/"
                    + array.getDataType());
          }
        }
      }
      if (!optimizerDataType.isFloating()) {
        throw new IllegalArgumentException(
            label + " optimizer data type must be floating: " + optimizerDataType);
      }
      return new ParameterLayout(slices, offset, parameterDataType, optimizerDataType, label);
    }
  }

  private record ParameterSlice(String name, Shape shape, long offset, long end) {

    private Shape flatShape() {
      return new Shape(end - offset);
    }

    private NDIndex index() {
      return new NDIndex().addSliceDim(offset, end);
    }
  }

  private record PreparedFlatOptimizerStep(
      ParameterScope scope,
      TransferBuffers canonicalBuffers,
      ParameterLayout layout,
      double gradientScale) {}

  private record PreparedOptimizerUpdate(Optimizer optimizer, PreparedFlatOptimizerStep prepared) {}

  public static class Lane implements AutoCloseable {
    private final Model model;
    private final boolean ownsModel;
    private final PtStream computeStream;
    private final boolean ownsComputeStream;
    final PtEvent stepCompletion;
    final PtEvent parametersReady;
    private final Map<ParameterScope, TransferBuffers> transferBuffers =
        new EnumMap<>(ParameterScope.class);
    private NDArray metricAccumulator;
    private PtPinnedBuffer metricReadbackBuffer;
    private PtCopyEvent metricReadbackEvent;
    private int metricReadbackCount;

    protected Lane(
        Model model,
        boolean ownsModel,
        boolean additionalComputeStream,
        DecisionExecutionContext executionContext) {
      this.model = model;
      this.ownsModel = ownsModel;
      ownsComputeStream = additionalComputeStream;
      if (!model.getNDManager().getDevice().isGpu()) {
        computeStream = null;
        stepCompletion = null;
        parametersReady = null;
        return;
      }
      if (!(model.getNDManager() instanceof PtNDManager pytorchManager)) {
        throw new IllegalArgumentException(
            "A dedicated Decision learner stream requires a PyTorch manager");
      }
      PtStream createdStream =
          additionalComputeStream
              ? ((PtEngine) pytorchManager.getEngine()).newStream(pytorchManager.getDevice())
              : executionContext.streams(pytorchManager.getDevice()).compute();
      PtEvent createdCompletion = null;
      PtEvent createdParametersReady = null;
      try {
        createdCompletion = createdStream.newEvent();
        createdParametersReady = createdStream.newEvent();
      } catch (RuntimeException | Error failure) {
        if (createdParametersReady != null) {
          createdParametersReady.close();
        }
        if (createdCompletion != null) {
          createdCompletion.close();
        }
        if (additionalComputeStream) createdStream.close();
        throw failure;
      }
      computeStream = createdStream;
      stepCompletion = createdCompletion;
      parametersReady = createdParametersReady;
    }

    public boolean hasDedicatedComputeStream() {
      return computeStream != null;
    }

    /** 学習ワーカーの演算用ストリームに切り替え、パラメーターの準備完了を待ってから演算する。終了時には元のストリームでも演算完了を待てるようにする。 */
    final LaneExecution openExecution() {
      if (computeStream == null) return LaneExecution.DIRECT;
      PtStreamScope streamScope = computeStream.openScope();
      try {
        parametersReady.waitOnStream();
        return new LaneExecution(streamScope, stepCompletion);
      } catch (RuntimeException | Error failure) {
        streamScope.close();
        throw failure;
      }
    }

    /** パラメーター初期化または配布の末尾を記録し、次の学習ワーカーストリームへデバイス側の依存関係を渡す。 */
    public void recordParametersReady() {
      if (parametersReady != null) {
        parametersReady.record();
        stepCompletion.record();
      }
    }

    public Model model() {
      return model;
    }

    public NDManager manager() {
      return model.getNDManager();
    }

    /** 勾配を計算するための{@link GradientCollector}を、この学習ワーカーの実行エンジンから生成する。 */
    public GradientCollector newGradientCollector() {
      return manager().getEngine().newGradientCollector();
    }

    /** FLOAT32の指標を、再利用可能なページ固定バッファへ非同期に転送する。完了前に次の転送を開始してはならない。 */
    public void enqueueMetricReadback(NDArray metrics) {
      if (metricReadbackEvent != null) {
        throw new IllegalStateException("A Decision metric readback is already pending");
      }
      if (metrics.getShape().dimension() != 1) {
        throw new IllegalArgumentException(
            "metrics must be one-dimensional: " + metrics.getShape());
      }
      if (metrics.getDataType() != DataType.FLOAT32) {
        throw new IllegalArgumentException("metrics must be FLOAT32: " + metrics.getDataType());
      }
      if (!(metrics instanceof PtNDArray pytorchMetrics)) {
        throw new IllegalArgumentException(
            "asynchronous Decision metric readback requires a PyTorch NDArray");
      }
      int count = Math.toIntExact(metrics.size());
      PtPinnedBuffer buffer = metricReadbackBuffer(count);
      metricReadbackEvent = pytorchMetrics.copyToPinnedBufferAsync(buffer);
      metricReadbackCount = count;
    }

    /** 非同期指標転送を待ち、再利用バッファから値を読む。 */
    public float[] awaitMetricReadback() {
      PtCopyEvent event = metricReadbackEvent;
      if (event == null) {
        throw new IllegalStateException("No Decision metric readback is pending");
      }
      try {
        event.synchronize();
        float[] values = new float[metricReadbackCount];
        metricReadbackBuffer.getByteBuffer().asFloatBuffer().get(values);
        return values;
      } finally {
        try {
          event.close();
        } finally {
          metricReadbackEvent = null;
          metricReadbackCount = 0;
        }
      }
    }

    /** 例外経路の未完了指標転送を同期して破棄する。 */
    public void closeMetricReadback() {
      PtCopyEvent event = metricReadbackEvent;
      if (event == null) {
        return;
      }
      try {
        event.close();
      } finally {
        metricReadbackEvent = null;
        metricReadbackCount = 0;
      }
    }

    private PtPinnedBuffer metricReadbackBuffer(int count) {
      if (metricReadbackBuffer == null || metricReadbackBuffer.size() < count) {
        if (metricReadbackBuffer != null) {
          metricReadbackBuffer.close();
        }
        if (!(manager() instanceof PtNDManager pytorchManager)) {
          throw new IllegalStateException(
              "asynchronous Decision metric readback requires a PyTorch manager");
        }
        metricReadbackBuffer = pytorchManager.allocatePinned(count, DataType.FLOAT32);
      }
      return metricReadbackBuffer;
    }

    /** 小分けにしたバッチ診断値をデバイス上で加算し、通常ステップごとのホスト同期を避ける。 */
    public void accumulateMetrics(NDArray metrics, double scale) {
      if (!Double.isFinite(scale) || scale < 0.0 || scale > Float.MAX_VALUE) {
        throw new IllegalArgumentException(
            "metric scale must be finite and non-negative: " + scale);
      }
      if (metrics.getShape().dimension() != 1) {
        throw new IllegalArgumentException(
            "metrics must be one-dimensional: " + metrics.getShape());
      }
      if (metrics.getDataType() != DataType.FLOAT32) {
        throw new IllegalArgumentException("metrics must be FLOAT32: " + metrics.getDataType());
      }
      if (metricAccumulator == null) {
        metricAccumulator = manager().zeros(metrics.getShape(), DataType.FLOAT32);
      } else if (!metricAccumulator.getShape().equals(metrics.getShape())) {
        throw new IllegalArgumentException(
            "metric shape changed: expected="
                + metricAccumulator.getShape()
                + " actual="
                + metrics.getShape());
      }
      try (NDArray detached = metrics.stopGradient();
          NDArray scaledMetrics = detached.mul((float) scale)) {
        metricAccumulator.addi(scaledMetrics);
      }
    }

    final float[] drainMetrics(int metricCount) {
      if (metricAccumulator == null) {
        return new float[metricCount];
      }
      if (metricAccumulator.getShape().size() != metricCount) {
        throw new IllegalArgumentException(
            "metric count changed: expected="
                + metricAccumulator.getShape().size()
                + " actual="
                + metricCount);
      }
      float[] result = metricAccumulator.toFloatArray();
      metricAccumulator.fillI(0.0f);
      return result;
    }

    final void clearMetrics() {
      if (metricAccumulator != null) {
        metricAccumulator.fillI(0.0f);
      }
    }

    final TransferBuffers transferBuffers(ParameterScope scope, ParameterLayout layout) {
      return transferBuffers.computeIfAbsent(
          scope,
          ignored -> {
            return new TransferBuffers(model, layout);
          });
    }

    /** 系列側の先行入力処理を停止する。GPU定数の解放より先に呼ぶ。 */
    protected void closeInputs() throws Exception {}

    /** 演算完了を確認した後に系列側のデバイス定数を解放する。 */
    protected void closeRuntime() throws Exception {}

    @Override
    public void close() {
      Throwable failure = null;
      try {
        closeMetricReadback();
      } catch (RuntimeException | Error closeFailure) {
        failure = closeFailure;
      }
      failure = closeAfter(this::closeInputs, failure);
      boolean computeDrained = computeStream == null;
      if (stepCompletion != null) {
        try {
          stepCompletion.synchronize();
          parametersReady.synchronize();
          computeDrained = true;
        } catch (RuntimeException | Error closeFailure) {
          failure = addCloseFailure(failure, closeFailure);
        }
      }
      for (TransferBuffers buffers : transferBuffers.values()) {
        failure = closeAfter(buffers, failure);
      }
      failure = closeAfter(this::closeRuntime, failure);
      if (metricAccumulator != null) {
        failure = closeAfter(metricAccumulator, failure);
      }
      if (metricReadbackBuffer != null) {
        failure = closeAfter(metricReadbackBuffer, failure);
      }
      if (stepCompletion != null) {
        failure = closeAfter(stepCompletion, failure);
      }
      if (parametersReady != null) {
        failure = closeAfter(parametersReady, failure);
      }
      if (ownsModel) {
        failure = closeAfter(model, failure);
      }
      if (ownsComputeStream && computeDrained) {
        failure = closeAfter(computeStream, failure);
      }
      if (failure instanceof RuntimeException runtimeFailure) {
        throw runtimeFailure;
      }
      if (failure instanceof Error error) {
        throw error;
      }
      if (failure != null) {
        throw new IllegalStateException("Failed to close a Decision learner lane", failure);
      }
    }

    private static Throwable closeAfter(AutoCloseable resource, Throwable failure) {
      try {
        resource.close();
        return failure;
      } catch (Throwable closeFailure) {
        return addCloseFailure(failure, closeFailure);
      }
    }

    private static Throwable addCloseFailure(Throwable primary, Throwable next) {
      if (primary == null) {
        return next;
      }
      primary.addSuppressed(next);
      return primary;
    }
  }

  /** 完了イベントを元のストリームへ渡す。GPU間の依存だけでホストを待機させない。 */
  static final class LaneExecution implements AutoCloseable {
    private static final LaneExecution DIRECT = new LaneExecution(null, null);

    private final PtStreamScope streamScope;
    private final PtEvent completion;

    private LaneExecution(PtStreamScope streamScope, PtEvent completion) {
      this.streamScope = streamScope;
      this.completion = completion;
    }

    @Override
    public void close() {
      if (completion == null) {
        return;
      }
      try {
        completion.record();
      } finally {
        streamScope.close();
      }
      completion.waitOnStream();
    }
  }

  private static final class TransferBuffers implements AutoCloseable {
    private final NDArray gradient;
    private final NDArray packingGradient;
    private final PtFlatGradientPacker gradientPacker;
    private final PtTensorCopyPlan parameterCopyPlan;
    private final NDList sourceParameters;
    private final List<NDArray> cpuGradientSlices;
    private final NDArray parameter;
    private final NDArray modelParameter;
    private final List<NDArray> parameterSlices;
    private final List<NDArray> modelParameterSlices;

    private TransferBuffers(Model model, ParameterLayout layout) {
      NDManager manager = model.getNDManager();
      Shape shape = new Shape(layout.elements());
      gradient = manager.zeros(shape, layout.optimizerDataType());
      packingGradient =
          layout.parameterDataType() == layout.optimizerDataType()
              ? gradient
              : manager.zeros(shape, layout.parameterDataType());
      parameter = manager.zeros(shape, layout.optimizerDataType());
      modelParameter =
          layout.parameterDataType() == layout.optimizerDataType()
              ? parameter
              : manager.zeros(shape, layout.parameterDataType());
      parameterSlices = new ArrayList<>(layout.slices().size());
      modelParameterSlices =
          modelParameter == parameter ? parameterSlices : new ArrayList<>(layout.slices().size());
      sourceParameters = new NDList();
      cpuGradientSlices = new ArrayList<>();
      PtTensorCopyPlan createdCopyPlan = null;
      try {
        NDList packedParameters = sourceParameters;
        ParameterList parameters = model.getBlock().getParameters();
        for (ParameterSlice slice : layout.slices()) {
          packedParameters.add(parameters.get(slice.name()).getArray());
          if (!manager.getDevice().isGpu()) {
            try (NDArray flatGradient = packingGradient.get(slice.index())) {
              cpuGradientSlices.add(flatGradient.reshape(slice.shape()));
            }
          }
          try (NDArray flatParameter = parameter.get(slice.index())) {
            parameterSlices.add(flatParameter.reshape(slice.shape()));
          }
          if (modelParameter != parameter) {
            try (NDArray flatModelParameter = modelParameter.get(slice.index())) {
              modelParameterSlices.add(flatModelParameter.reshape(slice.shape()));
            }
          }
        }
        if (manager.getDevice().isGpu()) {
          PtEngine engine = (PtEngine) manager.getEngine();
          createdCopyPlan =
              engine.newTensorCopyPlan(new NDList(modelParameterSlices), packedParameters);
          parameterCopyPlan = createdCopyPlan;
          gradientPacker = engine.newFlatGradientPacker(packedParameters, packingGradient);
        } else {
          parameterCopyPlan = null;
          gradientPacker = null;
        }
      } catch (RuntimeException | Error failure) {
        if (createdCopyPlan != null) {
          createdCopyPlan.close();
        }
        if (modelParameterSlices != parameterSlices) {
          closeViews(modelParameterSlices, failure);
        }
        closeViews(parameterSlices, failure);
        closeViews(cpuGradientSlices, failure);
        if (modelParameter != parameter) {
          modelParameter.close();
        }
        parameter.close();
        if (packingGradient != gradient) {
          packingGradient.close();
        }
        gradient.close();
        throw failure;
      }
    }

    private NDArray gradient() {
      return gradient;
    }

    private NDArray packedGradient() {
      return packingGradient;
    }

    /** パラメーター精度の固定バッファへ連結し、そのままデバイス間転送する。 */
    private void packAndClearGradients(boolean allowMissing) {
      if (gradientPacker != null) {
        gradientPacker.packAndClear(
            allowMissing
                ? PtFlatGradientPacker.MissingGradientPolicy.ZERO
                : PtFlatGradientPacker.MissingGradientPolicy.ERROR);
        return;
      }
      for (int index = 0; index < sourceParameters.size(); index++) {
        NDArray source = sourceParameters.get(index);
        NDArray destination = cpuGradientSlices.get(index);
        if (!source.hasGradient()) {
          if (!allowMissing)
            throw new IllegalStateException("Missing CPU training gradient at parameter " + index);
          destination.fillI(0);
          continue;
        }
        try (NDArray sourceGradient = source.getGradient()) {
          sourceGradient.copyTo(destination);
          sourceGradient.fillI(0);
        }
      }
    }

    /** 連結した勾配を、オプティマイザーが使う精度のバッファへ変換する。両者が同じバッファなら何もしない。 */
    private void prepareOptimizerGradient() {
      if (packingGradient != gradient) {
        packingGradient.copyTo(gradient);
      }
    }

    private NDArray parameter() {
      return parameter;
    }

    private NDArray modelParameter() {
      return modelParameter;
    }

    private NDArray parameterSlice(int ordinal) {
      return parameterSlices.get(ordinal);
    }

    private void synchronizeModelParameters() {
      if (parameterCopyPlan != null) parameterCopyPlan.copy();
      else {
        for (int index = 0; index < sourceParameters.size(); index++) {
          modelParameterSlices.get(index).copyTo(sourceParameters.get(index));
        }
      }
    }

    @Override
    public void close() {
      if (parameterCopyPlan != null) parameterCopyPlan.close();
      if (gradientPacker != null) gradientPacker.close();
      cpuGradientSlices.forEach(NDArray::close);
      if (modelParameterSlices != parameterSlices) {
        modelParameterSlices.forEach(NDArray::close);
      }
      parameterSlices.forEach(NDArray::close);
      if (modelParameter != parameter) {
        modelParameter.close();
      }
      parameter.close();
      if (packingGradient != gradient) {
        packingGradient.close();
      }
      gradient.close();
    }

    private static void closeViews(List<NDArray> views, Throwable failure) {
      for (int index = views.size() - 1; index >= 0; index--) {
        try {
          views.get(index).close();
        } catch (RuntimeException closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
    }
  }
}
