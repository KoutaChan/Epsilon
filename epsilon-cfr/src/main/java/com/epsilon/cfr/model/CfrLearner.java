package com.epsilon.cfr.model;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Autocast;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.nano.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionAutocast;
import com.epsilon.network.NetworkDevices;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.training.DataParallelGroup;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Objects;

/**
 * LCFRの回帰損失をNanoと同じ学習実行基盤へ接続する。
 *
 * <p>ワーカー、GPUストリーム、勾配集約、AdamWの一括更新、パラメーター配布は {@link DataParallelGroup}
 * が管理する。ここではLCFRの教師データと入力の対応だけを定義する。
 */
final class CfrLearner implements AutoCloseable {
  private final CfrModelConfig config;
  private final CfrModel.Role role;
  private final Model canonical;
  private final CfrNetwork network;
  private final NetworkDevices devices;
  private final Optimizer optimizer;
  private final DecisionExecutionContext execution = new DecisionExecutionContext();
  private final CfrObservationCache observations = new CfrObservationCache();
  private CfrOptimizerGroup<Lane> parallel;
  private boolean pendingUpdates;
  private boolean closed;

  CfrLearner(
      CfrModelConfig config,
      CfrModel.Role role,
      Model canonical,
      CfrNetwork network,
      NetworkDevices devices) {
    this.config = Objects.requireNonNull(config);
    this.role = Objects.requireNonNull(role);
    this.canonical = Objects.requireNonNull(canonical);
    this.network = Objects.requireNonNull(network);
    if (devices == null || !canonical.getNDManager().getDevice().equals(devices.primary())) {
      throw new IllegalArgumentException("Canonical CFR model must use the first learner device");
    }
    var unique = new HashSet<Device>();
    for (Device device : devices) {
      if (!unique.add(Objects.requireNonNull(device)))
        throw new IllegalArgumentException("Duplicate learner device");
    }
    this.devices = devices;
    optimizer =
        Optimizer.adamW()
            .optLearningRateTracker(Tracker.fixed(config.learningRate()))
            .optWeightDecays(config.weightDecay())
            .build();
  }

  CfrModel.TrainingMetrics train(List<CfrTrainingRow> rows, long nextUpdate) {
    ensureOpen();
    if (rows.isEmpty() || nextUpdate < 1) throw new IllegalArgumentException("Invalid CFR update");
    List<List<List<Input>>> assignments = assign(rows);
    CfrOptimizerGroup<Lane> group = parallel();
    List<Integer> active = new ArrayList<>();
    List<DataParallelGroup.LaneWork<Lane, Double>> work = new ArrayList<>();
    for (int lane = 0; lane < assignments.size(); lane++) {
      List<List<Input>> batches = assignments.get(lane);
      if (batches.isEmpty()) continue;
      active.add(lane);
      work.add(worker -> backward(worker, batches, rows.size()));
    }
    int[] lanes = active.stream().mapToInt(Integer::intValue).toArray();
    group.setActiveGradientLanes(lanes);
    try {
      double loss =
          group.executeAssigned(lanes, work).stream().mapToDouble(Double::doubleValue).sum();
      if (!Double.isFinite(loss)) throw new IllegalStateException("Nonfinite CFR objective");
      group.applyFlattenedOptimizerStep(optimizer);
      pendingUpdates = true;
      return new CfrModel.TrainingMetrics(rows.size(), loss, group.gradientNorm(), nextUpdate);
    } catch (IOException failure) {
      clearFailedStep(group, failure);
      throw new UncheckedIOException("CFR learner execution failed", failure);
    } catch (RuntimeException | Error failure) {
      clearFailedStep(group, failure);
      throw failure;
    }
  }

  private static void clearFailedStep(CfrOptimizerGroup<?> group, Throwable failure) {
    try {
      group.clearTrainingGradients();
    } catch (RuntimeException | Error cleanupFailure) {
      failure.addSuppressed(cleanupFailure);
    }
  }

  private List<List<List<Input>>> assign(List<CfrTrainingRow> rows) {
    List<Input> inputs = new ArrayList<>();
    for (var entry : CfrSampleMultiplicity.aggregate(rows)) {
      CfrTrainingRow row = entry.row();
      var observation = observations.get(row);
      var host = observation.host();
      int[] actions = row.legalActionIds();
      if (host.legalActionCount(0) != actions.length)
        throw new IllegalArgumentException("CFR target action count mismatch");
      for (int action = 0; action < actions.length; action++) {
        if (host.legalActionId(0, action) != actions[action])
          throw new IllegalArgumentException("CFR target action order mismatch");
      }
      inputs.add(new Input(row, observation, entry.count()));
    }
    List<List<List<Input>>> assignments = new ArrayList<>(devices.size());
    for (int lane = 0; lane < devices.size(); lane++) assignments.add(new ArrayList<>());
    List<List<Input>> batches = new ArrayList<>();
    var buckets = new LinkedHashMap<DecisionBucket, List<Input>>();
    for (Input input : inputs)
      buckets
          .computeIfAbsent(input.observation().host().bucket(), ignored -> new ArrayList<>())
          .add(input);
    // 重複集約前の抽選数から分割上限を決め、複数デバイスへの割り当ての偏りを抑える。
    int microBatchSize =
        Math.min(config.microBatchSize(), Math.ceilDiv(rows.size(), devices.size()));
    for (List<Input> bucket : buckets.values()) {
      for (int from = 0; from < bucket.size(); from += microBatchSize)
        batches.add(bucket.subList(from, Math.min(bucket.size(), from + microBatchSize)));
    }
    // 候補の容量区分を揃え、重いバッチから負荷の少ないデバイスへ割り当てる。
    batches.sort(Comparator.comparingLong(CfrLearner::estimatedWork).reversed());
    long[] work = new long[devices.size()];
    for (List<Input> batch : batches) {
      int lane = 0;
      for (int candidate = 1; candidate < work.length; candidate++)
        if (work[candidate] < work[lane]) lane = candidate;
      assignments.get(lane).add(batch);
      work[lane] += estimatedWork(batch);
    }
    return assignments;
  }

  private static long estimatedWork(List<Input> batch) {
    var bucket = batch.getFirst().observation().host().bucket();
    return batch.size()
        * (128L + (long) bucket.legalActionCapacity() * bucket.actionTransitionCapacity());
  }

  private double backward(Lane lane, List<List<Input>> batches, int totalRows) {
    double objective = 0;
    for (List<Input> part : batches) {
      try (NDManager scope = lane.manager().newSubManager();
          var collector = lane.newGradientCollector()) {
        var store = new ParameterStore(scope, true);
        NDArray loss = batchLoss(lane, scope, store, part, totalRows);
        float value;
        if (scope.getDevice().isGpu()) {
          // Nanoと同じく損失値の回収を逆伝播より先に予約し、ホスト側で逆伝播全体の完了を待たずに済むようにする。
          try (NDArray metric = loss.reshape(1)) {
            try {
              lane.enqueueMetricReadback(metric);
              collector.backward(loss);
              value = lane.awaitMetricReadback()[0];
            } finally {
              lane.closeMetricReadback();
            }
          }
        } else {
          collector.backward(loss);
          value = loss.getFloat();
        }
        if (!Float.isFinite(value)) throw new IllegalStateException("Nonfinite CFR objective");
        objective += value;
      }
    }
    return objective;
  }

  private NDArray batchLoss(
      Lane lane, NDManager scope, ParameterStore store, List<Input> part, int totalRows) {
    var observations = part.stream().map(Input::observation).toList();
    var host = CfrObservationCodec.batch(observations);
    var batch =
        DecisionBatchTransfer.transferTrainingToDevice(
            scope, host.sliceRows(0, host.size()), lane.transfer(), config.tensorTransfer());
    int capacity = host.bucket().legalActionCapacity();
    float[] targets = new float[Math.multiplyExact(part.size(), capacity)];
    float[] multiplicities = new float[part.size()];
    for (int row = 0; row < part.size(); row++) {
      Input input = part.get(row);
      multiplicities[row] = input.count();
      double[] teacher =
          role == CfrModel.Role.REGRET ? input.row().regretTargets() : input.row().targetPolicy();
      for (int action = 0; action < teacher.length; action++) {
        float value = (float) teacher[action];
        if (!Float.isFinite(value))
          throw new IllegalArgumentException("Unrepresentable CFR teacher");
        targets[row * capacity + action] = value;
      }
    }
    NDArray target = scope.create(targets, new Shape(part.size(), capacity));
    NDArray multiplicity = scope.create(multiplicities, new Shape(part.size(), 1));
    NDArray scores;
    try (Autocast ignored =
        EpsilonDecisionAutocast.open(scope, config.precision(scope.getDevice()))) {
      scores = lane.network.scores(store, batch, true);
    }
    // 順伝播だけを混合精度で計算し、損失はFP32にする。toTypeで勾配を切らず、Nanoと同じ型昇格を使う。
    if (scores.getDataType() != DataType.FLOAT32)
      scores = scores.add(scope.zeros(new Shape(1), DataType.FLOAT32));
    NDArray mask = batch.legalActionMask().toType(DataType.FLOAT32, false);
    return objective(role, scores, target, mask, multiplicity, totalRows);
  }

  /** 抽出済みの出現回数だけを掛け、分母には全デバイスで共通の集約前の行数を使う。 */
  static NDArray objective(
      CfrModel.Role role,
      NDArray scores,
      NDArray target,
      NDArray mask,
      NDArray multiplicity,
      int totalRows) {
    return role == CfrModel.Role.REGRET
        ? scores.sub(target).square().mul(mask).mul(multiplicity).sum().div(totalRows)
        : scores
            .add(mask.neg().add(1).mul(-1e9f))
            .logSoftmax(1)
            .mul(target)
            .mul(multiplicity)
            .sum()
            .neg()
            .div(totalRows);
  }

  void saveOptimizer(Path path) throws IOException {
    ensureOpen();
    awaitModelReady();
    optimizer.saveState(path);
  }

  /** 学習とは別のストリームでの保存・推論・複製に先立つ一度だけの完了待ち。 */
  void awaitModelReady() {
    ensureOpen();
    if (pendingUpdates) {
      parallel.awaitParameterUpdates();
      pendingUpdates = false;
    }
  }

  void loadOptimizer(Path path) throws IOException {
    ensureOpen();
    optimizer.loadState(canonical.getNDManager(), path);
  }

  /** パラメーターを外から復元した後は、次の更新時に共通基盤の連続バッファも構築し直す。 */
  void synchronizeParameters() {
    ensureOpen();
    awaitModelReady();
    if (parallel != null) {
      parallel.close();
      parallel = null;
    }
  }

  private CfrOptimizerGroup<Lane> parallel() {
    if (parallel != null) return parallel;
    var lanes = new ArrayList<Lane>();
    CfrOptimizerGroup<Lane> created = null;
    try {
      lanes.add(new Lane(canonical, network, false, execution));
      for (int index = 1; index < devices.size(); index++) {
        Model replica = Model.newInstance("epsilon-cfr-learner", devices.get(index), "PyTorch");
        try {
          CfrNetwork replicaNetwork = new CfrNetwork(config);
          replica.setBlock(replicaNetwork);
          replicaNetwork.initialize(replica.getNDManager(), DataType.FLOAT32, new Shape(1, 1));
          DataParallelGroup.copyParameters(canonical, replica);
          lanes.add(new Lane(replica, replicaNetwork, true, execution));
        } catch (RuntimeException | Error failure) {
          replica.close();
          throw failure;
        }
      }
      created = new CfrOptimizerGroup<>(lanes, execution, config.gradientClip());
      // 成功した更新の勾配は共通のpackAndClearで消去する。途中で失敗した際の部分勾配だけ別途破棄する。
      created.clearTrainingGradients();
      parallel = created;
      return created;
    } catch (RuntimeException | Error failure) {
      if (created != null) {
        try {
          created.close();
        } catch (RuntimeException closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      } else {
        for (Lane lane : lanes) {
          try {
            lane.close();
          } catch (RuntimeException closeFailure) {
            failure.addSuppressed(closeFailure);
          }
        }
      }
      throw failure;
    }
  }

  private void ensureOpen() {
    if (closed) throw new IllegalStateException("CFR learner is closed");
  }

  @Override
  public void close() {
    if (closed) return;
    closed = true;
    try {
      if (parallel != null) parallel.close();
    } finally {
      try {
        observations.close();
      } finally {
        execution.close();
      }
    }
  }

  private record Input(CfrTrainingRow row, CfrObservationCodec.Decoded observation, int count) {}

  private static final class Lane extends DataParallelGroup.Lane {
    private final CfrNetwork network;
    private DecisionBatchTransfer.Workspace transfer;

    Lane(Model model, CfrNetwork network, boolean ownsModel, DecisionExecutionContext execution) {
      super(model, ownsModel, false, execution);
      this.network = network;
    }

    DecisionBatchTransfer.Workspace transfer() {
      if (transfer == null) transfer = new DecisionBatchTransfer.Workspace(manager());
      return transfer;
    }

    @Override
    public void close() {
      try {
        if (transfer != null) transfer.close();
      } finally {
        super.close();
      }
    }
  }
}
