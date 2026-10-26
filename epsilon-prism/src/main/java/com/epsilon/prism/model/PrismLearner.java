package com.epsilon.prism.model;

import ai.djl.Model;
import ai.djl.engine.Autocast;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.PairList;
import com.epsilon.nano.ai.decision.input.DecisionBatchTransfer;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionAutocast;
import com.epsilon.nano.ai.model.EpsilonTileRelationEncoder;
import com.epsilon.network.NetworkDevices;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.training.DataParallelGroup;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.atomic.LongAdder;

/**
 * 1 回の当てはめの学習器。網の損失（{@link Objective}）を Nano と同じ学習実行基盤へ接続する。AdamW はこの学習器ごとに新しく作るので、反復をまたいで
 * 持つ最適化器の状態はない。レーンは FP32 で、レーン 0 が正本の網、残りはデータ並列の複製。
 *
 * <p>ワーカー、GPU ストリーム、勾配集約、AdamW の一括更新、パラメーター配布は {@link DataParallelGroup} が管理する。勾配のノルムの上限は当てはめの
 * 設定から受け取る（model.json の値は使わない）。損失は (1/B)·Σ_微小バッチ {@link Objective#total}（B は 1 歩の全レーンの {@link
 * Objective#contributes} の行数）。
 */
final class PrismLearner<E extends Example> implements AutoCloseable {
  /** 損失をホストへ回収する歩の間隔（同期を減らす。報告する損失はその歩の標本）。 */
  static final int METRIC_EVERY = 20;

  private final ModelConfig config;
  private final Model canonical;
  private final PrismNetwork network;
  private final NetworkDevices devices;
  private final LearningRateSchedule schedule;
  private final float gradientClip;
  private final Objective<E> objective;
  private final Optimizer optimizer;
  private final DecisionExecutionContext execution = new DecisionExecutionContext();
  private OptimizerGroup<Lane> parallel;
  private boolean pendingUpdates;
  private int step;

  /** 今の更新の学習率（{@link #train} が決め、AdamW の更新の中で読む）。 */
  private volatile float rate;

  /** 段階ごとの時間（報告だけ）: 行の結合（先読みスレッドの時間）、デバイスへの転送、順伝播と逆伝播、最適化の 1 歩。 */
  private final LongAdder packNanos = new LongAdder(), transferNanos = new LongAdder();

  private final LongAdder computeNanos = new LongAdder(), optimizerNanos = new LongAdder();

  PrismLearner(
      ModelConfig config,
      Model canonical,
      PrismNetwork network,
      NetworkDevices devices,
      LearningRateSchedule schedule,
      float gradientClip,
      Objective<E> objective) {
    this.config = config;
    this.canonical = canonical;
    this.network = network;
    this.devices = devices;
    this.schedule = schedule;
    this.gradientClip = gradientClip;
    this.objective = objective;
    optimizer =
        Optimizer.adamW()
            .optLearningRateTracker(
                new Tracker() {
                  @Override
                  public float getNewValue(int numUpdate) {
                    return rate;
                  }
                })
            .optWeightDecays(config.weightDecay())
            .build();
  }

  /** 1 歩分の行をレーンごとに割り当て、微小バッチごとに入力を結合する（ホスト側だけ。学習のスレッドでなくてよい）。 */
  PrismModel.Packed<E> pack(List<E> rows) {
    long started = System.nanoTime();
    List<List<List<E>>> assignments = assign(rows);
    List<List<PrismModel.Part<E>>> lanes = new ArrayList<>(assignments.size());
    for (List<List<E>> batches : assignments) {
      List<PrismModel.Part<E>> parts = new ArrayList<>(batches.size());
      for (List<E> part : batches)
        parts.add(
            new PrismModel.Part<>(
                part.size() == 1
                    ? part.getFirst().host()
                    : DecisionHostBatch.concatenate(part.stream().map(Example::host).toList()),
                part));
      lanes.add(parts);
    }
    packNanos.add(System.nanoTime() - started);
    int divisor = 0;
    for (E row : rows) if (objective.contributes(row)) divisor++;
    return new PrismModel.Packed<>(lanes, Math.max(1, divisor));
  }

  /** 結合済みの 1 歩分で 1 回の更新。返り値はその更新の前の損失（回収しない歩は NaN）。 */
  double train(PrismModel.Packed<E> packed) {
    rate = (float) schedule.rate(++step);
    OptimizerGroup<Lane> group = parallel();
    List<Integer> active = new ArrayList<>();
    List<DataParallelGroup.LaneWork<Lane, Double>> work = new ArrayList<>();
    for (int lane = 0; lane < packed.lanes().size(); lane++) {
      List<PrismModel.Part<E>> parts = packed.lanes().get(lane);
      if (parts.isEmpty()) continue;
      active.add(lane);
      work.add(worker -> backward(worker, parts, packed.rows()));
    }
    int[] lanes = active.stream().mapToInt(Integer::intValue).toArray();
    group.setActiveGradientLanes(lanes);
    try {
      long started = System.nanoTime();
      double loss =
          group.executeAssigned(lanes, work).stream().mapToDouble(Double::doubleValue).sum();
      long computed = System.nanoTime();
      group.applyFlattenedOptimizerStep(optimizer);
      computeNanos.add(computed - started);
      optimizerNanos.add(System.nanoTime() - computed);
      pendingUpdates = true;
      return loss;
    } catch (IOException failure) {
      clearFailedStep(group, failure);
      throw new UncheckedIOException("Prism learner execution failed", failure);
    } catch (RuntimeException | Error failure) {
      clearFailedStep(group, failure);
      throw failure;
    }
  }

  private static void clearFailedStep(OptimizerGroup<?> group, Throwable failure) {
    try {
      group.clearTrainingGradients();
    } catch (RuntimeException | Error cleanupFailure) {
      failure.addSuppressed(cleanupFailure);
    }
  }

  /** 容量区分ごとに微小バッチへ分け、重いバッチから負荷の少ないデバイスへ割り当てる。 */
  private List<List<List<E>>> assign(List<E> inputs) {
    List<List<List<E>>> assignments = new ArrayList<>(devices.size());
    for (int lane = 0; lane < devices.size(); lane++) assignments.add(new ArrayList<>());
    List<List<E>> batches = new ArrayList<>();
    var buckets = new LinkedHashMap<DecisionBucket, List<E>>();
    for (E input : inputs)
      buckets.computeIfAbsent(input.host().bucket(), ignored -> new ArrayList<>()).add(input);
    int microBatchSize =
        Math.min(config.microBatchSize(), Math.ceilDiv(inputs.size(), devices.size()));
    for (List<E> bucket : buckets.values()) {
      for (int from = 0; from < bucket.size(); from += microBatchSize)
        batches.add(bucket.subList(from, Math.min(bucket.size(), from + microBatchSize)));
    }
    batches.sort(Comparator.comparingLong((List<E> batch) -> estimatedWork(batch)).reversed());
    long[] work = new long[devices.size()];
    for (List<E> batch : batches) {
      int lane = 0;
      for (int candidate = 1; candidate < work.length; candidate++)
        if (work[candidate] < work[lane]) lane = candidate;
      assignments.get(lane).add(batch);
      work[lane] += estimatedWork(batch);
    }
    return assignments;
  }

  private static long estimatedWork(List<? extends Example> batch) {
    var bucket = batch.getFirst().host().bucket();
    return batch.size()
        * (128L + (long) bucket.legalActionCapacity() * bucket.actionTransitionCapacity());
  }

  private double backward(Lane lane, List<PrismModel.Part<E>> parts, int totalRows) {
    double total = 0;
    for (PrismModel.Part<E> part : parts) {
      try (NDManager scope = lane.manager().newSubManager();
          var collector = lane.newGradientCollector()) {
        var store = new ParameterStore(scope, true);
        NDArray loss = batchLoss(lane, scope, store, part, totalRows);
        float value;
        if (!scope.getDevice().isGpu()) {
          collector.backward(loss);
          value = loss.getFloat();
        } else if (step % METRIC_EVERY == 1) {
          // 損失の回収（ホストへの同期）は METRIC_EVERY 歩に 1 回。Nano と同じく回収を逆伝播より先に予約する
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
          value = Float.NaN;
        }
        total += value;
      }
    }
    return total;
  }

  /** (1/B)·{@link Objective#total}。B は 1 歩の全レーンで数えた分母（レーンをまたいで同じ尺度）。 */
  private NDArray batchLoss(
      Lane lane, NDManager scope, ParameterStore store, PrismModel.Part<E> part, int totalRows) {
    var host = part.host();
    long packed = System.nanoTime();
    var batch =
        DecisionBatchTransfer.transferTrainingToDevice(
            scope,
            host.sliceRows(0, host.size()),
            lane.transfer(),
            config.transfer(scope.getDevice()));
    transferNanos.add(System.nanoTime() - packed);
    int capacity = host.bucket().legalActionCapacity();
    NDArray scores;
    try (Autocast ignored =
        EpsilonDecisionAutocast.open(scope, config.precision(scope.getDevice()))) {
      scores = lane.network.scores(store, batch, true, lane.runtime());
    }
    // 出力の頭は FP32 なので s は FP32。念のため、混合精度の出力なら toType で勾配を切らず、Nano と同じ型昇格を使う
    if (scores.getDataType() != DataType.FLOAT32)
      scores = scores.add(scope.zeros(new Shape(1), DataType.FLOAT32));
    return objective
        .total(scope, scores, batch.legalActionMask(), part.rows(), capacity)
        .div(totalRows);
  }

  /** 勾配のノルムを上限で切った歩数の見積もり（まだ歩を進めていなければ 0）。 */
  int clippedSteps() {
    return parallel == null ? 0 : parallel.clipped();
  }

  /** 学習とは別のストリームでの保存・推論・複製に先立つ一度だけの完了待ち。 */
  void awaitModelReady() {
    if (pendingUpdates) {
      parallel.awaitParameterUpdates();
      pendingUpdates = false;
    }
  }

  private OptimizerGroup<Lane> parallel() {
    if (parallel != null) return parallel;
    var lanes = new ArrayList<Lane>();
    OptimizerGroup<Lane> created = null;
    try {
      lanes.add(new Lane(canonical, network, false, execution));
      for (int index = 1; index < devices.size(); index++) {
        Model replica = Model.newInstance("epsilon-prism-learner", devices.get(index), "PyTorch");
        try {
          PrismNetwork replicaNetwork = new PrismNetwork(config);
          replica.setBlock(replicaNetwork);
          replicaNetwork.initialize(replica.getNDManager(), DataType.FLOAT32, new Shape(1, 1));
          DataParallelGroup.copyParameters(canonical, replica);
          lanes.add(new Lane(replica, replicaNetwork, true, execution));
        } catch (RuntimeException | Error failure) {
          replica.close();
          throw failure;
        }
      }
      created = new OptimizerGroup<>(lanes, execution, gradientClip, name -> true);
      // 成功した更新の勾配は共通の packAndClear で消去する。途中で失敗した際の部分勾配だけ別途破棄する
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

  /** 段階ごとの秒数 {行の結合, 転送, 順伝播と逆伝播, 最適化}。 */
  double[] stageSeconds() {
    return new double[] {
      packNanos.sum() / 1e9,
      transferNanos.sum() / 1e9,
      computeNanos.sum() / 1e9,
      optimizerNanos.sum() / 1e9
    };
  }

  /** 更新の完了を待ってから、データ並列の複製と連続バッファを捨てる（重みは正本の網に残る）。 */
  @Override
  public void close() {
    try {
      awaitModelReady();
    } finally {
      try {
        if (parallel != null) parallel.close();
      } finally {
        execution.close();
      }
    }
  }

  private static final class Lane extends DataParallelGroup.Lane {
    private final PrismNetwork network;
    private DecisionBatchTransfer.Workspace transfer;
    private PairList<String, Object> runtime;

    Lane(
        Model model, PrismNetwork network, boolean ownsModel, DecisionExecutionContext execution) {
      super(model, ownsModel, false, execution);
      this.network = network;
    }

    /** Nano の実行時パラメータ（レーンごとに 1 回作る）。 */
    PairList<String, Object> runtime() {
      if (runtime == null) runtime = EpsilonTileRelationEncoder.createRuntimeParameters(manager());
      return runtime;
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
