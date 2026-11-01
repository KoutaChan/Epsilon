package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.engine.PtFlatGradientPacker;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.hip.HipAttention;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.TrainingKernels;
import com.epsilon.prism.model.AttentionTape;
import com.epsilon.prism.model.FitProbe;
import com.epsilon.prism.model.FitProbe.Stage;
import com.epsilon.prism.model.LearningRateSchedule;
import com.epsilon.prism.model.Outputs;
import com.epsilon.prism.model.Pass;
import com.epsilon.prism.model.Precision;
import com.epsilon.prism.model.PrismLoss;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.StagedBatch;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.training.DataParallelGroup;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.LongAdder;

/**
 * 1 回の当てはめの学習器。デバイスごとのレーンで微小バッチの順伝播と逆伝播をし、勾配を集めて AdamW（重みの
 * 減衰 0、FP32 の連続バッファ）で 1 歩進める。レーン 0 が正本の網で、残りはデータ並列の複製。GPU のレーンは BF16、CPU のレーンは FP32 で
 * 計算する。GPU のレーンの背骨の注意と正規化は、組み立てが libprism_hip.so を同梱していれば（{@link HipLibrary#bundled}）自前の HIP の演算
 * （{@link AttentionTape}）をレーンの計算ストリームに積み、同梱していなければ djl-rocm の演算。CPU のレーンは djl の演算。
 *
 * <p>1 歩の損失は Σ_微小バッチ {@link PrismLoss#total}（分母 B_C は 1 歩の全てのレーンで数え、補助の項は微小バッチの元の塊の行の割合
 * {@link Microbatch#share} とゲイン K′ を掛ける。replay は K′ = 1）。勾配のノルムは上限で切り、切った歩数を見積もる。
 *
 * <p>教師（錨の s を混ぜたもの。{@link Teacher}）は微小バッチを作るときに出来ている。網の入力・索引・損失の入力（教師を含む）は歩の
 * 作り手がページ固定の領域に前もって並べてあり（{@link StagedBatch}）、レーンは型ごとに 1 回の非同期の写しを計算ストリームに積むだけに
 * する（{@link Microbatch#send}）。ホストは写しも索引の組み立ても待たずに順伝播・損失・逆伝播を積み続け、写しはストリームの順で順伝播の
 * 前に終わる。
 *
 * <p>温め（{@link #rehearse}）は最初の歩の前に、レーンと複製を作り、各レーンで微小バッチを 1 つ通して核と作業域を用意する。
 *
 * <p>当てはめの計測（{@link FitProfile}）では、レーンごとの時計（{@link FitProfile.Clock}）を順伝播の種類と注意の記録に渡し、微小バッチの
 * 段（転送・網と逆伝播の段・損失・閉じる）に印を付ける。計測しないときの時計は何もしない。
 */
final class Learner implements AutoCloseable {

  /** 損失をホストへ回収する歩の間隔（GPU。同期を減らし、報告する損失はその歩の標本）。 */
  static final int METRIC_EVERY = 20;

  private static final float WEIGHT_DECAY = 0f;

  private final Model canonical;
  private final PrismNetwork network;
  private final NetworkDevices devices;
  private final LearningRateSchedule schedule;
  private final float gradientClip;
  private final PrismLoss loss;
  private final Optimizer optimizer;

  /** 補助の頭の項に掛けるゲイン K′（{@link Teacher#gain}）。 */
  private final float gain;

  /** 当てはめの計測（無効なら時計は何もしない）。 */
  private final FitProfile profile;

  private final DecisionExecutionContext execution = new DecisionExecutionContext();
  private final LongAdder transferNanos = new LongAdder();
  private final LongAdder computeNanos = new LongAdder();
  private final LongAdder optimizerNanos = new LongAdder();
  private ParallelOptimizer parallel;
  private boolean pendingUpdates;
  private int step;

  /** 今の更新の学習率（{@link #train} が決め、AdamW の更新の中で読む）。 */
  private volatile float rate;

  /**
   * 学習器を作る。
   *
   * @param canonical 正本の網のモデル（ブロックは {@link PrismNetwork}、デバイスは devices の先頭）
   * @param devices 学習のデバイス（先頭が正本）
   * @param plan 当てはめの計画
   * @param gain 補助の頭の項に掛けるゲイン K′（{@link Teacher#gain}）
   * @param profile 当てはめの計測（レーンの時計を作り、計る歩の時間を足す）
   */
  Learner(Model canonical, NetworkDevices devices, FitPlan plan, float gain, FitProfile profile) {
    if (!(canonical.getBlock() instanceof PrismNetwork block)) {
      throw new IllegalArgumentException("the model does not hold a PRISM network");
    }
    if (!canonical.getNDManager().getDevice().equals(devices.primary())) {
      throw new IllegalArgumentException("the model is not on the primary learner device");
    }
    this.canonical = canonical;
    this.network = block;
    this.devices = devices;
    schedule = plan.schedule();
    gradientClip = plan.learner().gradientClip();
    loss = new PrismLoss(plan.auxiliary());
    this.gain = gain;
    this.profile = profile;
    optimizer =
        Optimizer.adamW()
            .optLearningRateTracker(
                new Tracker() {
                  @Override
                  public float getNewValue(int numUpdate) {
                    return rate;
                  }
                })
            .optWeightDecays(WEIGHT_DECAY)
            .build();
  }

  /**
   * 1 歩分で 1 回更新する。計測の同期する歩では、歩の始め・レーンの仕事の後・最適化の後に全てのレーンを待つ（{@link FitProfile.Timing}）。
   *
   * @param prepared 1 歩分の微小バッチ
   * @return 更新の前の損失（回収しない歩は NaN）
   */
  double train(StepFeed.Step prepared) {
    rate = (float) schedule.rate(++step);
    ParallelOptimizer group = parallel();
    FitProfile.Timing timing = profile.timing(step);
    List<Integer> active = new ArrayList<>();
    List<DataParallelGroup.LaneWork<NetworkLane, Double>> work = new ArrayList<>();
    for (int lane = 0; lane < prepared.lanes().size(); lane++) {
      List<Microbatch> parts = prepared.lanes().get(lane);
      if (parts.isEmpty()) {
        continue;
      }
      active.add(lane);
      work.add(worker -> backward(worker, parts, prepared, timing));
    }
    int[] lanes = active.stream().mapToInt(Integer::intValue).toArray();
    group.setActiveGradientLanes(lanes);
    try {
      timing.settle(group);
      long started = System.nanoTime();
      double value =
          group.executeAssigned(lanes, work).stream().mapToDouble(Double::doubleValue).sum();
      timing.settle(group);
      long computed = System.nanoTime();
      group.applyFlattenedOptimizerStep(optimizer);
      timing.settle(group);
      long updated = System.nanoTime();
      computeNanos.add(computed - started);
      optimizerNanos.add(updated - computed);
      profile.record(timing, prepared, computed - started, updated - computed);
      pendingUpdates = true;
      return value;
    } catch (IOException failure) {
      clearFailedStep(group, failure);
      throw new UncheckedIOException("PRISM learner execution failed", failure);
    } catch (RuntimeException | Error failure) {
      clearFailedStep(group, failure);
      throw failure;
    }
  }

  /**
   * 学習器を温める: レーンと複製を作り（{@link #parallel}）、各レーンで歩のそのレーンの最初の微小バッチの順伝播・損失・逆伝播を 1 回通して、
   * 勾配を捨てる。歩の数・学習率・最適化の状態・計測・損失の回収には触れず、乱数も使わない。捨てた後のパラメーターの .grad は未定義で、
   * 温めないときの最初の歩の前と同じ（{@link NetworkLane#forgetGradients}）。なので温めても当てた重みは変わらない。
   *
   * @param prepared 温めに使う歩（どの行でもよい）
   */
  void rehearse(StepFeed.Step prepared) {
    ParallelOptimizer group = parallel();
    List<Integer> active = new ArrayList<>();
    List<DataParallelGroup.LaneWork<NetworkLane, Boolean>> work = new ArrayList<>();
    for (int lane = 0; lane < prepared.lanes().size(); lane++) {
      List<Microbatch> parts = prepared.lanes().get(lane);
      if (parts.isEmpty()) {
        continue;
      }
      active.add(lane);
      work.add(worker -> rehearse(worker, parts.getFirst(), prepared.denominator()));
    }
    try {
      group.executeAssigned(active.stream().mapToInt(Integer::intValue).toArray(), work);
    } catch (IOException failure) {
      throw new UncheckedIOException("PRISM learner rehearsal failed", failure);
    }
  }

  /**
   * 1 つの微小バッチの順伝播・損失・逆伝播を通し、勾配を捨てる（段の時計に印を付けない）。返り値は使わない（レーンの仕事の結果は null に
   * できない）。
   */
  private Boolean rehearse(NetworkLane lane, Microbatch part, int denominator) {
    try (NDManager scope = lane.manager().newSubManager();
        GradientCollector collector = lane.newGradientCollector()) {
      StagedBatch.Sent sent = part.send(scope, lane.computeStream());
      AttentionTape tape =
          lane.attention == null ? null : new AttentionTape(lane.attention, FitProbe.NONE);
      Outputs outputs =
          lane.network.forward(
              new ParameterStore(scope, false),
              sent.inputs(),
              Pass.training(lane.precision, part.measured(), part.auxiliary(), tape, FitProbe.NONE));
      NDArray value = loss.total(outputs, sent.loss(), gain, denominator, part.share());
      backward(collector, tape, value, FitProbe.NONE);
    }
    lane.forgetGradients();
    return Boolean.TRUE;
  }

  /** レーンの微小バッチを順に通し、勾配を溜める。返り値はその損失の和（回収しない歩は NaN）。 */
  private double backward(
      NetworkLane lane, List<Microbatch> parts, StepFeed.Step prepared, FitProfile.Timing timing) {
    double total = 0;
    boolean gpu = lane.manager().getDevice().isGpu();
    FitProfile.Clock clock = lane.clock;
    clock.begin(timing);
    for (Microbatch part : parts) {
      try (NDManager scope = lane.manager().newSubManager();
          GradientCollector collector = lane.newGradientCollector()) {
        long started = System.nanoTime();
        StagedBatch.Sent sent = part.send(scope, lane.computeStream());
        transferNanos.add(System.nanoTime() - started);
        clock.mark(Stage.TRANSFER);
        AttentionTape tape =
            lane.attention == null ? null : new AttentionTape(lane.attention, clock);
        Outputs outputs =
            lane.network.forward(
                new ParameterStore(scope, false),
                sent.inputs(),
                Pass.training(lane.precision, part.measured(), part.auxiliary(), tape, clock));
        NDArray value =
            loss.total(outputs, sent.loss(), gain, prepared.denominator(), part.share());
        clock.mark(Stage.LOSS);
        if (!gpu) {
          collector.backward(value);
          clock.mark(Stage.BACKWARD);
          total += value.getFloat();
        } else if (step % METRIC_EVERY == 1) {
          // 回収（ホストへの同期）は METRIC_EVERY 歩に 1 回。回収を逆伝播より先に予約する
          try (NDArray metric = value.reshape(1)) {
            try {
              lane.enqueueMetricReadback(metric);
              backward(collector, tape, value, clock);
              total += lane.awaitMetricReadback()[0];
            } finally {
              lane.closeMetricReadback();
            }
          }
        } else {
          backward(collector, tape, value, clock);
          total = Double.NaN;
        }
      }
      clock.mark(Stage.RELEASE);
    }
    return total;
  }

  /** 損失から逆伝播する（自前の注意の記録があれば記録が段に分けて印を付け、なければ全体を 1 つの段にする）。 */
  private static void backward(
      GradientCollector collector, AttentionTape tape, NDArray value, FitProbe probe) {
    if (tape == null) {
      collector.backward(value);
      probe.mark(Stage.BACKWARD);
    } else {
      tape.backward(collector, value);
    }
  }

  private static void clearFailedStep(ParallelOptimizer group, Throwable failure) {
    try {
      group.clearTrainingGradients();
    } catch (RuntimeException | Error cleanup) {
      failure.addSuppressed(cleanup);
    }
  }

  /**
   * 勾配のノルムを上限で切った歩数の見積もりを返す。
   *
   * @return 歩数（まだ歩を進めていなければ 0）
   */
  int clippedSteps() {
    return parallel == null ? 0 : parallel.clipped();
  }

  /**
   * 段階ごとの秒数を返す。
   *
   * @return {入力と損失の教師の転送, 順伝播と逆伝播, 最適化}
   */
  double[] stageSeconds() {
    return new double[] {
      transferNanos.sum() / 1e9, computeNanos.sum() / 1e9, optimizerNanos.sum() / 1e9
    };
  }

  /** データ並列のレーンを作る（1 回だけ）。 */
  private ParallelOptimizer parallel() {
    if (parallel != null) {
      return parallel;
    }
    List<NetworkLane> lanes = new ArrayList<>();
    ParallelOptimizer created = null;
    try {
      lanes.add(new NetworkLane(canonical, network, false, execution, profile));
      PrismShape shape = network.shape();
      for (int index = 1; index < devices.size(); index++) {
        Model replica =
            Model.newInstance("epsilon-prism-v2-learner", devices.get(index), "PyTorch");
        try {
          PrismNetwork copy = new PrismNetwork(shape);
          replica.setBlock(copy);
          copy.initialize(replica.getNDManager());
          DataParallelGroup.copyParameters(canonical, replica);
          lanes.add(new NetworkLane(replica, copy, true, execution, profile));
        } catch (RuntimeException | Error failure) {
          replica.close();
          throw failure;
        }
      }
      created = new ParallelOptimizer(lanes, execution, gradientClip);
      // 成功した更新の勾配は連続バッファへの詰めで消す。途中で失敗した歩の部分の勾配だけ別に捨てる
      created.clearTrainingGradients();
      parallel = created;
      return created;
    } catch (RuntimeException | Error failure) {
      if (created != null) {
        try {
          created.close();
        } catch (RuntimeException closing) {
          failure.addSuppressed(closing);
        }
      } else {
        for (NetworkLane lane : lanes) {
          try {
            lane.close();
          } catch (RuntimeException closing) {
            failure.addSuppressed(closing);
          }
        }
      }
      throw failure;
    }
  }

  /** 更新の完了を待ってから、データ並列の複製と連続バッファを捨てる（重みは正本の網に残る）。 */
  @Override
  public void close() {
    try {
      if (pendingUpdates) {
        parallel.awaitParameterUpdates();
        pendingUpdates = false;
      }
    } finally {
      try {
        if (parallel != null) {
          parallel.close();
        }
      } finally {
        execution.close();
      }
    }
  }

  /** 1 つのデバイスの網と計算の型（GPU なら計算ストリームに積む注意の演算も）と、段の時計。 */
  private static final class NetworkLane extends DataParallelGroup.Lane {

    private final PrismNetwork network;
    private final Precision precision;

    /** 背骨の注意と正規化の自前の演算（libprism_hip.so を同梱した組み立ての GPU のレーンだけ。ほかは null で djl の演算）。 */
    private final TrainingKernels attention;

    /** 当てはめの計測の時計（計測しないなら何もしない）。 */
    private final FitProfile.Clock clock;

    /**
     * レーンを作る。
     *
     * @param profile 当てはめの計測（レーンの時計を作る）
     */
    NetworkLane(
        Model model,
        PrismNetwork network,
        boolean ownsModel,
        DecisionExecutionContext execution,
        FitProfile profile) {
      super(model, ownsModel, false, execution);
      this.network = network;
      Device device = model.getNDManager().getDevice();
      precision = device.isGpu() ? Precision.BFLOAT16 : Precision.FLOAT32;
      attention =
          device.isGpu() && HipLibrary.bundled()
              ? HipAttention.on(HipLibrary.load(), computeStream())
              : null;
      clock = profile.clock(computeStream());
    }

    /**
     * パラメーターの .grad を未定義に戻す（レーンの計算ストリームの上で呼ぶ）。0 で埋めると次の逆伝播が 0 に足す形になり、勾配の −0 が +0 に
     * 変わり得るので、未定義に戻して温めないときと同じ状態にする。勾配の詰め手の置き場は .grad を戻すためだけに作り、すぐに捨てる。
     */
    void forgetGradients() {
      NDList parameters = new NDList();
      long elements = 0;
      for (Parameter parameter : network.getParameters().values()) {
        NDArray array = parameter.getArray();
        parameters.add(array);
        elements += array.size();
      }
      PtEngine engine = (PtEngine) manager().getEngine();
      try (NDArray destination = manager().zeros(new Shape(elements), DataType.BFLOAT16);
          PtFlatGradientPacker packer = engine.newFlatGradientPacker(parameters, destination)) {
        packer.clearParameterGradients();
      }
    }

    @Override
    protected void closeRuntime() {
      clock.close();
    }
  }

  /**
   * FP32 の連続バッファの AdamW の勾配の準備。勾配のノルムをデバイスの上で測って上限で切り、切った歩数を
   * 見積もる（報告だけ）。全てのパラメーターを更新する。
   */
  private static final class ParallelOptimizer extends DataParallelGroup<NetworkLane> {

    private final float gradientClip;
    private int clipped;
    private int steps;

    ParallelOptimizer(
        List<NetworkLane> lanes, DecisionExecutionContext execution, float gradientClip) {
      super(lanes, ParameterScope.ALL, null, DataType.FLOAT32, execution, (scope, name) -> true);
      this.gradientClip = gradientClip;
      recordParameterReadiness();
    }

    @Override
    protected boolean allowsZeroGradientUpdates() {
      return true;
    }

    @Override
    protected void prepareOptimizerGradient(ParameterScope scope, NDArray gradient) {
      steps++;
      if (gradientClip <= 0) {
        return;
      }
      try (NDManager temporary = gradient.getManager().newSubManager()) {
        // 倍率 min(1, 上限 / ノルム) をデバイスの上で作って掛ける（ホストへ値を戻さない）
        NDArray squared = gradient.square();
        squared.attach(temporary);
        NDArray norm = squared.sum().sqrt();
        norm.attach(temporary);
        NDArray factor = norm.maximum(gradientClip).pow(-1).mul(gradientClip);
        factor.attach(temporary);
        if (steps % METRIC_EVERY == 1) {
          float value = norm.getFloat();
          if (!Float.isFinite(value)) {
            throw new IllegalStateException("Nonfinite gradient norm");
          }
          if (value > gradientClip) {
            clipped += METRIC_EVERY;
          }
        }
        gradient.muli(factor);
      }
    }

    int clipped() {
      return clipped;
    }
  }
}
