package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.hip.HipAttention;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.RelationAttention;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.model.AttentionTape;
import com.epsilon.prism.model.FitProbe;
import com.epsilon.prism.model.FitProbe.Stage;
import com.epsilon.prism.model.LearningRateSchedule;
import com.epsilon.prism.model.LossInputs;
import com.epsilon.prism.model.NetworkInputs;
import com.epsilon.prism.model.Outputs;
import com.epsilon.prism.model.Pass;
import com.epsilon.prism.model.Precision;
import com.epsilon.prism.model.PrismLoss;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.Uploads;
import com.epsilon.prism.model.ValueLabel;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.training.DataParallelGroup;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.LongAdder;

/**
 * 1 回の当てはめの学習器。デバイスごとのレーンで微小バッチの順伝播と逆伝播をし、勾配を集めて AdamW（重みの
 * 減衰 0、FP32 の連続バッファ）で 1 歩進める。レーン 0 が正本の網で、残りはデータ並列の複製。GPU のレーンは BF16、CPU のレーンは FP32 で
 * 計算する。GPU のレーンの背骨の注意は、組み立てが libprism_hip.so を同梱していれば（{@link HipLibrary#bundled}）自前の HIP の演算
 * （{@link AttentionTape}）をレーンの計算ストリームに積み、同梱していなければ djl-rocm の演算。CPU のレーンは djl の演算。
 *
 * <p>1 歩の損失は Σ_微小バッチ {@link PrismLoss#total}（分母 B_C は 1 歩の全てのレーンで数え、補助の項は微小バッチの元の塊の行の割合
 * {@link Microbatch#share} とゲイン K′ を掛ける。replay は K′ = 1）。勾配のノルムは上限で切り、切った歩数を見積もる。
 *
 * <p>錨（{@link Teacher#anchor}）は、各レーンが自分のデバイスに読む（{@link Anchor}。道は {@link Teacher#anchorInference}）。
 * 微小バッチごとに、生徒の順伝播の前に錨の s を出してホストに戻してから教師に混ぜる。HIP の錨はレーンの計算ストリームに計画を 1 つ開き、
 * eager の錨は配列をその中で閉じるので、錨と生徒の配列は同時にデバイスに載らない。混ぜた教師は網の入力と一緒に順伝播の前にデバイスへ送る
 * （{@link LossInputs}）。順伝播の後に作ると配列の作成の同期でホストが順伝播の終わりまで待ち、その間に損失と逆伝播を積めない。
 *
 * <p>当てはめの計測（{@link FitProfile}）では、レーンごとの時計（{@link FitProfile.Clock}）を順伝播の種類と注意の記録に渡し、微小バッチの
 * 段（錨・混ぜる・転送・網と逆伝播の段・損失・閉じる）に印を付ける。計測しないときの時計は何もしない。
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

  /** 教師の作り方（錨の網のディレクトリ・錨の重み・補助の頭の項に掛けるゲイン K′・錨の推論の道）。 */
  private final Teacher teacher;

  /** 微小バッチの容量（HIP の錨の計画の大きさ）。 */
  private final BatchPlan.Capacity capacity;

  /** 当てはめの計測（無効なら時計は何もしない）。 */
  private final FitProfile profile;

  private final DecisionExecutionContext execution = new DecisionExecutionContext();
  private final LongAdder transferNanos = new LongAdder();
  private final LongAdder anchorNanos = new LongAdder();
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
   * @param teacher 教師の作り方（錨があれば各レーンが読む。補助の頭の項のゲインは {@link Teacher#gain}）
   * @param profile 当てはめの計測（レーンの時計を作り、計る歩の時間を足す）
   */
  Learner(Model canonical, NetworkDevices devices, FitPlan plan, Teacher teacher, FitProfile profile) {
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
    this.teacher = teacher;
    this.profile = profile;
    capacity = StepFeed.capacity(plan.tokenBudget());
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

  /** レーンの微小バッチを順に通し、勾配を溜める。返り値はその損失の和（回収しない歩は NaN）。 */
  private double backward(
      NetworkLane lane, List<Microbatch> parts, StepFeed.Step prepared, FitProfile.Timing timing) {
    double total = 0;
    boolean gpu = lane.manager().getDevice().isGpu();
    FitProfile.Clock clock = lane.clock;
    clock.begin(timing);
    for (Microbatch part : parts) {
      List<ValueLabel> labels = lane.anchor == null ? part.labels() : anchored(lane, part);
      try (NDManager scope = lane.manager().newSubManager();
          GradientCollector collector = lane.newGradientCollector()) {
        long started = System.nanoTime();
        NetworkInputs inputs = part.inputs(scope, lane.uploads);
        LossInputs targets = part.lossInputs(inputs, labels);
        transferNanos.add(System.nanoTime() - started);
        clock.mark(Stage.TRANSFER);
        AttentionTape tape =
            lane.attention == null ? null : new AttentionTape(lane.attention, clock);
        Outputs outputs =
            lane.network.forward(
                new ParameterStore(scope, false),
                inputs,
                Pass.training(lane.precision, part.measured(), part.auxiliary(), tape, clock));
        NDArray value =
            loss.total(outputs, targets, teacher.gain(), prepared.denominator(), part.share());
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

  /** レーンの錨で微小バッチの s を出し、教師に混ぜる。錨のデバイスの仕事はこの中で終わる（生徒の順伝播と重ならない）。 */
  private List<ValueLabel> anchored(NetworkLane lane, Microbatch part) {
    long started = System.nanoTime();
    Anchor.Scores scores = lane.anchor.scores(part.batch());
    anchorNanos.add(System.nanoTime() - started);
    lane.clock.mark(Stage.ANCHOR);
    List<ValueLabel> mixed = teacher.mix(part.labels(), scores);
    lane.clock.mark(Stage.MIX);
    return mixed;
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

  /**
   * 錨の s を出した秒数の和を返す（全てのレーンの和。順伝播と逆伝播の段に含まれる）。
   *
   * @return 秒
   */
  double anchorSeconds() {
    return anchorNanos.sum() / 1e9;
  }

  /** データ並列のレーンを作る（1 回だけ）。 */
  private ParallelOptimizer parallel() {
    if (parallel != null) {
      return parallel;
    }
    List<NetworkLane> lanes = new ArrayList<>();
    ParallelOptimizer created = null;
    try {
      Path anchor = teacher.anchored() ? teacher.anchor() : null;
      lanes.add(
          new NetworkLane(
              canonical, network, false, execution, anchor, teacher.anchorInference(), capacity, profile));
      PrismShape shape = network.shape();
      for (int index = 1; index < devices.size(); index++) {
        Model replica =
            Model.newInstance("epsilon-prism-v2-learner", devices.get(index), "PyTorch");
        try {
          PrismNetwork copy = new PrismNetwork(shape);
          replica.setBlock(copy);
          copy.initialize(replica.getNDManager());
          DataParallelGroup.copyParameters(canonical, replica);
          lanes.add(
              new NetworkLane(
                  replica, copy, true, execution, anchor, teacher.anchorInference(), capacity, profile));
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

  /** 1 つのデバイスの網と計算の型（GPU なら計算ストリームに積む注意の演算も）と、送り出しの置き場と、錨の網と、段の時計。 */
  private static final class NetworkLane extends DataParallelGroup.Lane {

    private final PrismNetwork network;
    private final Precision precision;

    /** 背骨の注意の自前の演算（libprism_hip.so を同梱した組み立ての GPU のレーンだけ。ほかは null で djl の演算）。 */
    private final RelationAttention attention;

    /** 当てはめの計測の時計（計測しないなら何もしない）。 */
    private final FitProfile.Clock clock;

    /** 微小バッチの索引と損失の入力をデバイスへ送る置き場（このレーンのスレッドだけが使う）。 */
    private final Uploads uploads;

    /** このデバイスに読んだ錨（錨がなければ null）。 */
    private final Anchor anchor;

    /**
     * レーンを作る。錨は最後に読む。
     *
     * @param anchorDirectory 錨の網のディレクトリ（錨がなければ null）
     * @param anchorInference 錨の s を出す推論の道
     * @param capacity 微小バッチの容量（HIP の錨の計画の大きさ）
     * @param profile 当てはめの計測（レーンの時計を作る）
     * @throws UncheckedIOException 錨の網を読めないとき
     */
    NetworkLane(
        Model model,
        PrismNetwork network,
        boolean ownsModel,
        DecisionExecutionContext execution,
        Path anchorDirectory,
        InferencePath anchorInference,
        BatchPlan.Capacity capacity,
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
      uploads = Uploads.on(manager());
      try {
        anchor =
            anchorDirectory == null
                ? null
                : Anchor.open(
                    anchorDirectory,
                    anchorInference,
                    device,
                    device.isGpu() ? computeStream() : null,
                    attention,
                    capacity);
      } catch (IOException failure) {
        throw new UncheckedIOException("cannot load the TEMPO anchor " + anchorDirectory, failure);
      }
    }

    @Override
    protected void closeRuntime() {
      try {
        if (anchor != null) {
          anchor.close();
        }
      } finally {
        try {
          uploads.close();
        } finally {
          clock.close();
        }
      }
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
