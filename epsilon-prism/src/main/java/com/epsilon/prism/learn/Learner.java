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
import ai.djl.pytorch.engine.PtStream;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.epsilon.agent.learn.DataParallelGroup;
import com.epsilon.agent.learn.DataParallelGroup.Update;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.hip.HipAttention;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.TrainingKernels;
import com.epsilon.prism.model.AttentionTape;
import com.epsilon.prism.model.FitProbe;
import com.epsilon.prism.model.FitProbe.Stage;
import com.epsilon.prism.model.LearningRateSchedule;
import com.epsilon.prism.model.Outputs;
import com.epsilon.prism.model.Pass;
import com.epsilon.prism.model.PrismLoss;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.StagedBatch;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.LongAdder;

/**
 * 1 回の学習で使う学習器。デバイスごとのレーンでマイクロバッチの順伝播と逆伝播を行い、勾配を集めて AdamW（重みの
 * 減衰 0、FP32 の連続バッファ）で 1 ステップ進める。レーン 0 がマスターのネットワークで、残りはデータ並列のレプリカ。GPU のレーンは BF16、
 * CPU のレーンは FP32 で計算する。ビルドに libprism_hip.so が同梱されていれば（{@link HipLibrary#bundled}）、GPU のレーンはバックボーンの
 * Attention と正規化に自前の HIP の演算（{@link AttentionTape}）を使い、レーンの計算ストリームに積む。同梱されていなければ djl-rocm の演算を
 * 使う。CPU のレーンは djl の演算を使う。
 *
 * <p>1 ステップの損失は、マイクロバッチごとの {@link PrismLoss#total} の和である。差分整合の分母 B_C は、1 ステップの全てのレーンを通して
 * 数える。補助の項には、元のバッチの行が占める割合（{@link Microbatch#share}）とゲイン K′ を掛ける（replay では K′ = 1）。勾配のノルムは
 * 上限でクリップし、クリップしたステップ数を見積もる。
 *
 * <p>教師（アンカーの s を混ぜたもの。{@link Teacher}）は、マイクロバッチを作るときにできている。ネットワークの入力・インデックス・損失の
 * 入力（教師を含む）は、{@link StepFeed} がページ固定の領域に前もって並べておく（{@link StagedBatch}）。レーンは、型ごとに 1 回の非同期の
 * 転送をレーンの転送用のストリームに積むだけである（{@link Microbatch#send}）。マイクロバッチ i の計算を積む前に i+1 の転送を積む。
 * 計算ストリームは i の順伝播の直前に i の転送が終わるのだけを待つ（{@link StagedBatch.Upload#receive}）ので、i+1 の転送は i の計算と
 * 重なる。ホストは転送もインデックスの組み立ても待たずに、順伝播・損失・逆伝播を積み続ける。
 *
 * <p>ウォームアップ（{@link #rehearse}）では、最初のステップの前にレーンとレプリカを作り、各レーンでマイクロバッチを 1 つ通して、カーネルと
 * 作業領域を用意する。
 *
 * <p>学習の計測（{@link FitProfile}）では、レーンごとのタイマー（{@link FitProfile.Clock}）を順伝播と Attention の記録に渡し、
 * マイクロバッチの段（転送・ネットワークと逆伝播の段・損失・解放）に印を付ける。計測しないときのタイマーは何もしない。
 */
final class Learner implements AutoCloseable {

  /** GPU で損失をホストへ読み戻すステップの間隔。同期を減らすためで、報告する損失は読み戻したステップの値で代表させる。 */
  static final int METRIC_EVERY = 20;

  private static final float WEIGHT_DECAY = 0f;

  private final Model canonical;
  private final PrismNetwork network;
  private final NetworkDevices devices;
  private final LearningRateSchedule schedule;
  private final float gradientClip;
  private final PrismLoss loss;
  private final Optimizer optimizer;

  /** 補助のヘッドの項に掛けるゲイン K′（{@link Teacher#gain}）。 */
  private final float gain;

  /** 学習の計測（無効ならタイマーは何もしない）。 */
  private final FitProfile profile;

  /**
   * レーンの計算用と転送用のストリームを借りるランタイム。所有者は呼び出し元で、学習器は閉じない。系譜では反復をまたいで同じものを渡す。
   * PyTorch のキャッシュはストリームごとなので、反復ごとに別のストリームになると、前の反復の作業領域のキャッシュが使い回されずに残るからである。
   */
  private final NetworkRuntime runtime;

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
   * @param canonical マスターのネットワークのモデル（ブロックは {@link PrismNetwork}、デバイスは学習に使うデバイスの先頭）
   * @param devices 学習に使うデバイス（先頭がマスター）
   * @param plan 学習の計画
   * @param gain 補助のヘッドの項に掛けるゲイン K′（{@link Teacher#gain}）
   * @param profile 学習の計測（レーンのタイマーを作り、計測するステップの時間を足す）
   * @param runtime ストリームを借りるランタイム（閉じるのは呼び出し元）
   */
  Learner(
      Model canonical,
      NetworkDevices devices,
      FitPlan plan,
      float gain,
      FitProfile profile,
      NetworkRuntime runtime) {
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
    this.runtime = runtime;
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
   * 1 ステップ分のマイクロバッチで 1 回更新する。計測のために同期するステップでは、ステップの始め・レーンの処理の後・最適化の後に、
   * 全てのレーンを待つ（{@link FitProfile.Timing}）。
   *
   * @param prepared 1 ステップ分のマイクロバッチ
   * @return 更新の前の損失（読み戻さないステップは NaN）
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
    try {
      timing.settle(group);
      long started = System.nanoTime();
      double value = group.execute(lanes, work).stream().mapToDouble(Double::doubleValue).sum();
      timing.settle(group);
      long computed = System.nanoTime();
      group.step(lanes, List.of(new Update(ParameterSet.ALL, optimizer, 1f)));
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
   * 学習器をウォームアップする。レーンとレプリカを作り（{@link #parallel}）、各レーンで、そのレーンの最初のマイクロバッチの順伝播・損失・
   * 逆伝播を 1 回通して勾配を捨てる。ステップ数・学習率・最適化の状態・計測・損失の読み戻しには触れず、乱数も使わない。捨てた後の
   * パラメーターの .grad は未定義になり、ウォームアップしないときの最初のステップの前と同じ状態になる（{@link NetworkLane#forgetGradients}）。
   * そのため、ウォームアップしても学習した重みは変わらない。
   *
   * @param prepared ウォームアップに使うステップ（どの行でもよい）
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
      group.execute(active.stream().mapToInt(Integer::intValue).toArray(), work);
    } catch (IOException failure) {
      throw new UncheckedIOException("PRISM learner rehearsal failed", failure);
    }
  }

  /**
   * 1 つのマイクロバッチの順伝播・損失・逆伝播を通し、勾配を捨てる。段のタイマーには印を付けない。戻り値は使わないが、レーンの処理の結果は
   * null にできないので常に {@code TRUE} を返す。
   */
  private Boolean rehearse(NetworkLane lane, Microbatch part, int denominator) {
    Pending pending = upload(lane, part);
    try (NDManager scope = pending.scope();
        GradientCollector collector = lane.newGradientCollector()) {
      StagedBatch.Sent sent = pending.upload().receive();
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

  /**
   * レーンのマイクロバッチを順に通し、勾配を溜める。戻り値は損失の和（読み戻さないステップは NaN）。
   *
   * <p>マイクロバッチ i の順伝播を積む前に、i+1 の転送を転送用のストリームに積む（{@link #upload}）。i+1 の配列は i+1 のマネージャーが
   * 持つので、i のマネージャーを閉じた後も、i+1 の計算を積み終えるまで残る。途中で失敗したら、転送を積んだだけの i+1 のマネージャーも閉じる。
   * 計算に使っていない配列なので、そのメモリは転送用のストリームの順で転送が済んだ後に使い回される。
   */
  private double backward(
      NetworkLane lane, List<Microbatch> parts, StepFeed.Step prepared, FitProfile.Timing timing) {
    double total = 0;
    boolean gpu = lane.manager().getDevice().isGpu();
    FitProfile.Clock clock = lane.clock;
    clock.begin(timing);
    long started = System.nanoTime();
    Pending next = upload(lane, parts.getFirst());
    transferNanos.add(System.nanoTime() - started);
    try {
      for (int index = 0; index < parts.size(); index++) {
        Microbatch part = parts.get(index);
        Pending current = next;
        next = null;
        try (NDManager scope = current.scope();
            GradientCollector collector = lane.newGradientCollector()) {
          long sending = System.nanoTime();
          if (index + 1 < parts.size()) {
            next = upload(lane, parts.get(index + 1));
          }
          StagedBatch.Sent sent = current.upload().receive();
          transferNanos.add(System.nanoTime() - sending);
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
            // 損失の読み戻し（ホストとの同期）は METRIC_EVERY ステップに 1 回だけ。読み戻しは逆伝播より先に予約しておく
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
    } finally {
      if (next != null) {
        next.scope().close();
      }
    }
    return total;
  }

  /**
   * 新しいマネージャーを作り、マイクロバッチの転送をレーンの転送用のストリームに積む。失敗したらそのマネージャーを閉じる。
   *
   * @return 積んだ転送と、転送先の配列を持つマネージャー
   */
  private static Pending upload(NetworkLane lane, Microbatch part) {
    NDManager scope = lane.manager().newSubManager();
    try {
      return new Pending(scope, part.send(scope, lane.copyStream));
    } catch (RuntimeException | Error failure) {
      scope.close();
      throw failure;
    }
  }

  /** 損失から逆伝播する。自前の Attention の記録があれば記録が段ごとに印を付け、なければ逆伝播の全体を 1 つの段として印を付ける。 */
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
      group.clearGradients();
    } catch (RuntimeException | Error cleanup) {
      failure.addSuppressed(cleanup);
    }
  }

  /**
   * 勾配のノルムを上限でクリップしたステップ数の見積もりを返す。
   *
   * @return ステップ数（まだ 1 ステップも進めていなければ 0）
   */
  int clippedSteps() {
    return parallel == null ? 0 : parallel.clipped();
  }

  /**
   * 段ごとにかかった秒数を返す。
   *
   * @return {入力と損失の教師の転送, 順伝播と逆伝播, 最適化} の秒数
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
      lanes.add(new NetworkLane(canonical, network, false, runtime, profile));
      PrismShape shape = network.shape();
      for (int index = 1; index < devices.size(); index++) {
        Model replica =
            Model.newInstance("epsilon-prism-v2-learner", devices.get(index), "PyTorch");
        try {
          PrismNetwork copy = new PrismNetwork(network.format(), shape);
          replica.setBlock(copy);
          copy.initialize(replica.getNDManager());
          DataParallelGroup.copyParameters(canonical, replica);
          lanes.add(new NetworkLane(replica, copy, true, runtime, profile));
        } catch (RuntimeException | Error failure) {
          replica.close();
          throw failure;
        }
      }
      // 更新に成功したステップの勾配は、連続バッファへ詰めるときに消える。途中で失敗したステップの残った勾配だけを別に捨てる
      created = new ParallelOptimizer(lanes, gradientClip);
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

  /**
   * 更新が済むのを待ってから、データ並列のレプリカ・連続バッファ・AdamW のモーメントを捨て、借りたストリームの処理が終わるのを待つ。
   * 重みはマスターのネットワークに残り、ストリームは閉じない。
   */
  @Override
  public void close() {
    try {
      if (pendingUpdates) {
        parallel.awaitParameterUpdates();
        pendingUpdates = false;
      }
    } finally {
      try (optimizer) {
        if (parallel != null) {
          parallel.close();
        }
      } finally {
        runtime.awaitIdle();
      }
    }
  }

  /**
   * 転送を積んだマイクロバッチ。
   *
   * @param scope 転送先の配列を持つマネージャー（そのマイクロバッチの計算を積み終えたら閉じる）
   * @param upload 積んだ転送
   */
  private record Pending(NDManager scope, StagedBatch.Upload upload) {}

  /**
   * 1 つのデバイスのレーン。ネットワークと計算の精度、GPU なら計算ストリームに積む Attention の演算、転送用のストリーム、段のタイマーを
   * 持つ。
   */
  private static final class NetworkLane extends DataParallelGroup.Lane {

    private final PrismNetwork network;
    private final Precision precision;

    /**
     * マイクロバッチの入力を転送するストリーム。GPU のレーンだけが持ち、学習器のランタイムにあるそのデバイスの h2d のストリームを使って、
     * 計算ストリームとは別に順に転送する。CPU のレーンでは null で、同期して転送する。
     */
    private final PtStream copyStream;

    /**
     * バックボーンの Attention と正規化の自前の演算。libprism_hip.so を同梱したビルドの GPU のレーンだけが持つ。ほかのレーンでは null で、
     * djl の演算を使う。
     */
    private final TrainingKernels attention;

    /** 学習の計測のタイマー（計測しないなら何もしない）。 */
    private final FitProfile.Clock clock;

    /**
     * レーンを作る。
     *
     * @param profile 学習の計測（レーンのタイマーを作る）
     */
    NetworkLane(
        Model model,
        PrismNetwork network,
        boolean ownsModel,
        NetworkRuntime runtime,
        FitProfile profile) {
      super(model, ownsModel, runtime);
      this.network = network;
      Device device = model.getNDManager().getDevice();
      precision = device.isGpu() ? Precision.BFLOAT16 : Precision.FLOAT32;
      copyStream = device.isGpu() ? runtime.streams(device).h2d() : null;
      attention =
          device.isGpu() && HipLibrary.bundled() ? HipAttention.on(HipLibrary.load(), computeStream()) : null;
      clock = profile.clock(computeStream());
    }

    /**
     * パラメーターの .grad を未定義に戻す（レーンの計算ストリームの上で呼ぶ）。0 で埋めると、次の逆伝播が 0 に足す形になり、勾配の −0 が
     * +0 に変わりうる。そのため未定義に戻し、ウォームアップしないときと同じ状態にする。勾配を詰めるパッカーとそのバッファは、.grad を
     * 戻すためだけに作ってすぐに捨てる。
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
   * 一緒に更新するパラメーターの組。全てのパラメーターを 1 つの組で更新する。補助の行の無いレーンには補助のヘッドの勾配が無く、どの行も
   * 補助のヘッドに届かないステップもあるので、勾配の欠けと 0 の勾配を許す。
   */
  private enum ParameterSet implements DataParallelGroup.ParameterGroup {
    ALL;

    @Override
    public boolean includes(String parameter) {
      return true;
    }

    @Override
    public boolean allowsMissingGradients() {
      return true;
    }

    @Override
    public boolean allowsZeroGradient() {
      return true;
    }
  }

  /**
   * FP32 の連続バッファで AdamW を行うデータ並列の組。勾配を準備するときに、勾配のノルムをデバイスの上で測って上限でクリップし、
   * クリップしたステップ数を見積もる（報告にだけ使う）。
   */
  private static final class ParallelOptimizer extends DataParallelGroup<NetworkLane> {

    private final float gradientClip;
    private int clipped;
    private int steps;

    ParallelOptimizer(List<NetworkLane> lanes, float gradientClip) {
      super(lanes);
      this.gradientClip = gradientClip;
    }

    @Override
    protected void prepareGradient(ParameterGroup group, NDArray gradient) {
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
