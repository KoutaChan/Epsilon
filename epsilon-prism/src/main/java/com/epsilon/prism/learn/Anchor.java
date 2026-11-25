package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.backend.KernelInputs;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.kernel.KernelProvider;
import com.epsilon.prism.kernel.KernelProvider.Plan;
import com.epsilon.prism.kernel.TrainingKernels;
import com.epsilon.prism.model.FrozenNetwork;
import com.epsilon.prism.model.NetworkInputs;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.policy.CollectionPlans;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * アンカー（学習器のネットワーク θ_t を凍結したもの）の s を、1 つのデバイスで求める。学習の前に全ての行の s を求める
 * {@link AnchorScores} が使う。推論のバックエンドは方策と同じく {@link Kernels} で選ぶ。専用カーネルでは収集の方策と同じ実行計画を使い
 * （渡したストリームに計画を 1 つ開く）、そうでなければ凍結したネットワークを eager で動かす。どちらも 1 回の呼び出しの中でデバイスでの
 * 処理を終え、s をホストに返す。スレッドセーフではない（同時に使うのは 1 つのスレッドだけ）。
 */
sealed interface Anchor extends AutoCloseable permits Anchor.Eager, Anchor.Native {

  /**
   * 1 つのバッチについて、アンカーが出した s。
   *
   * @param values s（行 r の k 番目の手は {@code values[starts[r] + k]}）
   * @param starts 行ごとの先頭の位置
   */
  record Scores(float[] values, int[] starts) {

    /** 配列が null でないことを確かめる。 */
    public Scores {
      Objects.requireNonNull(values, "values");
      Objects.requireNonNull(starts, "starts");
    }
  }

  /**
   * バッチの s を求める。
   *
   * @param inputs 確定したバッチ
   * @return s
   */
  Scores scores(HostInputs inputs);

  @Override
  void close();

  /**
   * アンカーのネットワークを読み込む。
   *
   * @param directory アンカーのネットワークのディレクトリ
   * @param path 計算方法（設定の inference）。専用カーネルは方策と同じく {@link CollectionPlans} で選ぶ
   * @param device デバイス
   * @param stream 専用カーネルの計算ストリーム。CPU なら null
   * @param attention バックボーンの Attention と正規化を計算する自前の演算（eager で推論するときに使う。null なら DJL の演算）
   * @param capacity マイクロバッチの容量
   * @return アンカー
   * @throws IOException アンカーのネットワークを読めないとき
   * @throws IllegalStateException 専用カーネルが扱う形なのに、デバイスのバックエンドのプロバイダーが無いか使えないとき
   */
  static Anchor open(
      Path directory,
      Kernels path,
      Device device,
      PtStream stream,
      TrainingKernels attention,
      BatchPlan.Capacity capacity)
      throws IOException {
    if (path == Kernels.NATIVE && device.isGpu()) {
      var header = PrismNetwork.readHeader(directory);
      var network = new PrismNetwork(header.format(), header.shape());
      var selected =
          CollectionPlans.select(network, List.of(device), Precision.BFLOAT16, capacity, null);
      if (selected.isPresent()) {
        var lane = selected.get().devices().get(device);
        return Native.open(
            directory,
            network,
            Objects.requireNonNull(stream, "stream"),
            lane.provider(),
            lane.request());
      }
    }
    Precision precision = device.isGpu() ? Precision.BFLOAT16 : Precision.FLOAT32;
    return new Eager(FrozenNetwork.load(directory, device, precision), attention);
  }

  /** eager で動かす凍結したネットワーク。s は [行, 手の容量] を行優先に平らにした形。インデックスを転送するバッファは、ネットワークに付けて 1 つだけ持つ。 */
  final class Eager implements Anchor {

    private final FrozenNetwork network;
    private final TrainingKernels attention;
    private final PinnedUploads uploads;

    Eager(FrozenNetwork network, TrainingKernels attention) {
      this.network = network;
      this.attention = attention;
      uploads = network.newUploads();
    }

    @Override
    public Scores scores(HostInputs inputs) {
      try (NDManager scope = network.newScope();
          InferenceMode inference = network.inference()) {
        NDArray s = network.scores(NetworkInputs.of(scope, uploads, inputs.input()), attention);
        int rows = Math.toIntExact(s.getShape().get(0));
        int capacity = Math.toIntExact(s.getShape().get(1));
        int[] starts = new int[rows];
        for (int row = 0; row < rows; row++) {
          starts[row] = row * capacity;
        }
        return new Scores(s.toFloatArray(), starts);
      }
    }

    @Override
    public void close() {
      network.close();
    }
  }

  /** 学習レーンと同じストリームにバッチを提出し、確定したスコアを合法手の順に読み出す。 */
  final class Native implements Anchor {
    private static final long VERSION = 0;
    private final Plan plan;
    private final PtStream stream;
    private final KernelInputs sections;
    private final ByteBuffer output;

    private Native(Plan plan, PtStream stream, KernelProvider.Request request) {
      this.plan = plan;
      this.stream = stream;
      sections = new KernelInputs(request.model());
      output =
          ByteBuffer.allocateDirect(Math.multiplyExact(request.capacity().actions(), Float.BYTES))
              .order(ByteOrder.LITTLE_ENDIAN);
    }

    static Native open(
        Path directory,
        PrismNetwork network,
        PtStream stream,
        KernelProvider provider,
        KernelProvider.Request request)
        throws IOException {
      try (var manager = NDManager.newBaseManager(Device.cpu())) {
        network.readWeights(directory, manager);
        var pack = WeightPack.publication(network, VERSION);
        Plan plan = provider.open(request, stream.getId());
        try {
          plan.publish(pack);
          return new Native(plan, stream, request);
        } catch (RuntimeException | Error failure) {
          try {
            plan.close();
          } catch (RuntimeException | Error problem) {
            failure.addSuppressed(problem);
          }
          throw failure;
        }
      }
    }

    @Override
    public Scores scores(HostInputs inputs) {
      var batch = sections.batch(inputs.input(), output);
      inputs.input().materializeEvents();
      plan.submit(batch, VERSION);
      try (var event = stream.newEvent();
          var ignored = stream.openScope()) {
        event.record();
        event.synchronize();
      }
      float[] scores = new float[batch.actions()];
      output.asFloatBuffer().get(scores);
      int[] starts = new int[batch.rows()];
      for (int row = 0; row < starts.length; row++) starts[row] = inputs.plan().firstAction(row);
      return new Scores(scores, starts);
    }

    @Override
    public void close() {
      plan.close();
    }
  }
}
