package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.backend.HipBackend;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.TrainingKernels;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.model.FrozenNetwork;
import com.epsilon.prism.model.NetworkInputs;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.policy.InferencePath;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * アンカー（学習器のネットワーク θ_t を凍結したもの）の s を、1 つのデバイスで求める。学習の前に全ての行の s を求める
 * {@link AnchorScores} が使う。推論のバックエンドは方策と同じく {@link InferencePath} で選ぶ。HIP なら収集の方策と同じ自前の推論を使い
 * （渡したストリームに計画を 1 つ開く）、そうでなければ凍結したネットワークを eager で動かす。どちらも 1 回の呼び出しの中でデバイスでの
 * 処理を終え、s をホストに返す。スレッドセーフではない（同時に使うのは 1 つのスレッドだけ）。
 */
sealed interface Anchor extends AutoCloseable permits Anchor.Eager, Anchor.Hip {

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
   * @param path 推論のバックエンド（設定の inference）。GPU で、専用のカーネルがある幅のときだけ HIP になる
   * @param device デバイス
   * @param stream 計算ストリーム（HIP で推論するときに使う。CPU なら null）
   * @param attention バックボーンの Attention と正規化を計算する自前の演算（eager で推論するときに使う。null なら djl-rocm の演算）
   * @param capacity マイクロバッチの容量
   * @return アンカー
   * @throws IOException アンカーのネットワークを読めないとき
   */
  static Anchor open(
      Path directory,
      InferencePath path,
      Device device,
      PtStream stream,
      TrainingKernels attention,
      BatchPlan.Capacity capacity)
      throws IOException {
    PrismShape shape = PrismNetwork.readShape(directory);
    if (path.resolve(device, shape) == InferencePath.HIP) {
      return Hip.open(directory, Objects.requireNonNull(stream, "stream"), capacity);
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

  /** 自前の HIP の推論（収集の方策と同じカーネル）。s はバッチの中の手の順に並ぶ。 */
  final class Hip implements Anchor {

    /** アンカーの重みのバージョン。計画に写したバージョンと照合するためだけに使う。 */
    private static final long VERSION = 0;

    private final HipBackend backend;

    private Hip(HipBackend backend) {
      this.backend = backend;
    }

    /** ネットワークを CPU に読み込んで重みの束を作り、渡したストリームに計画を開いて重みを写す。 */
    static Hip open(Path directory, PtStream stream, BatchPlan.Capacity capacity) throws IOException {
      WeightPack pack = WeightPack.load(directory, VERSION);
      HipBackend backend = HipBackend.open(HipLibrary.load(), stream, pack.layout(), capacity);
      try {
        backend.publish(pack);
        return new Hip(backend);
      } catch (RuntimeException | Error failure) {
        try {
          backend.close();
        } catch (RuntimeException | Error problem) {
          failure.addSuppressed(problem);
        }
        throw failure;
      }
    }

    @Override
    public Scores scores(HostInputs inputs) {
      BatchPlan plan = inputs.plan();
      int[] starts = new int[plan.rows()];
      for (int row = 0; row < starts.length; row++) {
        starts[row] = plan.firstAction(row);
      }
      return new Scores(backend.forward(inputs.input(), VERSION), starts);
    }

    @Override
    public void close() {
      backend.close();
    }
  }
}
