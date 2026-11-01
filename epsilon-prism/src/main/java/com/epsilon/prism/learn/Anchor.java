package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.backend.HipBackend;
import com.epsilon.prism.backend.WeightPack;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.hip.RelationAttention;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.model.FrozenNetwork;
import com.epsilon.prism.model.NetworkInputs;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.Uploads;
import com.epsilon.prism.policy.InferencePath;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;

/**
 * 錨（凍結した学習の網 θ_t）の s を 1 つのデバイスで出す（当てはめの前に全ての行の分を出す {@link AnchorScores} が使う）。道は方策と同じ
 * {@link InferencePath} で選ぶ。HIP なら収集の方策と同じ自前の推論（渡したストリームに計画を 1 つ開く）、そうでなければ eager の凍結した網。
 * どちらも 1 回の呼び出しの中でデバイスの仕事を終え、s をホストに返す。スレッド安全ではない（同時に使うのは 1 つのスレッドだけ）。
 */
sealed interface Anchor extends AutoCloseable permits Anchor.Eager, Anchor.Hip {

  /**
   * 塊の錨の s。
   *
   * @param values s（行 r の k 番目の手は {@code values[starts[r] + k]}）
   * @param starts 行ごとの先頭
   */
  record Scores(float[] values, int[] starts) {

    /** 配列があることを確かめる。 */
    public Scores {
      Objects.requireNonNull(values, "values");
      Objects.requireNonNull(starts, "starts");
    }
  }

  /**
   * 塊の s を出す。
   *
   * @param inputs 封じた塊
   * @return s
   */
  Scores scores(HostInputs inputs);

  @Override
  void close();

  /**
   * 錨の網を読む。
   *
   * @param directory 錨の網のディレクトリ
   * @param path 推論の道（設定の inference。GPU・前線なし・特化した幅のときだけ HIP になる）
   * @param device デバイス
   * @param stream 計算ストリーム（HIP の道で使う。CPU は null）
   * @param attention 背骨の注意の自前の演算（eager の道で使う。null なら djl-rocm の演算）
   * @param capacity 微小バッチの容量
   * @return 錨
   * @throws IOException 錨の網を読めないとき
   */
  static Anchor open(
      Path directory,
      InferencePath path,
      Device device,
      PtStream stream,
      RelationAttention attention,
      BatchPlan.Capacity capacity)
      throws IOException {
    if (path.resolve(device, PrismNetwork.readShape(directory)) == InferencePath.HIP) {
      return Hip.open(directory, Objects.requireNonNull(stream, "stream"), capacity);
    }
    return new Eager(FrozenNetwork.load(directory, device), attention);
  }

  /** eager の凍結した網。s は [行, 手の容量] を行優先に平らにした形。索引の送り出しの置き場は網に付けて 1 つ持つ。 */
  final class Eager implements Anchor {

    private final FrozenNetwork network;
    private final RelationAttention attention;
    private final Uploads uploads;

    Eager(FrozenNetwork network, RelationAttention attention) {
      this.network = network;
      this.attention = attention;
      uploads = network.newUploads();
    }

    @Override
    public Scores scores(HostInputs inputs) {
      try (NDManager scope = network.newScope();
          InferenceMode inference = network.inference()) {
        NDArray s =
            network.scores(
                NetworkInputs.of(scope, uploads, inputs.input(), inputs.frontier()), attention);
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

  /** 自前の HIP の推論（収集の方策と同じ K0〜K7）。s は塊の手の順。 */
  final class Hip implements Anchor {

    /** 錨の重みの版（計画に写した版と照らすだけ）。 */
    private static final long VERSION = 0;

    private final HipBackend backend;

    private Hip(HipBackend backend) {
      this.backend = backend;
    }

    /** 網を CPU に読んで重みの束を作り、渡したストリームに計画を開いて写す。 */
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
