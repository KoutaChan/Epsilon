package com.epsilon.classic;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.classic.model.ClassicNetwork;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Random;

/**
 * テスト用の、乱数の重みを持つ小さなチェックポイント。重みは N(0, 1/入力の幅)、LayerNorm のゲインは 1 ＋ N(0, 0.1²)、ほかの 1 次元の
 * パラメーターは N(0, 0.1²) にするので、0 で初期化する層も 0 のままにならない。中身はシードだけで決まる。classic のテスト用の jar に
 * 入り、ほかのモジュールのテストも Nano・Major のチェックポイントをこれで作る。
 */
public final class RandomCheckpoints {

  private RandomCheckpoints() {}

  /**
   * 乱数の重みを持つネットワークを CPU で作り、新しいディレクトリに書く。
   *
   * @param directory 書くディレクトリ（まだ無いもの）
   * @param seed 重みのシード
   */
  public static void write(Path directory, Architecture architecture, Variant variant, long seed) throws IOException {
    RankUtility utility = architecture == Architecture.NANO ? RankUtility.TENHOU : RankUtility.AVERAGE_RANK;
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), "PyTorch")) {
      ClassicNetwork network = new ClassicNetwork(architecture, variant, utility);
      network.initialize(manager);
      Random random = new Random(seed);
      for (Pair<String, Parameter> pair : network.getParameters()) {
        NDArray array = pair.getValue().getArray();
        Shape shape = array.getShape();
        float[] values = new float[Math.toIntExact(shape.size())];
        boolean gain = pair.getKey().endsWith("_gamma");
        double scale = shape.dimension() == 1 ? 0.1 : 1 / Math.sqrt(shape.get(shape.dimension() - 1));
        for (int index = 0; index < values.length; index++) {
          values[index] = (float) ((gain ? 1 : 0) + scale * random.nextGaussian());
        }
        array.set(values);
      }
      ClassicCheckpoint.write(directory, network, 0);
    }
  }
}
