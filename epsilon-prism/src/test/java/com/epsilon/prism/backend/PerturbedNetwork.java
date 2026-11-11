package com.epsilon.prism.backend;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import java.util.Random;

/** HIP のテストに使うネットワーク。初期化した後、全てのパラメーターを乱数で揺らす（揺らし方は四人打ちと三人打ちで同じ）。 */
public final class PerturbedNetwork {

  private PerturbedNetwork() {}

  /**
   * ネットワークを作り、全てのパラメーターを揺らす。0 で初期化する層も効くようにし、関係のバイアス、コードの表、2 段目の出力は大きく揺らして、
   * 式の取り違えが結果に出るようにする。
   *
   * @param manager パラメーターのマネージャー
   * @param format 入力の形式
   * @param shape 形
   * @param seed 乱数の種
   * @return ネットワーク
   */
  public static PrismNetwork of(NDManager manager, InputFormat format, PrismShape shape, long seed) {
    manager.getEngine().setRandomSeed((int) seed);
    PrismNetwork network = new PrismNetwork(format, shape);
    network.initialize(manager);
    Random random = new Random(seed);
    for (Pair<String, Parameter> pair : network.getParameters()) {
      String name = pair.getKey();
      NDArray array = pair.getValue().getArray();
      float[] values = array.toFloatArray();
      double sigma = sigma(name.substring(name.indexOf('_') + 1), name);
      for (int i = 0; i < values.length; i++) {
        values[i] += (float) (random.nextGaussian() * sigma);
      }
      array.set(values);
    }
    return network;
  }

  /** パラメーターごとの揺らす大きさ（正規乱数の標準偏差）。 */
  private static double sigma(String local, String name) {
    if (local.equals("relation")) {
      return 1.0;
    }
    if (local.equals("codeKey") || local.equals("codeValue")) {
      return 0.3;
    }
    if (name.contains("reread") && local.equals("output")) {
      return 0.1;
    }
    boolean head = name.contains("discardValue") || name.contains("gates");
    if (head
        && (local.equals("output")
            || local.equals("exact")
            || local.equals("kindOffset")
            || local.equals("exactProjection")
            || local.equals("outputBias")
            || local.equals("exactOutput"))) {
      return 0.1;
    }
    return 0.02;
  }
}
