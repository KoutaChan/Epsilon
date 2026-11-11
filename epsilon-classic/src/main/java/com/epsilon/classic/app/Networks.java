package com.epsilon.classic.app;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.pytorch.jni.JniUtils;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.model.ClassicNetwork;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;

/** コマンドが共有する、ネットワークの作り方とデバイスの決め方。 */
final class Networks {

  private Networks() {}

  /** アーキテクチャごとの既定の効用を返す。Nano は TENHOU、Major は AVERAGE_RANK で、どちらも元の Nano・Major の学習と同じ。 */
  static RankUtility utility(Architecture architecture) {
    return architecture == Architecture.NANO ? RankUtility.TENHOU : RankUtility.AVERAGE_RANK;
  }

  /**
   * 初期化したままのネットワークを作る。初期値は乱数の種で決まる。
   *
   * @param manager パラメーターを持つマネージャー
   */
  static ClassicNetwork create(Architecture architecture, Variant variant, NDManager manager, long seed) {
    ClassicNetwork network = new ClassicNetwork(architecture, variant, utility(architecture));
    synchronized (Networks.class) {
      manager.getEngine().setRandomSeed(Long.hashCode(seed));
      network.initialize(manager);
    }
    return network;
  }

  /**
   * デバイスを決め、torch を読み込む。CPU の演算のスレッド数も決める。
   *
   * @param name {@code cpu} か {@code gpuN}
   * @param threads CPU の演算のスレッド数（0 なら変えない）
   */
  static Device device(String name, int threads) {
    NetworkRuntime.initialize();
    Engine.getEngine(PtEngine.ENGINE_NAME);
    Device device = NetworkDevices.device(name);
    if (threads > 0) {
      JniUtils.setNumThreads(threads);
    }
    return device;
  }
}
