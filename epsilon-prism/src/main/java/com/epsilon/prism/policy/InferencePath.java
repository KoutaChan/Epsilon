package com.epsilon.prism.policy;

import ai.djl.Device;
import com.epsilon.prism.backend.WeightLayout;
import com.epsilon.prism.model.PrismShape;

/**
 * 網の推論の道（設定 {@code epsilon.prism.inference}）。どちらも同じ関数で、HIP は eager の BF16 の丸めの所に合わせる。
 */
public enum InferencePath {

  /** DJL の eager の網（GPU は BF16、CPU は FP32）。関係の符号 [行, S, S] を作るので、1 回の順伝播の詰めたトークンを上限で抑える。 */
  EAGER,

  /**
   * 自前の HIP の実行系。塊ごとにページ固定の blob の H2D 1 回 → K0〜K7 → s の D2H 1 回を、実行枠ごとの計算ストリームに積む。
   * libprism_hip.so（サーバで -Pnative-hip で組み立てる）が読めなければ、方策を開くときに例外で止まる（eager に落ちない）。
   */
  HIP;

  /**
   * このデバイスと網の形で使う道。HIP を選んでも、CPU、前線の腕、カーネルの特化のない幅では eager にする（HIP のカーネルが持たない形。
   * {@link WeightLayout#supports}）。
   *
   * @param device 方策のデバイス（GPU の列なら先頭。CPU と GPU は混ざらない）
   * @param shape 網の形
   * @return 使う道
   */
  public InferencePath resolve(Device device, PrismShape shape) {
    return this == HIP && device.isGpu() && WeightLayout.supports(shape) ? HIP : EAGER;
  }
}
