package com.epsilon.prism.policy;

import ai.djl.Device;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.prism.backend.WeightLayout;
import com.epsilon.prism.model.PrismShape;

/** ネットワークの推論のバックエンド。どちらも同じ関数を計算し、HIP は BF16 に丸める箇所を eager に合わせる。 */
public enum InferencePath {

  /** DJL の eager のネットワーク。関係のコード [行, S, S] を作るので、1 回の順伝播で扱うトークン数（パディング込み）を上限で抑える。 */
  EAGER,

  /**
   * 自前の HIP のランタイム（BF16）。バッチごとに、ページ固定の blob の H2D 1 回 → 形の記述に並んだ順のカーネル → s の D2H 1 回を、
   * スロットごとの計算ストリームに積む。libprism_hip.so が読めなければ、方策を開くときに
   * 例外で止まる（eager に切り替えない）。
   */
  HIP;

  /** 設定から推論のバックエンドを選ぶ。{@link Kernels#NATIVE} は、PRISM 専用のカーネルを使う HIP になる。 */
  public static InferencePath of(Kernels kernels) {
    return switch (kernels) {
      case EAGER -> EAGER;
      case NATIVE -> HIP;
    };
  }

  /**
   * このデバイスとネットワークの形で使うバックエンドを返す。HIP を選んでも、CPU と、専用のカーネルが無い幅では eager にする（HIP の
   * カーネルが扱えない形。{@link WeightLayout#supports}）。
   *
   * @param device 方策のデバイス（GPU の列なら先頭。CPU と GPU は混ざらない）
   * @param shape ネットワークの形
   * @return 使うバックエンド
   */
  public InferencePath resolve(Device device, PrismShape shape) {
    return this == HIP && device.isGpu() && WeightLayout.supports(shape) ? HIP : EAGER;
  }
}
