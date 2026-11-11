package com.epsilon.agent.runtime;

/** 推論のバックエンド。 */
public enum Kernels {
  /** DJL の演算を 1 つずつ呼ぶ。どのデバイスでも動く。 */
  EAGER,
  /** モデルの系列ごとの専用のカーネル（PRISM なら HIP）。カーネルを持たない系列や、カーネルが扱えない形のネットワークは、開くときに拒む。 */
  NATIVE
}
