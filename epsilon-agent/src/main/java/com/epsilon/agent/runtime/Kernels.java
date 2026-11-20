package com.epsilon.agent.runtime;

/** 推論のバックエンド。 */
public enum Kernels {
  /** DJL の演算を 1 つずつ呼ぶ。どのデバイスでも動く。 */
  EAGER,
  /** モデルの系列・形・デバイスに応じて推論カーネルを選ぶ。 */
  NATIVE
}
