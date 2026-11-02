package com.epsilon.network;

/** ネイティブエンジンの初期化前に、全モデル共通の実行既定値を設定する。 */
public final class NetworkRuntime {
  static {
    // 小さい GRP の CPU 演算でスレッド過多を避け、明示された DJL/OpenMP/MKL 設定は優先する。
    String ompThreads = System.getenv("OMP_NUM_THREADS");
    String mklThreads = System.getenv("MKL_NUM_THREADS");
    if (System.getProperty("ai.djl.pytorch.num_threads") == null
        && (ompThreads == null || ompThreads.isBlank())
        && (mklThreads == null || mklThreads.isBlank())) {
      System.setProperty("ai.djl.pytorch.num_threads", "1");
    }
  }

  private NetworkRuntime() {}

  /** エンジン取得・モデル生成より先に呼ぶ。クラス初期化により一度だけ適用する。 */
  public static void initialize() {}
}
