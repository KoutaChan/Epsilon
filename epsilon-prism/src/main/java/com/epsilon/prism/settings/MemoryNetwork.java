package com.epsilon.prism.settings;

/** バージョンが持つ 2 つのネットワークのどちらを使うか。方策の記憶を使わないときは、2 つは同じネットワークになる。 */
public enum MemoryNetwork {
  /** 学習するネットワーク θ。 */
  ITERATE,
  /** 打つためのネットワーク A（方策の記憶で平均したもの）。 */
  AVERAGE
}
