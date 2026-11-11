package com.epsilon.prism.hip;

/**
 * K7 が手の値を作る方法。並びは native と同じ。価値グラフでの手の扱いごとにどの方法を使うかは、WeightPack のヘッダーにある表で決まる。
 */
public enum HipComposition {
  /** 代表の遷移の打牌の値をそのまま手の値にする。 */
  DISCARD_VALUE,
  /** 同じ牌をダマで切る手の値に、ダマの遷移を基準にした立直のゲートの差を足す。 */
  RIICHI_GATE,
  /** 見送りの遷移を基準にしたゲートの差を手の値にする（見送りの手の値は 0）。 */
  RESPONSE_GATE,
  /** 打牌と立直の手の値の最大に、基準の手の表現を基準にしたゲートの差を足す。 */
  TURN_GATE,
  /** 手の値を 0 にする。 */
  ZERO
}
