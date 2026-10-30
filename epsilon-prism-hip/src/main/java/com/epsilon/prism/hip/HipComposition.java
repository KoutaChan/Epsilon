package com.epsilon.prism.hip;

/** K7 が手の値を作る方法（native の順と同じ）。役割から作り方への写しは WeightPack の頭が持つ。 */
public enum HipComposition {
  /** s = B(代表の遷移)。 */
  DISCARD_VALUE,
  /** s = s[ダマ] + Δ_RIICHI（基準はダマの遷移）。 */
  RIICHI_GATE,
  /** s = Δ（基準は PASS の遷移、PASS ≡ 0）。 */
  RESPONSE_GATE,
  /** s = M + Δ（M は打牌と立直の s の最大、基準は ē）。 */
  TURN_GATE,
  /** s = 0。 */
  ZERO
}
