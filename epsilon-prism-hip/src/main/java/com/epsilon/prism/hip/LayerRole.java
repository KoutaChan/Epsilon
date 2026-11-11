package com.epsilon.prism.hip;

/** バックボーンの層の種類。並びは native の LayerRole と同じ。 */
public enum LayerRole {
  /** 全てのトークンがクエリになる層。イベントの残差も書き換える。 */
  ALL_TOKENS,
  /** 局・席・牌のトークンだけがクエリになる層。イベントは最後の全トークンの層の出力のままキーとバリューとして読み、書き換えない。 */
  FIXED_TOKENS
}
