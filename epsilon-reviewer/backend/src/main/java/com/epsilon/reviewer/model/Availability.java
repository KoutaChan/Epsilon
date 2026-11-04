package com.epsilon.reviewer.model;

/** モデルが使えるか。起動時に中身のハッシュを作らずに決め、開く時に変わったと分かれば MODEL_CHANGED にする。 */
public enum Availability {
  AVAILABLE,
  /** 系列の提供元が jar に無い。 */
  UNSUPPORTED_SERIES,
  /** チェックポイントが読めない・シンボリックリンク・空。 */
  CHECKPOINT_UNREADABLE,
  /** 開く時にチェックポイントが登録の時と違った（その後は使えない）。 */
  MODEL_CHANGED
}
