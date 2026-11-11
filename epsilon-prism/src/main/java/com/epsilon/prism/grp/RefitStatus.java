package com.epsilon.prism.grp;

/** refits.jsonl の 1 行の状態（{@link RefitRecord#status}）。 */
public enum RefitStatus {

  /** 学習を終えて候補を書いた。次のコホートで関門にかけるのを待っている。 */
  TRAINED,

  /** 関門に合格した。学習の JVM が読むのは、この状態の候補だけ。 */
  READY,

  /** 使わなかった。関門に落ちたとき、監視の判定で戻したとき、学び直しを止めたとき、学習の JVM が開けなかったときにこうなる。 */
  REJECTED,

  /** 学習か公開が例外で失敗した。 */
  FAILED
}
