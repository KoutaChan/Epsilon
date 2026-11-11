package com.epsilon.agent.play;

/** 卓のコアが時間を測る段。その場で答える打ち手の評価と O の取り込みは、半荘を進める処理の中で測った時間。 */
public enum TableStage {
  /** その場で答える打ち手の評価。 */
  INLINE_POLICY,
  /** 記録 O の取り込み。回数は取り込んだ O の数。 */
  CAPTURE,
  /** ドライバがイベントを受けて処理していた時間。 */
  DRIVER_BUSY,
  /** ドライバがイベントを待っていた時間。 */
  DRIVER_WAIT
}
