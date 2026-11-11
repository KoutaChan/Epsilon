package com.epsilon.agent.app;

/** 終了の種類。それぞれの数が終了コード。 */
public enum Exit {
  /** 頼まれた処理を終えた。する事が無かったときもこれ。 */
  OK(0),
  /** 例外で止まった。 */
  FAILED(1),
  /** 知らないコマンドか、引数か設定の誤り。 */
  USAGE(2),
  /** 停止の要求を受けて、セーフポイントで止まった。もう一度起動すれば続きから動く。 */
  STOPPED(3),
  /** 確かめるコマンドが問題を見つけた。 */
  PROBLEMS(4),
  /** 起動し直す必要がある（速さの設定の変更など）。 */
  RESTART(75),
  /** GPU を譲るために、セーフポイントで止まった。 */
  YIELD(76),
  /** 2 回目の割り込みか、停止の要求を受けないコマンドへの割り込みで、セーフポイントを待たずに終わった。 */
  INTERRUPTED(130);

  private final int code;

  Exit(int code) {
    this.code = code;
  }

  public int code() {
    return code;
  }
}
