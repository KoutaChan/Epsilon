package com.epsilon.prism.run;

/**
 * 系譜の中でバージョンの網 1 席対 相手 3 席の卓（x1n3。{@link NanoTable}）を測るか（設定 {@code epsilon.prism.x1n3}）。測るバージョン・局数・相手・出力先は
 * x1n3Every・x1n3From・x1n3Games・x1n3Opponent・x1n3Dir（{@link X1n3Outputs}）。
 */
public enum X1n3 {

  /** 測らない。 */
  OFF,

  /** バージョンが x1n3Every の倍数（x1n3From 以上）のたびに測る。 */
  ON
}
