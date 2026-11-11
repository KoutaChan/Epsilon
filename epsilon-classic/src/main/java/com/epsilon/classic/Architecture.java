package com.epsilon.classic;

import com.epsilon.agent.policy.Series;

/** ネットワークのアーキテクチャ。系列と隠れ層の幅を決める。入力のフィールドの幅と語彙は、フィールドの定数の側で決める。 */
public enum Architecture {
  /** 隠れ層の幅 256。 */
  NANO(Series.NANO, 256, false),
  /** 隠れ層の幅 384。和了したときの点数の計算を持ち、公開された履歴を入力に使う。 */
  MAJOR(Series.MAJOR, 384, true);

  private final Series series;
  private final int hidden;
  private final boolean publicHistory;

  Architecture(Series series, int hidden, boolean publicHistory) {
    this.series = series;
    this.hidden = hidden;
    this.publicHistory = publicHistory;
  }

  public Series series() {
    return series;
  }

  /** 隠れ層の幅。 */
  public int hidden() {
    return hidden;
  }

  /**
   * 状態の入力に公開された履歴を持つなら true。公開された履歴とは、公開されたイベントと、各席が立直・副露・槓をした時点との関係のこと。
   */
  public boolean publicHistory() {
    return publicHistory;
  }
}
