package com.epsilon.prism.grp;

/** 主経路の対局を記録するか、記録したうえで裏で採点するかの設定。どれを選んでも、ラベルの行は変わらない。 */
public enum PhiMonitor {

  /** 記録しない。 */
  OFF,

  /** 主経路の半荘を grp/records/c&lt;t&gt;.bin に、全ての局の始まりの卓と結果を grp/counts/c&lt;t&gt;.bin に記録する（{@link GameLogObserver}）。 */
  RECORD,

  /** 記録したうえで、使っている Φ と影の Φ で採点し、結果を reports/phimon.jsonl に書く（{@link GrpBackground}）。 */
  SCORE;

  /** 主経路の対局を記録するなら true。 */
  public boolean records() {
    return this != OFF;
  }

  /** 採点するなら true。 */
  public boolean scores() {
    return this == SCORE;
  }
}
