package com.epsilon.prism.grp;

/** 主経路の対局の記録と裏の採点（設定 {@code epsilon.prism.phiMonitor}）。どれでもラベル行は変わらない。 */
public enum PhiMonitor {

  /** 記録しない。 */
  OFF,

  /** 主経路の対局を grp/games/c&lt;t&gt;.bin.gz に記録する（{@link GameLogObserver}）。 */
  RECORD,

  /** 記録し、裏の作業（{@link GrpWorkerMain}）が使用中の Φ と影の Φ で採点して reports/phimon.jsonl に書く。 */
  SCORE;

  /** 主経路の対局を記録するなら true。 */
  public boolean records() {
    return this != OFF;
  }

  /** 裏の作業が採点するなら true。 */
  public boolean scores() {
    return this == SCORE;
  }
}
