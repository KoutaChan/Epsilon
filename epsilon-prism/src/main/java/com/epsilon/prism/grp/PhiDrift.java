package com.epsilon.prism.grp;

/** ずれの警報（{@link DriftAlarm}）の使い方（設定 {@code epsilon.prism.phiDrift}）。 */
public enum PhiDrift {

  /** 警報を計算しない。 */
  OFF,

  /** 警報の ON/OFF を記録するだけ。 */
  REPORT,

  /** 記録し、ON なら学習を前倒しする（grpRefit = SCHEDULE_AND_DRIFT と組む）。 */
  REFIT
}
