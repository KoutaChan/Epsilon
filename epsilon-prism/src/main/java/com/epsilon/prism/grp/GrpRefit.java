package com.epsilon.prism.grp;

/** GRP の学習の時期の決め方（設定 {@code epsilon.prism.grpRefit}）。 */
public enum GrpRefit {

  /** grp-refit の要求だけ。 */
  MANUAL,

  /** grpRefitSchedule の間隔（{@link RefitScheduler}）。 */
  SCHEDULE,

  /** 予定表に加えて、ずれの警報（{@link DriftAlarm}）が ON なら前倒しする。 */
  SCHEDULE_AND_DRIFT;

  /** 予定表で学習するなら true。 */
  public boolean scheduled() {
    return this != MANUAL;
  }

  /** 警報で前倒しするなら true。 */
  public boolean drift() {
    return this == SCHEDULE_AND_DRIFT;
  }
}
