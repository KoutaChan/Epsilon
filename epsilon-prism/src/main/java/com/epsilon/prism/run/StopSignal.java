package com.epsilon.prism.run;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sun.misc.Signal;

/**
 * 系譜を止める合図。立つと系譜（{@link Lineage}）は次の安全点（{@link SafePoint}）で止まり、train は 0 で終わる。実行中の段は最後まで終える。
 * train の JVM では SIGTERM で立つ（{@link #onTerm}。2 回目からの TERM も同じで、何もしない）。
 */
public final class StopSignal {
  private static final Logger LOG = LoggerFactory.getLogger(StopSignal.class);

  /** 系譜が合図を見る場所。どこで止まっても、再開は同じ結果に戻る。 */
  public enum SafePoint {
    /** 反復 t のコホートを封印した後（運の精算の前）。番号は反復。再開では封印したコホートを使い回す。 */
    SEALED,

    /** 反復 t を終えて state.json を進めた後（x1n3 の前）。番号は反復。 */
    ITERATED,

    /** バージョン V の x1n3 を測った後。番号はバージョン。 */
    MEASURED
  }

  private volatile boolean requested;

  /** 試験で合図を立てる場所（null なら立てない）。 */
  private final SafePoint trigger;

  private final int triggerNumber;

  private StopSignal(SafePoint trigger, int triggerNumber) {
    this.trigger = trigger;
    this.triggerNumber = triggerNumber;
  }

  /** 頼まれるまで立たない合図。 */
  public static StopSignal none() {
    return new StopSignal(null, 0);
  }

  /** SIGTERM で立つ合図（JVM の既定の TERM の動き＝すぐ終わる、を置き換える）。 */
  public static StopSignal onTerm() {
    StopSignal stop = none();
    Signal.handle(new Signal("TERM"), signal -> stop.request("SIG" + signal.getName()));
    return stop;
  }

  /** 安全点 point の番号 number に着いたところで立つ合図（試験）。 */
  static StopSignal at(SafePoint point, int number) {
    return new StopSignal(point, number);
  }

  /** 合図を立てる。 */
  public void request(String why) {
    if (!requested) LOG.info("{}: stopping at the next safe point", why);
    requested = true;
  }

  /** 合図が立っていれば true。 */
  public boolean requested() {
    return requested;
  }

  /** 安全点 point（番号 number）で止まるなら true。 */
  boolean stopsAt(SafePoint point, int number) {
    if (point == trigger && number == triggerNumber) requested = true;
    if (requested) LOG.info("stopped at {} {}", point, number);
    return requested;
  }
}
