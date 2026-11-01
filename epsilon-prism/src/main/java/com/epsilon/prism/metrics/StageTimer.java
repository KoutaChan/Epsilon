package com.epsilon.prism.metrics;

import com.epsilon.prism.collect.EventLedger;
import java.util.EnumMap;
import java.util.Map;
import java.util.concurrent.atomic.LongAdder;

/** 段ごとの壁時計の累計と回数。どのスレッドからでも加算でき、製品に常設する（負荷は1回あたり System.nanoTime 2回）。 */
public final class StageTimer {
  public enum Stage {
    /** 主経路の進行（進行ワーカー）。 */
    MAIN_ADVANCE,
    /** 写しの進行。 */
    COPY_ADVANCE,
    /** 分岐の進行。 */
    FORK_ADVANCE,
    /** 候補集合と逸脱の計算（進行ワーカー）。 */
    CANDIDATES,
    /** inline の方策の評価。 */
    INLINE_POLICY,
    DRIVER_BUSY,
    DRIVER_WAIT,
    /**
     * 運の精算の帳簿の計算（{@link EventLedger}。主経路・写し・分岐の進行の内側の時間で、回数は呼び出しの数。帳簿を
     * 付けない収集では数えないので記録に出ない）。
     */
    LEDGER,
    /**
     * 記録 O の捕捉（prepare の進行の内側の時間。回数は O を読む方策に出す問い合わせと行に残す判断の数。どちらもない収集では数えない）。
     */
    V2_CAPTURE,
    /**
     * 行を作る判断の記録（record ごと。O の写しと神託）。ここから下の段は行を書く収集でだけ数え、進行の段は進行の内側の時間。
     */
    V2_ORACLE,
    /** 聴牌への到達の追跡（枠の席の打牌ごと）。 */
    V2_REACH,
    /** 局末の事実（枠のある経路の局末ごと）。 */
    V2_HAND_END,
    /** 局末の 3 語（ドライバ。回数は枠の数）。 */
    V2_OUTCOMES,
    /** 行を作って書き手に渡す（ドライバ。行ごと。圧縮は書き手のスレッドで、ここに入らない）。 */
    V2_ROWS
  }

  private final LongAdder[] nanos = new LongAdder[Stage.values().length];
  private final LongAdder[] counts = new LongAdder[Stage.values().length];

  public StageTimer() {
    for (int i = 0; i < nanos.length; i++) {
      nanos[i] = new LongAdder();
      counts[i] = new LongAdder();
    }
  }

  public void add(Stage stage, long elapsedNanos) {
    nanos[stage.ordinal()].add(elapsedNanos);
    counts[stage.ordinal()].increment();
  }

  /** count 回分の時間をまとめて足す（呼び出しごとに足すと共有の加算が重い段。{@link Stage#LEDGER}）。 */
  public void add(Stage stage, long elapsedNanos, long count) {
    nanos[stage.ordinal()].add(elapsedNanos);
    counts[stage.ordinal()].add(count);
  }

  public record Entry(double seconds, long count) {}

  public Map<Stage, Entry> snapshot() {
    var result = new EnumMap<Stage, Entry>(Stage.class);
    for (Stage stage : Stage.values()) {
      long count = counts[stage.ordinal()].sum();
      if (count > 0) result.put(stage, new Entry(nanos[stage.ordinal()].sum() / 1e9, count));
    }
    return result;
  }
}
