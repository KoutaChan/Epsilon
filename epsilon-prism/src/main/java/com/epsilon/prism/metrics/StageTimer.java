package com.epsilon.prism.metrics;

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
    /** 測定器の手の問い合わせ（probe-nano だけ。通常の収集では数えないので記録に出ない）。 */
    PROBE,
    DRIVER_BUSY,
    DRIVER_WAIT
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
