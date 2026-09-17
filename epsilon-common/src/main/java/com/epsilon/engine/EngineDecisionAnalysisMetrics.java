package com.epsilon.engine;

import java.util.concurrent.atomic.AtomicLong;

/** 判断時の手牌解析とキャッシュ利用の計測値。計測を無効にした場合はカウンターを更新しない。 */
public final class EngineDecisionAnalysisMetrics {

  private static final boolean ENABLED = Boolean.getBoolean("epsilon.engine.metrics");
  private static final AtomicLong shapeHits = new AtomicLong();
  private static final AtomicLong shapeMisses = new AtomicLong();

  private EngineDecisionAnalysisMetrics() {}

  static void recordShape(boolean hit) {
    if (!ENABLED) {
      return;
    }
    (hit ? shapeHits : shapeMisses).incrementAndGet();
  }

  public static Snapshot snapshot() {
    return new Snapshot(shapeHits.get(), shapeMisses.get());
  }

  public record Snapshot(long shapeHits, long shapeMisses) {
    public static Snapshot empty() {
      return new Snapshot(0L, 0L);
    }
  }
}
