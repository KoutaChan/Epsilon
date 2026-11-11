package com.epsilon.agent.observe;

import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.rules.Variant;
import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * ウォームアップの後は、判断の時点ごとに判断する全ての席の観測を書いて読んでも、メモリを確保しないことを確かめる。自己対局を進める
 * エンジンも確保しない。
 */
public class ObservationAllocationTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final int WARM_UP = 20_000;
  private static final int MEASURED = 100_000;

  /**
   * 確保の数え方そのものの揺れとして許すバイト数。測るのは 10^5 ステップなので、1 ステップに 1 回でも確保（16 バイト以上）があれば、
   * 合計は MB 単位になる。
   */
  private static final long SLACK_BYTES = 4_096;

  private final RandomPlay play = new RandomPlay(VARIANT, 23);
  private final ObservationWriter writer = new ObservationWriter(VARIANT);
  private final ObservationBatch batch = new ObservationBatch(VARIANT, 1_024);
  private final Observation read = new Observation(VARIANT);

  @Test
  public void capturesAllocateNothingOnceWarm() {
    ThreadMXBean threads = ManagementFactory.getThreadMXBean() instanceof ThreadMXBean bean ? bean : null;
    if (threads == null || !threads.isThreadAllocatedMemorySupported() || !threads.isThreadAllocatedMemoryEnabled()) {
      throw new SkipException("Per-thread allocation counters are not available");
    }
    long checksum = run(WARM_UP);
    long before = threads.getCurrentThreadAllocatedBytes();
    checksum += run(MEASURED);
    long allocated = threads.getCurrentThreadAllocatedBytes() - before;
    Assert.assertTrue(
        allocated < SLACK_BYTES, allocated + " bytes over " + MEASURED + " steps (checksum " + checksum + ")");
  }

  /**
   * 自己対局を決まったステップ数だけ進め、判断の時点ごとに、判断する全ての席の観測を書いて読む。最適化で処理が消されないように、
   * 読んだ値の和を返す。
   */
  private long run(int steps) {
    long checksum = 0;
    for (int step = 0; step < steps; step++) {
      if (play.boundary() == Boundary.DECISIONS) {
        Decisions decisions = play.engine.decisions();
        batch.clear();
        for (int slot = 0; slot < decisions.size(); slot++) {
          int index = writer.capture(play.engine.view(decisions.seat(slot)), decisions, slot, batch);
          read.bind(batch, index);
          checksum += read.transitions() + read.events() + read.wins();
        }
      }
      play.step();
    }
    return checksum;
  }
}
