package com.epsilon.agent.facts;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.agent.observe.RandomPlay;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * ウォームアップの後はメモリを確保しないことを確かめる。測る処理は、判断の時点ごとに判断する全ての席で、観測から牌の枚数・公開情報・
 * 手の形を作り、待ちを採点すること。
 */
public class FactsAllocationTest {

  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final int WARM_UP = 20_000;
  private static final int MEASURED = 100_000;

  /**
   * 確保を数える仕組みそのものの揺れとして許す幅（バイト）。測るのは 10^5 ステップなので、1 ステップに 1 回でも確保（16 バイト以上）
   * があれば、合計は MB 単位になる。
   */
  private static final long SLACK_BYTES = 4_096;

  private final RandomPlay play = new RandomPlay(VARIANT, 61);
  private final ObservationWriter writer = new ObservationWriter(VARIANT);
  private final ObservationBatch batch = new ObservationBatch(VARIANT, 1_024);
  private final Observation o = new Observation(VARIANT);
  private final TileCounts tiles = new TileCounts(VARIANT);
  private final SeatFacts seats = new SeatFacts(VARIANT);
  private final ShapeFacts shape = new ShapeFacts(VARIANT);
  private final WaitScores scores = new WaitScores(VARIANT);
  private final HandShape hand = new HandShape();

  @Test
  public void factsAllocateNothingOnceWarm() {
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
   * 自己対局を決まったステップ数だけ進める。判断の時点ごとに、判断する全ての席で牌の枚数・公開情報・手の形を作り、聴牌なら待ちを
   * 採点する。最適化で処理が消されないように、読んだ値の和を返す。
   */
  private long run(int steps) {
    long checksum = 0;
    for (int step = 0; step < steps; step++) {
      if (play.boundary() == Boundary.DECISIONS) {
        Decisions decisions = play.engine.decisions();
        batch.clear();
        for (int slot = 0; slot < decisions.size(); slot++) {
          o.bind(batch, writer.capture(play.engine.view(decisions.seat(slot)), decisions, slot, batch));
          checksum += facts();
        }
      }
      play.step();
    }
    return checksum;
  }

  private long facts() {
    tiles.load(o);
    seats.load(o);
    ObservedHand.shape(tiles, seats, hand);
    shape.analyze(hand, tiles);
    long checksum = tiles.totalUnseen() + seats.rank(0) + shape.acceptance() + seats.danger(1, 0, tiles);
    if (shape.waits() == 0) {
      return checksum;
    }
    ObservedHand.context(o, tiles, seats, scores.context());
    scores.load(shape.waits(), tiles.unseenRed());
    return checksum + scores.score(0, WaitScores.Method.RON, WaitScores.WinningTile.PLAIN).base();
  }
}
