package com.epsilon.mahjong.util;

import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.util.function.IntToLongFunction;
import org.testng.Assert;
import org.testng.SkipException;

/** ウォームアップの後の処理がメモリを確保しないことを、このスレッドが確保したバイト数で確かめる。 */
public final class AllocationProbe {

  /** 確保量のカウンタ自体の揺れとして許すバイト数。測る回数が 10^5 回以上なら、1 回に 1 つ（16 バイト以上）オブジェクトを確保するだけで合計は MB 単位になり、この値を大きく超える。 */
  private static final long SLACK_BYTES = 4_096;

  private AllocationProbe() {}

  /**
   * 処理をウォームアップの回数だけ走らせてから、測る回数だけ走らせる。測る間にこのスレッドが確保したバイト数が、許すバイト数
   * 未満であることを確かめる。スレッドごとの確保量を数えられない JVM では SkipException を投げる。
   *
   * @param warmUp ウォームアップの回数
   * @param measured 測る回数
   * @param rounds 受け取った回数だけ処理を走らせる関数。最適化で処理が消されないよう、チェックサムを返す
   */
  public static void assertNoAllocation(int warmUp, int measured, IntToLongFunction rounds) {
    ThreadMXBean threads = ManagementFactory.getThreadMXBean() instanceof ThreadMXBean bean ? bean : null;
    if (threads == null || !threads.isThreadAllocatedMemorySupported() || !threads.isThreadAllocatedMemoryEnabled()) {
      throw new SkipException("Per-thread allocation counters are not available");
    }
    long checksum = rounds.applyAsLong(warmUp);
    long before = threads.getCurrentThreadAllocatedBytes();
    checksum += rounds.applyAsLong(measured);
    long allocated = threads.getCurrentThreadAllocatedBytes() - before;
    Assert.assertTrue(
        allocated < SLACK_BYTES, allocated + " bytes over " + measured + " rounds (checksum " + checksum + ")");
  }
}
