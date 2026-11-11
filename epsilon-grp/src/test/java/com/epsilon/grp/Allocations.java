package com.epsilon.grp;

import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import org.testng.Assert;
import org.testng.SkipException;

/** 推論で繰り返し通る処理がメモリを確保しないことを確かめる。今のスレッドが確保した量を ThreadMXBean で数える。 */
final class Allocations {

  /** ウォームアップの後の 1 回で許す確保の量（バイト）。数える仕組みの誤差を見込んだ分。 */
  private static final long SLACK = 1_024;

  private Allocations() {}

  /**
   * ウォームアップの後、1 回の実行で確保する量が許す量より少ないことを確かめる。確保の量を数えられない JVM ではテストを飛ばす。
   *
   * @param what 失敗したときのメッセージに書く、処理の名前
   * @param warmups ウォームアップの回数
   * @param task 測る処理
   */
  static void assertSteady(String what, int warmups, Runnable task) {
    ThreadMXBean threads = (ThreadMXBean) ManagementFactory.getThreadMXBean();
    if (!threads.isThreadAllocatedMemorySupported() || !threads.isThreadAllocatedMemoryEnabled()) {
      throw new SkipException("Thread allocation counting is not available");
    }
    for (int round = 0; round < warmups; round++) {
      task.run();
    }
    long before = threads.getCurrentThreadAllocatedBytes();
    task.run();
    long allocated = threads.getCurrentThreadAllocatedBytes() - before;
    Assert.assertTrue(allocated < SLACK, allocated + " bytes for " + what);
  }
}
