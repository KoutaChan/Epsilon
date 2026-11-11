package com.epsilon.agent.learn;

import static org.testng.Assert.assertEquals;

import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.testng.annotations.Test;

/** 段ごとの時間と回数を、複数のスレッドから足し合わせる。一度も数えなかった段は結果に出さない。 */
public class StageTimerTest {

  private enum Stage {
    ADVANCE,
    ENCODE,
    WRITE
  }

  @Test
  public void stagesSumTimesAndCountsFromManyThreads() throws Exception {
    StageTimer<Stage> timer = new StageTimer<>(Stage.class);
    try (ExecutorService pool = Executors.newFixedThreadPool(4)) {
      for (int task = 0; task < 100; task++) {
        pool.execute(() -> timer.add(Stage.ADVANCE, 1_000_000));
      }
      pool.shutdown();
      pool.awaitTermination(10, TimeUnit.SECONDS);
    }
    timer.add(Stage.WRITE, 3_000_000_000L, 6);
    Map<Stage, StageTimer.Entry> snapshot = timer.snapshot();
    assertEquals(List.copyOf(snapshot.keySet()), List.of(Stage.ADVANCE, Stage.WRITE));
    assertEquals(snapshot.get(Stage.ADVANCE), new StageTimer.Entry(0.1, 100));
    assertEquals(snapshot.get(Stage.WRITE), new StageTimer.Entry(3.0, 6));
  }
}
