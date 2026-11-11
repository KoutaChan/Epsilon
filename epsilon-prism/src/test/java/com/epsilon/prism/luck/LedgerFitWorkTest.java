package com.epsilon.prism.luck;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.luck.LedgerFitTest.SyntheticRow;
import com.sun.management.ThreadMXBean;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.SplittableRandom;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 当てはめの和の作業領域のテスト。半荘ごとに束ねる和は、同じ半荘の行が続く間は表を引かずに前の束を使う。それでも、半荘が行き来しても、
 * 半荘 0 から始まっても、行ごとに表を引いたときと同じ束に同じ順で足し、表のキーの順も同じになる。JIT のウォームアップの後の和は、
 * 表にもうある半荘の行ではメモリを確保せず、新しい半荘でも束 1 つ分だけ確保する。シャード 1 以降のテープは、ブロックと行の表の容量の中では
 * 行ごとに確保しない。
 */
public class LedgerFitWorkTest {
  private static final Logger log = LoggerFactory.getLogger(LedgerFitWorkTest.class);

  private static final int F = LedgerColumn.COUNT;

  /** 半荘を行き来する行の並び。半荘 0 から始まり、7 割は前の行と同じ半荘、残りは 0〜299 のどれか（前に出た半荘に戻ることもある）。 */
  private static List<SyntheticRow> wandering(int count, long seed) {
    var random = new SplittableRandom(seed);
    List<SyntheticRow> rows = new ArrayList<>(count);
    long game = 0;
    for (int i = 0; i < count; i++) {
      if (i > 0 && random.nextInt(10) >= 7) game = random.nextInt(300);
      DecisionClass c = i % 3 == 0 ? DecisionClass.CALL : DecisionClass.DISCARD_MID;
      rows.add(
          LedgerFitTest.row(
              game, c, LedgerFitTest.ones(), LedgerFitTest.planted(), 0.3, random));
    }
    return rows;
  }

  @Test
  public void gameClustersMatchALookupPerRow() {
    List<SyntheticRow> rows = wandering(20_000, 41);
    var accumulator = new LedgerFit.Accumulator();
    Map<Long, double[]> expected = new HashMap<>();
    for (SyntheticRow row : rows) {
      accumulator.add(row);
      double[] game = expected.computeIfAbsent(row.gameId(), ignored -> new double[1 + F]);
      for (int a = 0; a < row.actions(); a++) {
        if (Float.isNaN(row.values()[a])) continue;
        game[0]++;
        for (int f = 0; f < F; f++) game[1 + f] += row.ledger(a, f);
      }
    }
    Assert.assertEquals(
        new ArrayList<>(accumulator.byGame.keySet()), new ArrayList<>(expected.keySet()), "cluster order");
    for (Map.Entry<Long, double[]> entry : expected.entrySet()) {
      double[] got = accumulator.byGame.get(entry.getKey());
      for (int k = 0; k <= F; k++) {
        Assert.assertEquals(
            Double.doubleToRawLongBits(got[k]),
            Double.doubleToRawLongBits(entry.getValue()[k]),
            "game " + entry.getKey() + " entry " + k);
      }
    }
    Assert.assertTrue(expected.containsKey(0L) && expected.size() > 250, "games " + expected.size());
  }

  /**
   * JIT のウォームアップの後は、表にある半荘の行を足しても確保はほぼ 0（半荘が替わるたびに表を引く分、キーのボクシングなどで 32 バイトまで）。
   * 新しい半荘の行を足すと、半荘ごとに束 1 つと、表の項目と、表が伸びる分（平均 400 バイト未満）だけ確保する。作業領域を使い回さなければ、
   * 行ごとに、測った手の番号の配列 2 つと、列の数の長さの double の配列 2 つ（合わせて 400 バイト近く）を確保することになる。
   */
  @Test
  public void theSumsAllocateOnlyForNewGames() {
    ThreadMXBean threads = allocationCounter();
    int games = 1_000, perGame = 16;
    var random = new SplittableRandom(7);
    List<SyntheticRow> rows = new ArrayList<>(games * perGame);
    for (int g = 0; g < games; g++)
      for (int r = 0; r < perGame; r++)
        rows.add(
            LedgerFitTest.row(
                5_000_000L + g,
                r % 3 == 0 ? DecisionClass.CALL : DecisionClass.DISCARD_MID,
                LedgerFitTest.ones(),
                LedgerFitTest.planted(),
                0.3,
                random));
    long fresh = Long.MAX_VALUE, again = Long.MAX_VALUE;
    for (int round = 0; round < 30; round++) {
      var accumulator = new LedgerFit.Accumulator();
      long before = threads.getCurrentThreadAllocatedBytes();
      for (SyntheticRow row : rows) accumulator.add(row);
      long first = threads.getCurrentThreadAllocatedBytes() - before;
      before = threads.getCurrentThreadAllocatedBytes();
      for (SyntheticRow row : rows) accumulator.add(row);
      long second = threads.getCurrentThreadAllocatedBytes() - before;
      if (round >= 27) {
        fresh = Math.min(fresh, first);
        again = Math.min(again, second);
      }
    }
    log.info(
        String.format(
            Locale.ROOT,
            "LEDGER fit sums: %d rows of %d games, new games %.1f B per game, known games %.2f B per row",
            rows.size(),
            games,
            (double) fresh / games,
            (double) again / rows.size()));
    Assert.assertTrue(again <= 32L * games + 4_096, "known games allocated " + again + " B");
    Assert.assertTrue(fresh <= 400L * games, "new games allocated " + fresh + " B");
  }

  /**
   * シャード 1 以降のテープは、最初の行でブロックを作った後、ブロックと行の表の容量（1024 行）の中では行ごとに確保しない（測った手の番号の
   * 配列も使い回す）。
   */
  @Test
  public void theTapeAppendsWithoutAllocating() {
    ThreadMXBean threads = allocationCounter();
    List<SyntheticRow> rows = new ArrayList<>();
    for (SyntheticRow row : LedgerFitTest.rows(1_001, 0, LedgerFitTest.ones(), LedgerFitTest.planted(), 0.3, 3)) {
      // コホートの行の帳簿は fp16 で正確に表せる値だけなので、テープもその値しか受け付けない
      float[] ledger = row.ledger().clone();
      for (int k = 0; k < ledger.length; k++) ledger[k] = Float.float16ToFloat(Float.floatToFloat16(ledger[k]));
      rows.add(LedgerFitTest.withValues(row, row.values(), ledger));
    }
    long best = Long.MAX_VALUE;
    for (int round = 0; round < 300; round++) {
      var tap = LedgerFitTest.CLASSIC.tap(2, index -> true);
      tap.add(1, rows.get(0));
      long before = threads.getCurrentThreadAllocatedBytes();
      for (int i = 1; i < rows.size(); i++) tap.add(1, rows.get(i));
      long allocated = threads.getCurrentThreadAllocatedBytes() - before;
      if (round >= 297) best = Math.min(best, allocated);
    }
    log.info(String.format(Locale.ROOT, "LEDGER tape: %d rows appended with %d B", rows.size() - 1, best));
    Assert.assertTrue(best <= 4_096, "tape allocated " + best + " B");
  }

  private static ThreadMXBean allocationCounter() {
    ThreadMXBean threads = (ThreadMXBean) ManagementFactory.getThreadMXBean();
    if (!threads.isThreadAllocatedMemorySupported() || !threads.isThreadAllocatedMemoryEnabled()) {
      throw new SkipException("Thread allocation counting is not available");
    }
    return threads;
  }
}
