package com.epsilon.prism.input;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.input.InputSchema.Axis;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * 塊の形の度数: 区切りの度数・Σ⌈T/16⌉²・T > 128 の行・行あたりのトークンと遷移と手が、行ごとに数え直した値と同じ。数える費用（ns/行）を
 * 記録に書く（記録の行の頭は「PRISM-BATCH-SHAPE」）。
 */
public class BatchShapeTest {
  private static final Logger log = LoggerFactory.getLogger(BatchShapeTest.class);

  private static final int ROWS = 512;
  private static final int REPEATS = 2_000;

  @Test
  public void countsMatchTheRows() {
    List<long[]> rows = new ArrayList<>(HipFixtures.mixedRows(ROWS, 31));
    rows.addAll(HipFixtures.longMixedRows());
    BatchPlan plan = HipFixtures.encode(rows).plan();
    BatchShape shape = new BatchShape();
    shape.add(plan);
    shape.add(plan);
    long pairs = 0;
    long longRows = 0;
    long[] histogram = new long[BatchShape.MAX_BLOCKS + 1];
    for (int row = 0; row < plan.rows(); row++) {
      int tokens = BatchPlan.FIXED_TOKENS + plan.events(row);
      int blocks = (int) Math.ceil(tokens / 16.0);
      histogram[blocks] += 2;
      pairs += 2L * blocks * blocks;
      longRows += tokens > 128 ? 2 : 0;
    }
    BatchShape.Summary summary = shape.summary();
    int n = plan.rows();
    assertEquals(summary.batches(), 2);
    assertEquals(summary.rows(), 2L * n);
    assertTrue(Arrays.equals(summary.histogram(), histogram), Arrays.toString(summary.histogram()));
    assertEquals(summary.longRows(), longRows);
    assertTrue(longRows >= 4, "the long rows must reach T > 128");
    assertEquals(summary.blockPairsPerRow(), (double) pairs / (2 * n), 1e-12);
    assertEquals(
        summary.tokensPerRow(),
        (BatchPlan.FIXED_TOKENS * (double) n + plan.count(Axis.EVENT)) / n,
        1e-9);
    assertEquals(summary.transitions(), 2L * plan.count(Axis.TRANSITION));
    assertEquals(summary.actions(), 2L * plan.count(Axis.ACTION));

    BatchShape timed = new BatchShape();
    long start = System.nanoTime();
    for (int repeat = 0; repeat < REPEATS; repeat++) {
      timed.add(plan);
    }
    double perRow = (System.nanoTime() - start) / ((double) REPEATS * n);
    log.info(
        "PRISM-BATCH-SHAPE {} add={} ns/row", summary.line(), String.format(Locale.ROOT, "%.2f", perRow));
  }
}
