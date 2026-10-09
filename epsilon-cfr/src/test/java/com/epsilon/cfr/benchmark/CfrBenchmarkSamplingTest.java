package com.epsilon.cfr.benchmark;

import com.epsilon.cfr.benchmark.CfrPerformanceBenchmark.SamplingMode;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.cfr.data.LogWeightedReplay;
import com.epsilon.cfr.data.ObservationPayload;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.random.RandomGenerator;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CfrBenchmarkSamplingTest {
  @Test
  public void uniformDistinctEnumeratesEveryOrderedPairOnceAndLoadsOnlySelectedRows() {
    List<CfrTrainingRow> source = rows(4);
    AtomicInteger loads = new AtomicInteger();
    var sampler =
        new LogWeightedReplay(
            source.size(),
            index -> source.get(index).logRegretWeight(),
            index -> {
              loads.incrementAndGet();
              return source.get(index);
            });
    var pairs = new HashSet<List<CfrSampleId>>();
    for (int first = 0; first < 4; first++) {
      for (int second = 0; second < 3; second++) {
        List<CfrTrainingRow> sample =
            SamplingMode.UNIFORM_DISTINCT.sample(sampler, 2, new Draws(first, second));
        Assert.assertNotEquals(sample.get(0).id(), sample.get(1).id());
        Assert.assertTrue(pairs.add(sample.stream().map(CfrTrainingRow::id).toList()));
        for (CfrTrainingRow row : sample)
          Assert.assertSame(row, source.get((int) row.id().decisionId()));
      }
    }
    Assert.assertEquals(pairs.size(), 12);
    Assert.assertEquals(loads.get(), 24);
  }

  @Test
  public void fullDistinctBatchIncludesZeroWeightRowsAndPreservesTargets() {
    List<CfrTrainingRow> source = rows(64);
    var sampler =
        new LogWeightedReplay(
            source.size(), index -> source.get(index).logRegretWeight(), source::get);
    List<CfrTrainingRow> sample =
        SamplingMode.UNIFORM_DISTINCT.sample(sampler, source.size(), new SplittableRandom(81));
    Assert.assertEquals(new HashSet<>(sample.stream().map(CfrTrainingRow::id).toList()).size(), 64);
    for (CfrTrainingRow row : sample)
      Assert.assertSame(row, source.get((int) row.id().decisionId()));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> SamplingMode.UNIFORM_DISTINCT.sample(sampler, 65, new SplittableRandom(81)));
  }

  @Test
  public void weightedModeKeepsProductionSamplingSequence() {
    var source = rows(8);
    var sampler =
        new LogWeightedReplay(
            source.size(), index -> source.get(index).logRegretWeight(), source::get);
    List<CfrTrainingRow> expected = sampler.sample(1024, new SplittableRandom(19));
    List<CfrTrainingRow> actual =
        SamplingMode.parse("weighted").sample(sampler, 1024, new SplittableRandom(19));
    for (int row = 0; row < expected.size(); row++)
      Assert.assertSame(actual.get(row), expected.get(row));
    Assert.assertEquals(SamplingMode.parse("uniform-distinct"), SamplingMode.UNIFORM_DISTINCT);
    Assert.expectThrows(IllegalArgumentException.class, () -> SamplingMode.parse("uniform"));
  }

  private static List<CfrTrainingRow> rows(int count) {
    List<CfrTrainingRow> rows = new ArrayList<>();
    for (int index = 0; index < count; index++) {
      rows.add(
          new CfrTrainingRow(
              new CfrSampleId(1, 0, index, 0),
              new int[] {0, 1},
              0,
              new double[] {.25, .75},
              .4,
              index == 0 ? Double.NEGATIVE_INFINITY : -index,
              -2 * index,
              ObservationPayload.copyOf(new byte[] {(byte) index}),
              new double[] {.2, -.1}));
    }
    return rows;
  }

  private static final class Draws implements RandomGenerator {
    private final int[] draws;
    private int position;

    Draws(int... draws) {
      this.draws = draws;
    }

    @Override
    public int nextInt(int bound) {
      int value = draws[position++];
      Assert.assertTrue(value >= 0 && value < bound);
      return value;
    }

    @Override
    public long nextLong() {
      throw new AssertionError("Unexpected random draw");
    }
  }
}
