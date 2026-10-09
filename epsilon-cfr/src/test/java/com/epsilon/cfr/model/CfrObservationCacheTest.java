package com.epsilon.cfr.model;

import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CfrObservationCacheTest {
  @Test
  public void reloadedRowsWithTheSameIdNeverReuseAnotherObservation() {
    var first = row(772);
    var replacement = withId(row(773), first.id());
    try (var cache = new CfrObservationCache()) {
      var original = cache.get(first);
      var reloaded = cache.get(replacement);
      Assert.assertSame(cache.get(first), original);
      Assert.assertSame(cache.get(replacement), reloaded);
      Assert.assertNotSame(original, reloaded);
      Assert.assertEquals(CfrObservationCodec.encodePayload(original), first.observation());
      Assert.assertEquals(CfrObservationCodec.encodePayload(reloaded), replacement.observation());
      Assert.assertFalse(first.observation().equals(replacement.observation()));
    }
  }

  @Test
  public void batchingAndEvictionPreserveRetainedObservationBytes() {
    var first = row(772);
    var second = row(773);
    var third = row(774);
    try (var cache = new CfrObservationCache(Long.MAX_VALUE, 2)) {
      var a = cache.get(first);
      var b = cache.get(second);
      var combined = CfrObservationCodec.batch(List.of(b, a, b));
      Assert.assertEquals(combined.size(), 3);
      Assert.assertEquals(
          CfrObservationCodec.encodePayload(new CfrObservationCodec.Decoded(combined.copyRow(1))),
          first.observation());
      Assert.assertSame(cache.get(first), a);
      cache.get(third);
      Assert.assertSame(cache.get(first), a);
      Assert.assertNotSame(cache.get(second), b);
      Assert.assertEquals(CfrObservationCodec.encodePayload(a), first.observation());
      Assert.assertEquals(CfrObservationCodec.encodePayload(b), second.observation());
      cache.close();
      Assert.assertNotSame(cache.get(first), a);
    }
  }

  @Test
  public void capacityIncludesDecodedArraysAndRetainedEncodedRow() {
    var row = row(772);
    // 符号化した元行だけが収まる容量では、復号済みの特徴配列を追加保持しない。
    try (var cache = new CfrObservationCache(row.serializedBytes(), 8)) {
      var first = cache.get(row);
      Assert.assertNotSame(cache.get(row), first);
      Assert.assertEquals(CfrObservationCodec.encodePayload(first), row.observation());
    }
  }

  private static CfrTrainingRow row(long seed) {
    var context = CfrModelTest.context(seed);
    int[] actions = context.legalActions().stream().mapToInt(action -> action.toIndex()).toArray();
    double[] policy = new double[actions.length];
    Arrays.fill(policy, 1d / actions.length);
    return new CfrTrainingRow(
        new CfrSampleId(1, seed, context.decisionId(), context.seat()),
        actions,
        0,
        policy,
        .5,
        0,
        0,
        CfrObservationCodec.encodePayload(context));
  }

  private static CfrTrainingRow withId(CfrTrainingRow row, CfrSampleId id) {
    return new CfrTrainingRow(
        id,
        row.legalActionIds(),
        row.selectedSlot(),
        row.targetPolicy(),
        row.utility(),
        row.logRegretWeight(),
        row.logAverageWeight(),
        row.observation());
  }
}
