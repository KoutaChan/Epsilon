package com.epsilon.runtime;

import java.util.ArrayList;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 推論要求の分類と分割、キュー拡張後の順序、最古の要求に基づく期限、待機中の公平な処理順序を検証する。 */
public class InferenceBatchQueueTest {
  @Test
  public void splitsAccumulatedRowsAtCapacityWithoutCopyingRequests() {
    var queue = new InferenceBatchQueue<Object, String, Object>(1000, (e, k) -> 4096);
    Object evaluator = new Object();
    var rows = new ArrayList<Object>();
    for (int i = 0; i < 4500; i++) {
      Object row = new Object();
      rows.add(row);
      queue.add(evaluator, "shape", row, i < 3000 ? 0 : 100);
    }
    var first = queue.remove(queue.peekBest(BatchDispatchReason.FULL, e -> true, 100));
    Assert.assertEquals(queue.nextDeadlineNanos(e -> true), 1100L);
    var second = queue.remove(queue.peekBest(BatchDispatchReason.SUPPLY, e -> true, 100));
    Assert.assertEquals(first.rows().size(), 4096);
    Assert.assertEquals(second.rows().size(), 404);
    for (int i = 0; i < rows.size(); i++) {
      Assert.assertSame(i < 4096 ? first.rows().get(i) : second.rows().get(i - 4096), rows.get(i));
    }
    Assert.assertFalse(queue.hasQueuedRows());
  }

  @Test
  public void separatesEqualEvaluatorsByIdentityAndKeysByCompatibility() {
    var queue = new InferenceBatchQueue<String, String, Integer>(1000, (e, k) -> 2);
    String first = new String("same-model-name");
    String second = new String("same-model-name");
    queue.add(first, "policy", 1, 0);
    queue.add(second, "policy", 2, 0);
    queue.add(first, "policy-value", 3, 0);
    Assert.assertNull(queue.peekBest(BatchDispatchReason.FULL, e -> true, 0));
    var a = queue.remove(queue.peekBest(BatchDispatchReason.SUPPLY, e -> true, 0));
    var b = queue.remove(queue.peekBest(BatchDispatchReason.SUPPLY, e -> true, 0));
    var c = queue.remove(queue.peekBest(BatchDispatchReason.SUPPLY, e -> true, 0));
    Assert.assertSame(a.evaluator(), first);
    Assert.assertSame(b.evaluator(), second);
    Assert.assertSame(c.evaluator(), first);
    Assert.assertEquals(a.rows(), List.of(1));
    Assert.assertEquals(b.rows(), List.of(2));
    Assert.assertEquals(c.rows(), List.of(3));
    Assert.assertEquals(c.key(), "policy-value");
  }

  @Test
  public void inspectingBlockedCandidateDoesNotAdvanceEvaluatorRotation() {
    var queue = new InferenceBatchQueue<Object, Integer, Integer>(1000, (e, k) -> 1);
    Object first = new Object();
    Object second = new Object();
    queue.add(first, 1, 10, 0);
    queue.add(second, 1, 20, 0);
    Assert.assertSame(queue.peekBest(BatchDispatchReason.FULL, e -> true, 0).evaluator(), first);
    Assert.assertSame(
        queue.peekBest(BatchDispatchReason.FULL, e -> e == second, 0).evaluator(), second);
    Assert.assertSame(queue.peekBest(BatchDispatchReason.FULL, e -> true, 0).evaluator(), first);
  }

  @Test
  public void deadlineRemainsAttachedToOldestRowWhenLaterRowsArrive() {
    var queue = new InferenceBatchQueue<String, String, Integer>(1000, (e, k) -> 100);
    queue.add("model", "rare", 1, 0);
    queue.add("model", "common", 2, 100);
    queue.add("model", "rare", 3, 900);
    Assert.assertNull(queue.peekBest(BatchDispatchReason.DEADLINE, e -> true, 999));
    var candidate = queue.peekBest(BatchDispatchReason.DEADLINE, e -> true, 1000);
    Assert.assertEquals(candidate.key(), "rare");
    var batch = queue.remove(candidate);
    Assert.assertEquals(batch.rows(), List.of(1, 3));
    Assert.assertEquals(queue.nextDeadlineNanos(e -> true), 1100L);
  }

  @Test
  public void expiredBlockedEvaluatorDoesNotMaskAnotherEvaluatorsDeadline() {
    var queue = new InferenceBatchQueue<String, String, Integer>(1000, (e, k) -> 4);
    queue.add("blocked-gpu", "shape", 1, 0);
    queue.add("available-gpu", "shape", 2, 500);
    Assert.assertEquals(queue.nextDeadlineNanos(e -> e.equals("available-gpu")), 1500L);
    Assert.assertNull(
        queue.peekBest(BatchDispatchReason.DEADLINE, e -> e.equals("available-gpu"), 1000));
    var candidate =
        queue.peekBest(BatchDispatchReason.DEADLINE, e -> e.equals("available-gpu"), 1500);
    Assert.assertEquals(candidate.evaluator(), "available-gpu");
  }

  @Test
  public void growingWrappedQueuePreservesReferenceOrderAndTimestamps() {
    var queue = new InferenceBatchQueue<String, String, Integer>(1000, (e, k) -> 16);
    for (int row = 0; row < 20; row++) queue.add("model", "shape", row, row);
    Assert.assertEquals(
        queue.remove(queue.peekBest(BatchDispatchReason.FULL, e -> true, 20)).rows().getLast(), 15);
    for (int row = 20; row < 80; row++) queue.add("model", "shape", row, row);
    var actual = new ArrayList<Integer>();
    while (queue.hasQueuedRows()) {
      var batch = queue.remove(queue.peekBest(BatchDispatchReason.SUPPLY, e -> true, 80));
      Assert.assertEquals(batch.enqueuedNanos(), batch.rows().getFirst().longValue());
      actual.addAll(batch.rows());
    }
    Assert.assertEquals(actual, java.util.stream.IntStream.range(16, 80).boxed().toList());
  }
}
