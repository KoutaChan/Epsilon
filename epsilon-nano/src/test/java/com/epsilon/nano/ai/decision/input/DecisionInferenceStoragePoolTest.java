package com.epsilon.nano.ai.decision.input;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.data.DecisionInputRowCodec;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionBinaryArrayCodec;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 借用入力の再利用が保存形式・推論入力・通常の所有入力へ影響しないことを確かめる。 */
public class DecisionInferenceStoragePoolTest {
  private static final long RETAINED_BYTES = 32L * 1024 * 1024;

  @Test
  public void reusedRowsAndBucketsPreserveCanonicalAndCompactedInputs() throws IOException {
    var workspaces = new DecisionBatchBuilder.EncodingWorkspacePool(1);
    var engine = new GameEngine(770);
    GameStepResult step = engine.stepHanchan();
    var initial = ((GameStepResult.AwaitingDecisions) step).decisions().getFirst();
    var initialObservation = engine.observation(initial.player());
    var wide = new DecisionBucket(40, 16);
    var retainedOwned =
        build(
            DecisionBatchBuilder.inference(1, wide, workspaces),
            initialObservation,
            initial.legalActions(),
            1);
    byte[] retainedBytes = encoded(retainedOwned, 0);
    int multipleTransitions = 0, forced = 0;
    try (var pool = new DecisionInferenceStoragePool(RETAINED_BYTES, 1)) {
      short[] sharedCategories;
      try (var lease = pool.acquire()) {
        var first =
            build(
                lease.inference(3, wide, workspaces),
                initialObservation,
                initial.legalActions(),
                3);
        sharedCategories = first.inputs().stateCategories();
        poison(first);
      }
      for (int boundary = 0;
          boundary < 160 && !(step instanceof GameStepResult.HanchanEnded);
          boundary++) {
        if (step instanceof GameStepResult.RoundSettled) {
          step = engine.stepHanchan();
          continue;
        }
        var selections = new EngineSelectionBuffer();
        for (var point : ((GameStepResult.AwaitingDecisions) step).decisions()) {
          var observation = engine.observation(point.player());
          for (var actions :
              List.of(point.legalActions(), List.of(point.legalActions().getLast()))) {
            var bucket = DecisionBatchBuilder.selectInferenceBucket(actions);
            int rows = boundary % 2 + 1;
            var expected =
                build(
                    DecisionBatchBuilder.inference(rows, bucket, workspaces),
                    observation,
                    actions,
                    rows);
            try (var lease = pool.acquire()) {
              var actual =
                  build(lease.inference(rows, bucket, workspaces), observation, actions, rows);
              Assert.assertSame(actual.inputs().stateCategories(), sharedCategories);
              Assert.assertEquals(actual.capacity(), rows);
              assertEquivalent(actual, expected);
              if (bucket.actionTransitionCapacity() > 1) multipleTransitions++;
              if (actions.size() == 1) forced++;
              poison(actual);
            }
          }
          selections.add(point.id(), point.legalActions().getFirst());
        }
        step = engine.commitDecisions(selections);
      }
      Assert.assertTrue(multipleTransitions > 0);
      Assert.assertTrue(forced > 0);
      Assert.assertEquals(pool.snapshot().borrowedBuffers(), 0);
      Assert.assertEquals(pool.snapshot().retainedBuffers(), 1);
    }
    Assert.assertEquals(encoded(retainedOwned, 0), retainedBytes);
  }

  @Test
  public void simultaneousLeasesOwnIndependentInputsAndCopiesSurviveReturn() throws IOException {
    var fixture = fixture(771);
    var workspaces = new DecisionBatchBuilder.EncodingWorkspacePool(1);
    var bucket = DecisionBatchBuilder.selectInferenceBucket(fixture.actions);
    try (var pool = new DecisionInferenceStoragePool(RETAINED_BYTES, 2)) {
      DecisionHostBatch independent;
      byte[] expected;
      try (var firstLease = pool.acquire();
          var secondLease = pool.acquire()) {
        var first = build(firstLease.inference(1, bucket, workspaces), fixture, 1);
        expected = encoded(first, 0);
        independent = first.copyRow(0);
        var second = build(secondLease.inference(1, bucket, workspaces), fixture, 1);
        Assert.assertNotSame(first.inputs().stateCategories(), second.inputs().stateCategories());
        Assert.assertNotSame(
            first.inputs().transitionCategories(0), second.inputs().transitionCategories(0));
        poison(second);
        Assert.assertEquals(encoded(first, 0), expected);
        Assert.assertEquals(pool.snapshot().borrowedBuffers(), 2);
      }
      try (var lease = pool.acquire()) {
        poison(build(lease.inference(1, bucket, workspaces), fixture, 1));
        Assert.assertEquals(encoded(independent, 0), expected);
      }
    }
  }

  @Test
  public void retentionLimitsApplyToBytesAndBufferCount() {
    var fixture = fixture(772);
    var workspaces = new DecisionBatchBuilder.EncodingWorkspacePool(1);
    var bucket = DecisionBatchBuilder.selectInferenceBucket(fixture.actions);
    long oneBuffer;
    try (var pool = new DecisionInferenceStoragePool(RETAINED_BYTES, 1)) {
      try (var lease = pool.acquire()) {
        build(lease.inference(1, bucket, workspaces), fixture, 1);
      }
      oneBuffer = pool.snapshot().retainedBytes();
      Assert.assertTrue(oneBuffer > 0);
      for (int repeat = 0; repeat < 10; repeat++) {
        try (var lease = pool.acquire()) {
          build(lease.inference(1, bucket, workspaces), fixture, 1);
          build(lease.inference(1, bucket, workspaces), fixture, 1);
        }
        Assert.assertEquals(
            pool.snapshot(), new DecisionInferenceStoragePool.Snapshot(0, 1, oneBuffer));
      }
    }
    try (var pool = new DecisionInferenceStoragePool(oneBuffer, 8)) {
      try (var lease = pool.acquire()) {
        build(lease.inference(1, bucket, workspaces), fixture, 1);
        build(lease.inference(1, bucket, workspaces), fixture, 1);
      }
      Assert.assertEquals(
          pool.snapshot(), new DecisionInferenceStoragePool.Snapshot(0, 1, oneBuffer));
      try (var lease = pool.acquire()) {
        build(lease.inference(3, bucket, workspaces), fixture, 3);
      }
      Assert.assertEquals(pool.snapshot(), new DecisionInferenceStoragePool.Snapshot(0, 0, 0));
    }
  }

  @Test
  public void failedConstructionAndPartialEncodingReturnStorageOnlyOnLeaseClose()
      throws IOException {
    var fixture = fixture(773);
    var workspaces = new DecisionBatchBuilder.EncodingWorkspacePool(1);
    var bucket = DecisionBatchBuilder.selectInferenceBucket(fixture.actions);
    try (var pool = new DecisionInferenceStoragePool(RETAINED_BYTES, 1)) {
      var lease = pool.acquire();
      Assert.expectThrows(
          IllegalArgumentException.class, () -> lease.inference(0, bucket, workspaces));
      Assert.assertEquals(pool.snapshot().borrowedBuffers(), 0);
      var builder = lease.inference(2, bucket, workspaces);
      try (var session = builder.openEncoding()) {
        session.encodeObservedInferenceRow(0, fixture.observation, fixture.actions);
        Assert.expectThrows(
            NullPointerException.class,
            () -> session.encodeObservedInferenceRow(1, null, fixture.actions));
      }
      Assert.assertEquals(pool.snapshot().borrowedBuffers(), 1);
      lease.close();
      lease.close();
      Assert.assertEquals(pool.snapshot().borrowedBuffers(), 0);
      Assert.expectThrows(
          IllegalStateException.class, () -> lease.inference(1, bucket, workspaces));
      try (var next = pool.acquire()) {
        assertEquivalent(
            build(next.inference(2, bucket, workspaces), fixture, 2),
            build(DecisionBatchBuilder.inference(2, bucket, workspaces), fixture, 2));
      }
    }
  }

  @Test
  public void poolClosePreservesBorrowedInputsAndDiscardsLateReturns() throws IOException {
    var fixture = fixture(774);
    var workspaces = new DecisionBatchBuilder.EncodingWorkspacePool(1);
    var bucket = DecisionBatchBuilder.selectInferenceBucket(fixture.actions);
    var pool = new DecisionInferenceStoragePool(RETAINED_BYTES, 1);
    try (var lease = pool.acquire()) {
      var batch = build(lease.inference(1, bucket, workspaces), fixture, 1);
      byte[] expected = encoded(batch, 0);
      pool.close();
      pool.close();
      Assert.assertEquals(encoded(batch, 0), expected);
      Assert.assertEquals(pool.snapshot().borrowedBuffers(), 1);
      Assert.expectThrows(IllegalStateException.class, pool::acquire);
      Assert.expectThrows(
          IllegalStateException.class, () -> lease.inference(1, bucket, workspaces));
    }
    Assert.assertEquals(pool.snapshot(), new DecisionInferenceStoragePool.Snapshot(0, 0, 0));
  }

  @Test
  public void concurrentBorrowersKeepTheirOwnEncodedRows() throws Exception {
    var workspaces = new DecisionBatchBuilder.EncodingWorkspacePool(4);
    try (var pool = new DecisionInferenceStoragePool(RETAINED_BYTES, 2);
        var workers = Executors.newFixedThreadPool(4)) {
      List<Future<?>> tasks = new ArrayList<>();
      for (int worker = 0; worker < 4; worker++) {
        int seed = 780 + worker;
        tasks.add(
            workers.submit(
                () -> {
                  var fixture = fixture(seed);
                  var bucket = DecisionBatchBuilder.selectInferenceBucket(fixture.actions);
                  try {
                    for (int repeat = 0; repeat < 20; repeat++) {
                      int rows = repeat % 3 + 1;
                      var expected =
                          build(
                              DecisionBatchBuilder.inference(rows, bucket, workspaces),
                              fixture,
                              rows);
                      try (var lease = pool.acquire()) {
                        var actual =
                            build(lease.inference(rows, bucket, workspaces), fixture, rows);
                        assertEquivalent(actual, expected);
                        poison(actual);
                      }
                    }
                  } catch (IOException failure) {
                    throw new AssertionError(failure);
                  }
                }));
      }
      for (var task : tasks) task.get(30, TimeUnit.SECONDS);
      Assert.assertEquals(pool.snapshot().borrowedBuffers(), 0);
      Assert.assertTrue(pool.snapshot().retainedBuffers() <= 2);
      Assert.assertTrue(pool.snapshot().retainedBytes() <= RETAINED_BYTES);
    }
  }

  private static void assertEquivalent(DecisionHostBatch actual, DecisionHostBatch expected)
      throws IOException {
    Assert.assertEquals(actual.size(), expected.size());
    Assert.assertEquals(actual.bucket(), expected.bucket());
    for (int row = 0; row < actual.size(); row++) {
      Assert.assertEquals(actual.legalActionCount(row), expected.legalActionCount(row));
      Assert.assertEquals(encoded(actual, row), encoded(expected, row));
    }
    var actualRows = actual.sliceRows(0, actual.size());
    var expectedRows = expected.sliceRows(0, expected.size());
    var indices = actualRows.inferenceIndexLayout(true);
    Assert.assertEquals(indices, expectedRows.inferenceIndexLayout(true));
    int[] actualIndices = new int[indices.totalCount()],
        expectedIndices = new int[indices.totalCount()];
    actualRows.copyInferenceIndicesTo(IntBuffer.wrap(actualIndices), indices);
    expectedRows.copyInferenceIndicesTo(IntBuffer.wrap(expectedIndices), indices);
    Assert.assertEquals(actualIndices, expectedIndices);
    var layout = actualRows.inferenceInputLayout(indices);
    short[] actualCategories = new short[layout.categoricalElementCount()];
    short[] expectedCategories = new short[actualCategories.length];
    float[] actualNumerics = new float[layout.numericElementCount()];
    float[] expectedNumerics = new float[actualNumerics.length];
    actualRows.copyInferenceInputsTo(
        ShortBuffer.wrap(actualCategories), FloatBuffer.wrap(actualNumerics), layout);
    expectedRows.copyInferenceInputsTo(
        ShortBuffer.wrap(expectedCategories), FloatBuffer.wrap(expectedNumerics), layout);
    Assert.assertEquals(actualCategories, expectedCategories);
    Assert.assertEquals(actualNumerics, expectedNumerics);
  }

  private static byte[] encoded(DecisionHostBatch batch, int row) throws IOException {
    var bytes = new ByteArrayOutputStream();
    try (var output = new DataOutputStream(bytes)) {
      DecisionInputRowCodec.write(
          output, batch.sliceRows(row, 1), new EpsilonDecisionBinaryArrayCodec.Scratch());
    }
    return bytes.toByteArray();
  }

  private static void poison(DecisionHostBatch batch) {
    Arrays.fill(batch.inputs().stateCategories(), (short) -1234);
    Arrays.fill(batch.inputs().stateNumerics(), Float.NaN);
    for (int row = 0; row < batch.size(); row++) {
      Arrays.fill(batch.inputs().transitionCategories(row), (short) -1234);
      Arrays.fill(batch.inputs().transitionNumerics(row), Float.NaN);
    }
  }

  private static DecisionHostBatch build(DecisionBatchBuilder builder, Fixture fixture, int rows) {
    return build(builder, fixture.observation, fixture.actions, rows);
  }

  private static DecisionHostBatch build(
      DecisionBatchBuilder builder, PublicObservation observation, List<Action> actions, int rows) {
    try (var session = builder.openEncoding()) {
      for (int row = 0; row < rows; row++)
        session.encodeObservedInferenceRow(row, observation, actions);
    }
    return builder.buildEncodedInferenceRows();
  }

  private static Fixture fixture(long seed) {
    var engine = new GameEngine(seed);
    var point = ((GameStepResult.AwaitingDecisions) engine.stepHanchan()).decisions().getFirst();
    return new Fixture(engine.observation(point.player()), point.legalActions());
  }

  private record Fixture(PublicObservation observation, List<Action> actions) {}
}
