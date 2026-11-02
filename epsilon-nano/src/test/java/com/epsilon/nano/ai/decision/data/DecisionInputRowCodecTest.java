package com.epsilon.nano.ai.decision.data;

import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.input.DecisionInferenceStoragePool;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

public class DecisionInputRowCodecTest {
  @Test
  public void rowOffsetsSelectDifferentEncodedStates() throws IOException {
    var expected = new ArrayList<byte[]>();
    DecisionBatchBuilder grouped = null;
    for (int row = 0; row < 3; row++) {
      var engine = new GameEngine(770 + row);
      var point = ((GameStepResult.AwaitingDecisions) engine.stepHanchan()).decisions().getFirst();
      var bucket = DecisionBatchBuilder.selectInferenceBucket(point.legalActions());
      if (grouped == null) grouped = DecisionBatchBuilder.inference(3, bucket);
      var single = DecisionBatchBuilder.inference(1, bucket);
      single.encodeObservedInferenceRow(
          0, engine.observation(point.player()), point.legalActions());
      expected.add(canonical(single.buildEncodedInferenceRows(), 0));
      grouped.encodeObservedInferenceRow(
          row, engine.observation(point.player()), point.legalActions());
    }
    Assert.assertFalse(Arrays.equals(expected.get(0), expected.get(1)));
    var host = grouped.buildEncodedInferenceRows();
    for (int row : new int[] {2, 0, 1}) {
      var encoded = DecisionInputRowCodec.encode(host.sliceRows(row, 1));
      Assert.assertEquals(encoded.openStream().readAllBytes(), expected.get(row));
    }
  }

  @Test
  public void compactRowsKeepCanonicalBytesAfterBackingStorageReuse() throws IOException {
    var workspace = new DecisionBatchBuilder.EncodingWorkspacePool(1);
    var retained = new ArrayList<DecisionInputRowCodec.EncodedRow>();
    var expected = new ArrayList<byte[]>();
    boolean single = false, multiple = false, reduced = false;
    try (var pool = new DecisionInferenceStoragePool(32L * 1024 * 1024, 4)) {
      var engine = new GameEngine(770);
      GameStepResult step = engine.stepHanchan();
      for (int boundary = 0;
          boundary < 160 && !(step instanceof GameStepResult.HanchanEnded);
          boundary++) {
        if (step instanceof GameStepResult.RoundSettled) {
          step = engine.stepHanchan();
          continue;
        }
        var selections = new EngineSelectionBuffer();
        for (var point : ((GameStepResult.AwaitingDecisions) step).decisions()) {
          var bucket = DecisionBatchBuilder.selectInferenceBucket(point.legalActions());
          try (var lease = pool.acquire()) {
            var builder = lease.inference(3, bucket, workspace);
            try (var session = builder.openEncoding()) {
              for (int row = 0; row < 3; row++)
                session.encodeObservedInferenceRow(
                    row, engine.observation(point.player()), point.legalActions());
            }
            var host = builder.buildEncodedInferenceRows();
            for (int row : new int[] {2, 0, 1}) {
              byte[] reference = canonical(host, row);
              var encoded = DecisionInputRowCodec.encode(host.sliceRows(row, 1));
              Assert.assertEquals(encoded.openStream().readAllBytes(), reference);
              var dense = DecisionInputRowCodec.encode(host.copyRow(row).sliceRows(0, 1));
              Assert.assertEquals(dense.openStream().readAllBytes(), reference);
              Assert.assertEquals(dense.storageBytes(), dense.size());
              reduced |= encoded.storageBytes() < encoded.size();
              retained.add(encoded);
              expected.add(reference);
            }
          }
          single |= bucket.actionTransitionCapacity() == 1;
          multiple |= bucket.actionTransitionCapacity() > 1;
          selections.add(point.id(), point.legalActions().getFirst());
        }
        step = engine.commitDecisions(selections);
      }
      Assert.assertEquals(pool.snapshot().borrowedBuffers(), 0);
    }
    Assert.assertTrue(single && multiple && reduced);
    for (int index = 0; index < retained.size(); index++)
      Assert.assertEquals(retained.get(index).openStream().readAllBytes(), expected.get(index));
  }

  @Test
  public void streamAndWritesPreserveZerosBoundsAndOwnership() throws IOException {
    var host = firstRow();
    var row = DecisionInputRowCodec.encode(host.sliceRows(0, 1));
    byte[] expected = canonical(host, 0);
    try (var input = row.openStream()) {
      Assert.assertEquals(input.available(), expected.length);
      Assert.assertEquals(input.skip(-1), 0L);
      int position = 0;
      byte[] target = new byte[53];
      while (position < expected.length) {
        int count = input.read(target, 7, 37);
        Assert.assertEquals(
            Arrays.copyOfRange(target, 7, 7 + count),
            Arrays.copyOfRange(expected, position, position + count));
        position += count;
      }
      Assert.assertEquals(input.available(), 0);
      Assert.assertEquals(input.read(), -1);
      Assert.assertEquals(input.read(target, 0, 1), -1);
      Assert.assertEquals(input.read(target, 0, 0), 0);
      Assert.assertEquals(input.skip(Long.MAX_VALUE), 0L);
      Assert.expectThrows(IndexOutOfBoundsException.class, () -> input.read(target, 52, 2));
    }
    for (int position = 0; position < expected.length; position += 31) {
      try (var input = row.openStream()) {
        Assert.assertEquals(input.skip(position), (long) position);
        Assert.assertEquals(input.read(), Byte.toUnsignedInt(expected[position]));
        Assert.assertEquals(input.skip(Long.MAX_VALUE), (long) expected.length - position - 1);
        Assert.assertEquals(input.read(), -1);
      }
    }
    var hostile =
        new OutputStream() {
          @Override
          public void write(int value) {}

          @Override
          public void write(byte[] bytes, int offset, int length) {
            Arrays.fill(bytes, (byte) 99);
          }
        };
    try (var input = row.openStream()) {
      Assert.assertEquals(input.transferTo(hostile), (long) expected.length);
    }
    row.writeTo(hostile, new byte[103]);
    for (int bufferSize : new int[] {1, 37, 64 * 1024}) {
      var output = new ByteArrayOutputStream();
      row.writeTo(output, new byte[bufferSize]);
      Assert.assertEquals(output.toByteArray(), expected);
    }
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> row.writeTo(new ByteArrayOutputStream(), new byte[0]));
    byte[] destination = new byte[expected.length + 7];
    Arrays.fill(destination, (byte) 19);
    row.copyTo(destination, 3);
    Assert.assertEquals(Arrays.copyOfRange(destination, 3, 3 + expected.length), expected);
    Assert.assertEquals(destination[2], (byte) 19);
    Assert.assertEquals(destination[destination.length - 1], (byte) 19);
    Arrays.fill(destination, (byte) 88);
    Assert.assertEquals(row.openStream().readAllBytes(), expected);
  }

  @Test
  public void zeroedWritesPreserveDestinationOffsetPositionLimitAndOrder() throws IOException {
    var compact = firstRow();
    for (var host : new DecisionHostBatch[] {compact, compact.copyRow(0)}) {
      var row = host.sliceRows(0, 1);
      byte[] expected = canonical(host, 0);
      for (var order : new ByteOrder[] {ByteOrder.BIG_ENDIAN, ByteOrder.LITTLE_ENDIAN}) {
        for (int arrayOffset : new int[] {0, 7}) {
          int position = 11;
          byte[] backing = new byte[arrayOffset + position + expected.length + 5];
          Arrays.fill(backing, (byte) 73);
          int start = arrayOffset + position;
          Arrays.fill(backing, start, start + expected.length, (byte) 0);
          byte[] reference = backing.clone();
          System.arraycopy(expected, 0, reference, start, expected.length);
          var destination = ByteBuffer.wrap(backing).position(arrayOffset).slice().order(order);
          int limit = position + expected.length + 3;
          destination.position(position).limit(limit).mark();
          DecisionInputRowCodec.writeZeroed(destination, row);
          Assert.assertEquals(backing, reference);
          Assert.assertEquals(destination.arrayOffset(), arrayOffset);
          Assert.assertEquals(destination.position(), position + expected.length);
          Assert.assertEquals(destination.limit(), limit);
          Assert.assertSame(destination.order(), order);
          Assert.assertEquals(destination.reset().position(), position);
        }
      }
    }
  }

  @Test
  public void denseInputPreservesRawFloatBitsInUnusedCells() throws IOException {
    var original = firstRow();
    var slice = original.sliceRows(0, 1);
    short[] categories = new short[slice.inputCategoricalElementCount()];
    float[] numerics = new float[slice.inputNumericElementCount()];
    slice.copyInputCategoriesTo(ShortBuffer.wrap(categories));
    slice.copyInputNumericsTo(FloatBuffer.wrap(numerics));
    numerics[numerics.length - 1] = Float.intBitsToFloat(0x80000000);
    numerics[numerics.length - 2] = Float.intBitsToFloat(0x7fc00019);
    var host =
        DecisionHostBatch.takeEncodedInferenceBatch(1, original.bucket(), categories, numerics);
    var encoded = DecisionInputRowCodec.encode(host.sliceRows(0, 1));
    Assert.assertEquals(encoded.storageBytes(), encoded.size());
    byte[] actual = encoded.openStream().readAllBytes();
    Assert.assertEquals(actual, canonical(host, 0));
    var bits = ByteBuffer.wrap(actual).order(ByteOrder.LITTLE_ENDIAN);
    Assert.assertEquals(bits.getInt(actual.length - 4), 0x80000000);
    Assert.assertEquals(bits.getInt(actual.length - 8), 0x7fc00019);
    // codecの復号時に非有限値を拒否する既存契約は変えず、符号化そのもののビット保存を検証する。
  }

  private static DecisionHostBatch firstRow() {
    var engine = new GameEngine(770);
    var point = ((GameStepResult.AwaitingDecisions) engine.stepHanchan()).decisions().getFirst();
    var builder =
        DecisionBatchBuilder.inference(
            1, DecisionBatchBuilder.selectInferenceBucket(point.legalActions()));
    builder.encodeObservedInferenceRow(0, engine.observation(point.player()), point.legalActions());
    return builder.buildEncodedInferenceRows();
  }

  private static byte[] canonical(DecisionHostBatch host, int row) throws IOException {
    var output = new ByteArrayOutputStream();
    try (var data = new DataOutputStream(output)) {
      DecisionInputRowCodec.write(
          data, host.sliceRows(row, 1), new EpsilonDecisionBinaryArrayCodec.Scratch());
    }
    return output.toByteArray();
  }
}
