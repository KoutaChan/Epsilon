package com.epsilon.cfr.data;

import com.epsilon.cfr.algorithm.LcfrTargets;
import com.epsilon.cfr.collection.Prediction;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.data.DecisionInputRowCodec;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ReadOnlyBufferException;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class ObservationPayloadTest {
  @Test
  public void compactDecisionInputKeepsCanonicalPayloadAndStreamContracts() throws IOException {
    var engine = new GameEngine(770);
    var point = ((GameStepResult.AwaitingDecisions) engine.stepHanchan()).decisions().getFirst();
    var builder =
        DecisionBatchBuilder.inference(
            1, DecisionBatchBuilder.selectInferenceBucket(point.legalActions()));
    builder.encodeObservedInferenceRow(0, engine.observation(point.player()), point.legalActions());
    var row = DecisionInputRowCodec.encode(builder.buildEncodedInferenceRows().sliceRows(0, 1));
    var payload = ObservationPayload.fromDecisionInput(0x43465233, row);
    byte[] reference = new byte[4 + row.size()];
    ByteBuffer.wrap(reference).putInt(0x43465233);
    row.copyTo(reference, 4);
    Assert.assertEquals(payload.size(), reference.length);
    Assert.assertEquals(payload.toByteArray(), reference);
    var dense = ObservationPayload.copyOf(reference);
    Assert.assertEquals(payload, dense);
    Assert.assertEquals(dense, payload);
    Assert.assertEquals(payload.hashCode(), dense.hashCode());
    payload.toByteArray()[5] = 99;
    Assert.assertFalse(payload.readOnlyBuffer().hasArray());
    try (var stream = payload.openStream()) {
      Assert.assertEquals(stream.available(), reference.length);
      Assert.assertEquals(stream.skip(7), 7L);
      Assert.assertEquals(stream.read(), Byte.toUnsignedInt(reference[7]));
      Assert.assertEquals(stream.available(), reference.length - 8);
      Assert.assertEquals(stream.skip(Long.MAX_VALUE), (long) reference.length - 8);
      Assert.assertEquals(stream.read(), -1);
    }
    try (var stream = payload.openStream()) {
      Assert.assertEquals(
          stream.transferTo(
              new OutputStream() {
                @Override
                public void write(int value) {}

                @Override
                public void write(byte[] bytes, int offset, int length) {
                  java.util.Arrays.fill(bytes, (byte) 98);
                }
              }),
          (long) reference.length);
    }
    Assert.assertEquals(payload.toByteArray(), reference);
    var expected = new ByteArrayOutputStream();
    dense.write(new DataOutputStream(expected));
    var actual = new ByteArrayOutputStream();
    payload.write(new DataOutputStream(actual));
    Assert.assertEquals(actual.toByteArray(), expected.toByteArray());
    Assert.assertEquals(ObservationPayload.read(input(actual.toByteArray())), payload);
  }

  @Test
  public void mutableInputsAndExportsNeverAliasOwnedBytes() throws IOException {
    byte[] source = {4, 5, 6};
    var payload = ObservationPayload.copyOf(source);
    source[0] = 99;
    payload.toByteArray()[1] = 98;
    var view = payload.readOnlyBuffer();
    Assert.assertTrue(view.isReadOnly());
    Assert.assertFalse(view.hasArray());
    Assert.expectThrows(ReadOnlyBufferException.class, () -> view.put(0, (byte) 97));
    Assert.expectThrows(ReadOnlyBufferException.class, view::array);
    Assert.assertEquals(view.get(), (byte) 4);
    Assert.assertEquals(payload.readOnlyBuffer().position(), 0);
    try (var first = payload.openStream();
        var second = payload.openStream()) {
      Assert.assertEquals(first.read(), 4);
      Assert.assertEquals(first.read(), 5);
      Assert.assertEquals(second.read(), 4);
      byte[] exported = second.readAllBytes();
      exported[0] = 96;
    }
    try (var input = payload.openStream()) {
      Assert.assertEquals(
          input.transferTo(
              new OutputStream() {
                @Override
                public void write(int value) {}

                @Override
                public void write(byte[] buffer, int offset, int length) {
                  java.util.Arrays.fill(buffer, (byte) 95);
                }
              }),
          3L);
    }
    Assert.assertEquals(payload.toByteArray(), new byte[] {4, 5, 6});
    Assert.assertEquals(payload, ObservationPayload.copyOf(new byte[] {4, 5, 6}));
    Assert.assertEquals(
        payload.hashCode(), ObservationPayload.copyOf(new byte[] {4, 5, 6}).hashCode());
    Assert.assertNotEquals(payload, ObservationPayload.copyOf(new byte[] {4, 5}));
  }

  @Test
  public void streamPreservesReadBoundsAndEndOfInput() throws IOException {
    var payload = ObservationPayload.copyOf(new byte[] {(byte) 255, 5, 6});
    try (var input = payload.openStream()) {
      Assert.assertEquals(input.available(), 3);
      Assert.assertEquals(input.skip(-1), 0L);
      Assert.assertEquals(input.read(), 255);
      var output = new byte[4];
      Assert.assertEquals(input.read(output, 1, 3), 2);
      Assert.assertEquals(output, new byte[] {0, 5, 6, 0});
      Assert.assertEquals(input.available(), 0);
      Assert.assertEquals(input.read(), -1);
      Assert.assertEquals(input.read(output, 0, 1), -1);
      Assert.assertEquals(input.read(output, 0, 0), 0);
      Assert.assertEquals(input.skip(Long.MAX_VALUE), 0L);
      Assert.expectThrows(IndexOutOfBoundsException.class, () -> input.read(output, 3, 2));
    }
    try (var input = payload.openStream()) {
      Assert.assertEquals(input.skip(Long.MAX_VALUE), 3L);
      Assert.assertEquals(input.read(), -1);
    }
  }

  @Test
  public void bufferSnapshotCopiesOnlyRemainingBytesWithoutMovingSource() {
    for (ByteBuffer buffer : List.of(ByteBuffer.allocate(8), ByteBuffer.allocateDirect(8))) {
      buffer.put(new byte[] {0, 1, 2, 3, 4, 5, 6, 7}).position(2).limit(6);
      var payload = ObservationPayload.copyOf(buffer);
      Assert.assertEquals(buffer.position(), 2);
      Assert.assertEquals(buffer.limit(), 6);
      buffer.put(2, (byte) 99);
      Assert.assertEquals(payload.toByteArray(), new byte[] {2, 3, 4, 5});
      Assert.assertEquals(payload.size(), 4);
    }
  }

  @Test
  public void immutablePayloadIsSharedFromPredictionThroughTargets() {
    double[] policy = {.25, .75};
    byte[] source = {4, 5, 6};
    var prediction = new Prediction(policy, ObservationPayload.copyOf(source));
    policy[0] = 1;
    source[0] = 99;
    prediction.targetPolicy()[0] = 1;
    Assert.assertEquals(prediction.targetPolicy(), new double[] {.25, .75});
    var decision =
        new CfrDecision(
            1,
            0,
            new int[] {3, 4},
            0,
            prediction.targetPolicy(),
            .25,
            prediction.observationPayload());
    var episode = new CfrEpisode(7, List.of(decision), new double[] {1, -1, 0, 0});
    var row = LcfrTargets.generate(episode, 1, 1).getFirst();
    Assert.assertSame(decision.observation(), prediction.observationPayload());
    Assert.assertSame(row.observation(), decision.observation());
    Assert.assertEquals(row.observation().toByteArray(), new byte[] {4, 5, 6});
  }

  @Test
  public void serializationKeepsTheExistingLengthPrefixedFormatAndChecks() throws IOException {
    for (int length : new int[] {0, 3, 1_000_000}) {
      byte[] source = new byte[length];
      for (int index = 0; index < length; index++) source[index] = (byte) index;
      var payload = ObservationPayload.copyOf(source);
      var expected = new ByteArrayOutputStream();
      try (var out = new DataOutputStream(expected)) {
        out.writeInt(length);
        out.write(source);
      }
      var actual = new ByteArrayOutputStream();
      try (var out = new DataOutputStream(actual)) {
        payload.write(out);
      }
      Assert.assertEquals(actual.toByteArray(), expected.toByteArray());
      byte[] encoded = actual.toByteArray();
      ObservationPayload read;
      try (var in = input(encoded)) {
        read = ObservationPayload.read(in);
      }
      Assert.assertEquals(read, payload);
      Assert.assertNotSame(read, payload);
      java.util.Arrays.fill(encoded, (byte) 0);
      Assert.assertEquals(read, payload);
    }
    for (int length : new int[] {-1, CfrDataChecks.MAX_OBSERVATION_BYTES + 1, Integer.MAX_VALUE}) {
      byte[] encoded = ByteBuffer.allocate(4).putInt(length).array();
      Assert.expectThrows(IOException.class, () -> ObservationPayload.read(input(encoded)));
    }
    Assert.expectThrows(
        EOFException.class, () -> ObservationPayload.read(input(new byte[] {0, 0, 0, 2, 7})));
    Assert.expectThrows(
        IllegalArgumentException.class, () -> ObservationPayload.copyOf((byte[]) null));
    Assert.expectThrows(
        IllegalArgumentException.class, () -> ObservationPayload.copyOf((ByteBuffer) null));
    Assert.expectThrows(
        IllegalArgumentException.class,
        () -> ObservationPayload.copyOf(new byte[CfrDataChecks.MAX_OBSERVATION_BYTES + 1]));
  }

  private static DataInputStream input(byte[] bytes) {
    return new DataInputStream(new ByteArrayInputStream(bytes));
  }
}
