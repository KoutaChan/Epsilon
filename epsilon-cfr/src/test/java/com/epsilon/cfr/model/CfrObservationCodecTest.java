package com.epsilon.cfr.model;

import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.data.ObservationPayload;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.data.DecisionInputRowCodec;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionBinaryArrayCodec;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.BufferOverflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CfrObservationCodecTest {
  @Test
  public void payloadContainsExactlyTheNanoBoardInput() throws IOException {
    var context = CfrModelTest.context(772);
    var row = CfrObservationCodec.build(context);
    var expected = new ByteArrayOutputStream();
    try (var output = new DataOutputStream(expected)) {
      output.writeInt(0x43465233);
      DecisionInputRowCodec.write(
          output, row.host().sliceRows(0, 1), new EpsilonDecisionBinaryArrayCodec.Scratch());
    }
    Assert.assertEquals(
        CfrObservationCodec.encodePayload(row).toByteArray(), expected.toByteArray());
    var metadata =
        new DecisionContext(
            99, 100, context.seat(), context.observation(), context.legalActions(), 37);
    Assert.assertEquals(
        CfrObservationCodec.encodePayload(metadata).toByteArray(),
        expected.toByteArray(),
        "Episode and policy-selection metadata must not enter the model input");
  }

  @Test
  public void oldSchemasTruncationAndTrailingDataAreRejected() {
    byte[] encoded = CfrObservationCodec.encodePayload(CfrModelTest.context(772)).toByteArray();
    byte[] old = encoded.clone();
    ByteBuffer.wrap(old).putInt(0x43465232);
    Assert.expectThrows(
        UncheckedIOException.class,
        () -> CfrObservationCodec.decode(ObservationPayload.copyOf(old)));
    for (int length : new int[] {0, 3, encoded.length - 1, encoded.length + 1}) {
      byte[] malformed = Arrays.copyOf(encoded, length);
      Assert.expectThrows(
          UncheckedIOException.class,
          () -> CfrObservationCodec.decode(ObservationPayload.copyOf(malformed)));
    }
  }

  @Test
  public void immutablePayloadSurvivesScratchReuseAndExternalArrayMutation() {
    var original = CfrObservationCodec.build(CfrModelTest.context(772));
    byte[] expected = CfrObservationCodec.encodePayload(original).toByteArray();
    var payload = CfrObservationCodec.encodePayload(original);
    var decoded = CfrObservationCodec.decode(payload);
    CfrObservationCodec.encodePayload(CfrModelTest.context(773));
    Arrays.fill(payload.toByteArray(), (byte) 0);
    Assert.assertEquals(payload.toByteArray(), expected);
    Assert.assertEquals(CfrObservationCodec.encodePayload(decoded).toByteArray(), expected);
    Assert.assertEquals(
        CfrObservationCodec.encodePayload(CfrObservationCodec.decode(payload)).toByteArray(),
        expected);
    var copied = CfrObservationCodec.decode(ObservationPayload.copyOf(expected));
    Arrays.fill(expected, (byte) 0);
    Assert.assertEquals(CfrObservationCodec.encodePayload(copied), payload);
  }

  @Test
  public void compactRowsAndEncodedBytesSurviveWorkspaceAndEngineReuse() throws IOException {
    var workspace = new DecisionBatchBuilder.EncodingWorkspacePool(1);
    var retained = new ArrayList<CfrObservationCodec.Decoded>();
    var expected = new ArrayList<byte[]>();
    var encoded = new ArrayList<byte[]>();
    var engine = new GameEngine(770);
    GameStepResult step = engine.stepHanchan();
    int multiTransitionRows = 0, singleTransitionRows = 0;
    for (int boundary = 0;
        boundary < 160 && !(step instanceof GameStepResult.HanchanEnded);
        boundary++) {
      if (step instanceof GameStepResult.RoundSettled) {
        step = engine.stepHanchan();
        continue;
      }
      var selections = new EngineSelectionBuffer();
      for (var point : ((GameStepResult.AwaitingDecisions) step).decisions()) {
        var context =
            new DecisionContext(
                770,
                point.id(),
                point.player(),
                engine.observation(point.player()),
                point.legalActions());
        var row = CfrObservationCodec.build(context, workspace);
        var referenceBuffer = new ByteArrayOutputStream();
        try (var output = new DataOutputStream(referenceBuffer)) {
          output.writeInt(0x43465233);
          DecisionInputRowCodec.write(
              output,
              row.host().copyRow(0).sliceRows(0, 1),
              new EpsilonDecisionBinaryArrayCodec.Scratch());
        }
        byte[] reference = referenceBuffer.toByteArray();
        byte[] actual = CfrObservationCodec.encodePayload(row).toByteArray();
        Assert.assertEquals(actual, reference);
        var grouped = DecisionBatchBuilder.inference(3, row.host().bucket(), workspace);
        try (var session = grouped.openEncoding()) {
          for (int index = 0; index < 3; index++)
            session.encodeObservedInferenceRow(
                index, context.observation(), context.legalActions());
        }
        var host = grouped.buildEncodedInferenceRows();
        for (int index = 0; index < 3; index++) {
          var payload = CfrObservationCodec.encodePayload(host, index);
          Assert.assertEquals(payload.toByteArray(), reference);
          Assert.assertEquals(
              CfrObservationCodec.encodePayload(CfrObservationCodec.decode(payload)).toByteArray(),
              reference);
        }
        retained.add(row);
        expected.add(reference);
        encoded.add(actual);
        if (row.host().bucket().actionTransitionCapacity() > 1) multiTransitionRows++;
        else singleTransitionRows++;
        selections.add(point.id(), point.legalActions().getFirst());
      }
      step = engine.commitDecisions(selections);
    }
    Assert.assertTrue(singleTransitionRows > 0);
    Assert.assertTrue(multiTransitionRows > 0);
    for (int index = 0; index < retained.size(); index++) {
      Assert.assertEquals(encoded.get(index), expected.get(index));
      Assert.assertEquals(
          CfrObservationCodec.encodePayload(retained.get(index)).toByteArray(),
          expected.get(index));
      Assert.assertEquals(
          CfrObservationCodec.encodePayload(
                  CfrObservationCodec.decode(ObservationPayload.copyOf(encoded.get(index))))
              .toByteArray(),
          expected.get(index));
    }
  }

  @Test
  public void directEncodingPreservesBufferBoundsOrderAndStreamFormat() throws IOException {
    var row = CfrObservationCodec.build(CfrModelTest.context(772)).host().sliceRows(0, 1);
    var expected = new ByteArrayOutputStream();
    try (var output = new DataOutputStream(expected)) {
      DecisionInputRowCodec.write(output, row, new EpsilonDecisionBinaryArrayCodec.Scratch());
    }
    int size = DecisionInputRowCodec.encodedSize(row);
    Assert.assertEquals(size, expected.size());
    for (ByteBuffer destination :
        List.of(ByteBuffer.allocate(size + 12), ByteBuffer.allocateDirect(size + 12))) {
      destination.order(ByteOrder.LITTLE_ENDIAN);
      for (int index = 0; index < 5; index++) destination.put(index, (byte) 91);
      for (int index = 5 + size; index < destination.capacity(); index++)
        destination.put(index, (byte) 92);
      destination.position(5).limit(5 + size);
      DecisionInputRowCodec.writeZeroed(destination, row);
      Assert.assertEquals(destination.position(), 5 + size);
      Assert.assertEquals(destination.limit(), 5 + size);
      Assert.assertSame(destination.order(), ByteOrder.LITTLE_ENDIAN);
      byte[] actual = new byte[size];
      destination.position(5).get(actual);
      Assert.assertEquals(actual, expected.toByteArray());
      destination.clear();
      for (int index = 0; index < 5; index++)
        Assert.assertEquals(destination.get(index), (byte) 91);
      for (int index = 5 + size; index < destination.capacity(); index++)
        Assert.assertEquals(destination.get(index), (byte) 92);
    }
    Assert.expectThrows(
        BufferOverflowException.class,
        () -> DecisionInputRowCodec.writeZeroed(ByteBuffer.allocate(size - 1), row));
  }

  @Test
  public void readOnlyBatchRetainsSingleRowStorageAndPreservesCombinedOrder() {
    var first = CfrObservationCodec.build(CfrModelTest.context(772));
    var second = CfrObservationCodec.build(CfrModelTest.context(773));
    Assert.assertEquals(first.host().bucket(), second.host().bucket());
    Assert.assertSame(CfrObservationCodec.batch(List.of(first)), first.host());
    var combined = CfrObservationCodec.batch(List.of(second, first));
    Assert.assertEquals(combined.size(), 2);
    Assert.assertEquals(
        CfrObservationCodec.encodePayload(new CfrObservationCodec.Decoded(combined.copyRow(0)))
            .toByteArray(),
        CfrObservationCodec.encodePayload(second).toByteArray());
    Assert.assertEquals(
        CfrObservationCodec.encodePayload(new CfrObservationCodec.Decoded(combined.copyRow(1)))
            .toByteArray(),
        CfrObservationCodec.encodePayload(first).toByteArray());
  }
}
