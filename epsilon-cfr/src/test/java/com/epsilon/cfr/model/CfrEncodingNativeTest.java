package com.epsilon.cfr.model;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.collection.Prediction;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.nano.ai.decision.input.DecisionInferenceStoragePool;
import com.epsilon.spi.PolicyExecutionContext;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletionException;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 並列に構築した盤面と保存payloadの所有権・順序を検証する。 */
public class CfrEncodingNativeTest {
  @Test(groups = "native")
  public void parallelEncodingAboveTheThresholdPreservesEveryPayloadAndRowOrder() {
    var bases = List.of(CfrModelTest.context(873), CfrModelTest.context(874));
    var contexts = new ArrayList<DecisionContext>();
    for (int index = 0; index < 192; index++) {
      var base = bases.get(index % bases.size());
      int actions =
          switch (index % 3) {
            case 0 -> 1;
            case 1 -> 2;
            default -> base.legalActions().size();
          };
      contexts.add(context(base, 1000 + index, actions));
    }
    try (var model =
            new CfrModel(
                new CfrModelConfig(
                    16,
                    32,
                    .001f,
                    0,
                    0,
                    918,
                    32,
                    DecisionComputePrecision.FLOAT32,
                    DecisionTensorTransfer.DIRECT_BUFFER,
                    EpsilonUtilityProfile.TENHOU),
                CfrModel.Role.AVERAGE,
                Device.cpu());
        var execution = new PolicyExecutionContext(4)) {
      var pending = model.encodeAsync(contexts, execution);
      // 別バッチと同時に作業領域を借用しても、構築済みの入力は独立して保持される。
      var reversed = new ArrayList<>(contexts);
      Collections.reverse(reversed);
      var nextPending = model.encodeAsync(reversed, execution);
      var encoded = pending.join();
      var nextEncoded = nextPending.join();
      Assert.assertEquals(encoded.size(), contexts.size());
      var expected = contexts.stream().map(CfrObservationCodec::encodePayload).toList();
      Assert.assertEquals(encoded.encodePayloadsAsync(execution).join(), expected);
      Assert.assertEquals(
          model.encode(contexts, execution).encodePayloadsAsync(execution).join(), expected);
      var predictions = model.evaluateEncoded(encoded).withObservations(expected);
      var decodedPredictions = model.evaluateObservations(expected);
      var nextPredictions =
          model
              .evaluateEncoded(nextEncoded)
              .withObservations(nextEncoded.encodePayloadsAsync(execution).join());
      for (int index = 0; index < contexts.size(); index++) {
        Assert.assertEquals(
            predictions.get(index).observationPayload(),
            CfrObservationCodec.encodePayload(contexts.get(index)),
            "Parallel encoder changed payload or ordering at row " + index);
        Assert.assertEquals(
            predictions.get(index).targetPolicy().length,
            contexts.get(index).legalActions().size());
        Assert.assertEquals(
            predictions.get(index).targetPolicy(), decodedPredictions.get(index).targetPolicy());
        Assert.assertEquals(
            nextPredictions.get(index).observationPayload(),
            CfrObservationCodec.encodePayload(reversed.get(index)));
      }
      Assert.assertEquals(model.encodeAsync(List.of(), execution).join().size(), 0);
      Assert.assertTrue(model.evaluate(List.of()).isEmpty());
      Assert.assertTrue(model.evaluateObservations(List.of()).isEmpty());

      // 借用経路を繰り返しても、公開APIから得た所有入力や以前の保存データは変更されない。
      try (var pool = new DecisionInferenceStoragePool(16L << 20, 8)) {
        for (var selected : List.of(contexts, reversed.subList(0, 3), reversed, contexts)) {
          List<Prediction> pooledPredictions;
          try (var storage = pool.acquire()) {
            var pooled = model.encodeAsync(selected, execution, storage).join();
            var saved = pooled.encodePayloadsAsync(execution);
            pooledPredictions = model.evaluateEncoded(pooled).withObservations(saved.join());
            Assert.assertTrue(pool.snapshot().borrowedBuffers() > 0);
          }
          Assert.assertEquals(pool.snapshot().borrowedBuffers(), 0);
          Assert.assertTrue(pool.snapshot().retainedBytes() <= 16L << 20);
          var wanted = model.evaluate(selected);
          for (int row = 0; row < selected.size(); row++) {
            Assert.assertEquals(
                pooledPredictions.get(row).observationPayload(),
                wanted.get(row).observationPayload());
            Assert.assertEquals(
                pooledPredictions.get(row).targetPolicy(), wanted.get(row).targetPolicy());
          }
        }
        var invalid = new ArrayList<>(contexts);
        var base = invalid.get(64);
        invalid.set(
            64,
            new DecisionContext(
                base.episodeId(), base.decisionId(), base.seat(), null, base.legalActions()));
        try (var storage = pool.acquire()) {
          Assert.expectThrows(
              CompletionException.class,
              () -> model.encodeAsync(invalid, execution, storage).join());
        }
        Assert.assertEquals(pool.snapshot().borrowedBuffers(), 0);
      }
      Assert.assertEquals(encoded.encodePayloadsAsync(execution).join(), expected);
      Assert.assertEquals(predictions.getFirst().observationPayload(), expected.getFirst());
    }
  }

  private static DecisionContext context(DecisionContext base, long episode, int actions) {
    return new DecisionContext(
        episode,
        base.decisionId(),
        base.seat(),
        base.observation(),
        base.legalActions().subList(0, actions));
  }
}
