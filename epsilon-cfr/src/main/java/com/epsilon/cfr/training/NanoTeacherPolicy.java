package com.epsilon.cfr.training;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.cfr.collection.CfrPolicy;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.collection.Prediction;
import com.epsilon.cfr.model.CfrObservationCodec;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** 初期蒸留用。Nanoの条件付き方策を、同じ順序の全合法手確率として取得する。 */
final class NanoTeacherPolicy implements CfrPolicy, AutoCloseable {
  private final Model model;
  private final EpsilonDecisionInferenceServer server;
  private final int maxBatch;

  NanoTeacherPolicy(Path checkpoint, Device device, int maxBatch) throws IOException {
    if (maxBatch < 1) throw new IllegalArgumentException("Positive teacher batch size required");
    this.maxBatch = maxBatch;
    model = EpsilonDecisionCheckpointManager.loadForInference(checkpoint, device);
    try {
      server = EpsilonDecisionInferenceServer.forFrozenModel(model, maxBatch);
    } catch (RuntimeException | Error failure) {
      model.close();
      throw failure;
    }
  }

  @Override
  public List<Prediction> evaluate(List<DecisionContext> batch) {
    Map<DecisionBucket, List<Integer>> groups = new LinkedHashMap<>();
    for (int index = 0; index < batch.size(); index++)
      groups
          .computeIfAbsent(
              DecisionBatchBuilder.selectInferenceBucket(batch.get(index).legalActions()),
              ignored -> new ArrayList<>())
          .add(index);
    Prediction[] result = new Prediction[batch.size()];
    for (var entry : groups.entrySet()) {
      var indices = entry.getValue();
      for (int from = 0; from < indices.size(); from += maxBatch) {
        List<Integer> part = indices.subList(from, Math.min(indices.size(), from + maxBatch));
        var builder = DecisionBatchBuilder.inference(part.size(), entry.getKey());
        for (int row = 0; row < part.size(); row++) {
          var context = batch.get(part.get(row));
          builder.encodeObservedInferenceRow(row, context.observation(), context.legalActions());
        }
        var inputs = builder.buildEncodedInferenceRows();
        var predictions = server.evaluateBatch(inputs);
        for (int row = 0; row < part.size(); row++) {
          int index = part.get(row);
          var context = batch.get(index);
          float[] values = predictions.get(row).policyProbabilities();
          if (values.length != context.legalActions().size())
            throw new IllegalStateException("Nano teacher legal-action shape mismatch");
          double[] policy = new double[values.length];
          double sum = 0;
          for (int action = 0; action < values.length; action++) {
            if (!Float.isFinite(values[action]) || values[action] < 0)
              throw new IllegalStateException("Invalid Nano teacher probability");
            policy[action] = values[action];
            sum += values[action];
          }
          if (!(sum > 0)) throw new IllegalStateException("Empty Nano teacher distribution");
          for (int action = 0; action < policy.length; action++) policy[action] /= sum;
          result[index] = new Prediction(policy, CfrObservationCodec.encodePayload(inputs, row));
        }
      }
    }
    return List.of(result);
  }

  @Override
  public void close() {
    try {
      server.close();
    } finally {
      model.close();
    }
  }
}
