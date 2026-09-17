package com.epsilon.nano.ai.decision.runtime;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.nano.config.settings.DecisionInferenceSettings;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.spi.ReviewPolicy;
import com.epsilon.spi.ReviewPolicyProvider;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** 系列固有のエンコーダーと凍結モデルから全合法候補の確率を返す。 */
public final class NanoReviewPolicyProvider implements ReviewPolicyProvider {
  @Override
  public String seriesId() {
    return "epsilon-nano";
  }

  @Override
  public ReviewPolicy openReview(Path checkpoint, Map<String, String> options) {
    Map<String, String> values = new LinkedHashMap<>(options);
    String settingPath = values.remove("settings");
    String deviceText = values.remove("device");
    String devices = values.remove("devices");
    if (deviceText == null) deviceText = devices;
    Device device = parseDevice(deviceText == null ? "cpu" : deviceText);
    try {
      SettingsLoader settings =
          EpsilonSettings.load(settingPath == null ? null : Path.of(settingPath), values);
      var inference = settings.bind(DecisionInferenceSettings.class);
      Model model =
          EpsilonDecisionCheckpointManager.loadForInference(checkpoint, device, inference);
      try {
        var server = EpsilonDecisionInferenceServer.forFrozenModel(model, 1);
        return new ReviewPolicy() {
          @Override
          public float[] probabilities(PublicObservation observation, List<Action> legalActions) {
            var builder =
                DecisionBatchBuilder.inference(
                    1, DecisionBatchBuilder.selectInferenceBucket(legalActions));
            builder.encodeObservedInferenceRow(0, observation, legalActions);
            return server
                .evaluateBatch(builder.buildEncodedInferenceRows())
                .getFirst()
                .policyProbabilities();
          }

          @Override
          public void close() {
            try {
              server.close();
            } finally {
              model.close();
            }
          }
        };
      } catch (RuntimeException | Error failure) {
        model.close();
        throw failure;
      }
    } catch (IOException failure) {
      throw new UncheckedIOException("Could not load the review model: " + checkpoint, failure);
    }
  }

  private static Device parseDevice(String value) {
    if ("cpu".equalsIgnoreCase(value)) return Device.cpu();
    if ("gpu".equalsIgnoreCase(value)) return Device.gpu(0);
    if (value.startsWith("gpu:")) return Device.gpu(Integer.parseInt(value.substring(4)));
    throw new IllegalArgumentException("Review device must be cpu or gpu:N.");
  }
}
