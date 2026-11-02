package com.epsilon.prism.play;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.spi.ReviewEvaluation;
import com.epsilon.spi.ReviewPolicy;
import com.epsilon.spi.ReviewPolicyProvider;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * epsilon-prism の版を牌譜解析に開く。手の価値は s（{@link SnapshotScorer}）、表示用の確率は softmax(s/τ)（最善手は貪欲と同じ）。
 *
 * <p>options: {@code settings} は系列の TOML、{@code device}（か {@code devices}）は epsilon.devices.inference を差し替え（既定 cpu、
 * {@code gpu} は {@code gpu:0}）、{@code temperature} は τ（既定 {@link ReviewEvaluation#DISPLAY_TEMPERATURE}）、残りは epsilon.* の上書き。
 */
public final class PrismReviewPolicyProvider implements ReviewPolicyProvider {
  @Override
  public String seriesId() {
    return "epsilon-prism";
  }

  @Override
  public ReviewPolicy openReview(Path checkpoint, Map<String, String> options) {
    Map<String, String> values = new LinkedHashMap<>(options);
    String temperatureValue = values.remove("temperature");
    double temperature =
        temperatureValue == null
            ? ReviewEvaluation.DISPLAY_TEMPERATURE
            : Double.parseDouble(temperatureValue.strip());
    if (!(temperature > 0)) throw new IllegalArgumentException("temperature must be positive");
    String settings = values.remove("settings");
    String device = values.remove("device");
    String devices = values.remove("devices");
    if (device == null) device = devices;
    if (device == null) device = "cpu";
    values.put(
        "epsilon.devices.inference", device.strip().equalsIgnoreCase("gpu") ? "gpu:0" : device);
    SnapshotScorer scorer;
    try {
      scorer =
          SnapshotScorer.open(
              checkpoint,
              EpsilonSettings.load(settings == null ? null : Path.of(settings), values));
    } catch (IOException failure) {
      throw new UncheckedIOException("Could not load the review model: " + checkpoint, failure);
    }
    return new ReviewPolicy() {
      @Override
      public ReviewEvaluation evaluate(PublicObservation observation, List<Action> legalActions) {
        float[] scores = scorer.scores(observation, legalActions);
        return ReviewEvaluation.valued(scores, ReviewEvaluation.softmax(scores, temperature));
      }

      @Override
      public void close() {
        scorer.close();
      }
    };
  }
}
