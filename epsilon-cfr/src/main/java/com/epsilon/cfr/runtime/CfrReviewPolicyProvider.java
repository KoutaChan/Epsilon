package com.epsilon.cfr.runtime;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.spi.ReviewPolicy;
import com.epsilon.spi.ReviewPolicyProvider;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/** Reviewerの盤面と合法手を、そのまま配布された平均方策へ渡す。 */
public final class CfrReviewPolicyProvider implements ReviewPolicyProvider {
  @Override
  public String seriesId() {
    return "epsilon-cfr";
  }

  @Override
  public ReviewPolicy openReview(Path checkpoint, Map<String, String> options) {
    CfrDeployment deployment = CfrDeployment.open(checkpoint, options);
    return new ReviewPolicy() {
      @Override
      public float[] probabilities(PublicObservation observation, List<Action> legal) {
        var context = deployment.context(0, 0, observation, legal);
        double[] probabilities =
            deployment.model.evaluate(List.of(context)).getFirst().targetPolicy();
        float[] result = new float[probabilities.length];
        for (int i = 0; i < result.length; i++) result[i] = (float) probabilities[i];
        return result;
      }

      @Override
      public void close() {
        deployment.close();
      }
    };
  }
}
