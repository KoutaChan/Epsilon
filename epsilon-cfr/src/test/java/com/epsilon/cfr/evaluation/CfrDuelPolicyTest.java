package com.epsilon.cfr.evaluation;

import com.epsilon.cfr.collection.CfrPolicy;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.collection.Prediction;
import com.epsilon.cfr.data.ObservationPayload;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.spi.DecisionRequest;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.Assert;
import org.testng.annotations.Test;

public class CfrDuelPolicyTest {
  @Test
  public void maximumProbabilitySelectionUsesOnlyTheBoardAndLegalActions() {
    DecisionRequest request = request();
    int count = request.legalActions().size();
    Assert.assertTrue(count >= 2);
    CfrPolicy source =
        rows -> {
          Assert.assertEquals(rows.size(), 2);
          for (DecisionContext context : rows) {
            Assert.assertEquals(context.episodeId(), 17L);
            Assert.assertEquals(context.decisionId(), request.id());
            Assert.assertSame(context.observation(), request.observation());
            Assert.assertEquals(context.legalActions(), request.legalActions());
          }
          double[] first = new double[count];
          Arrays.fill(first, .25 / (count - 1));
          first[count - 1] = .75;
          double[] tied = new double[count];
          tied[0] = tied[1] = .5;
          return List.of(
              new Prediction(first, ObservationPayload.copyOf(new byte[0])),
              new Prediction(tied, ObservationPayload.copyOf(new byte[0])));
        };
    try (var policy = new CfrDuelPolicy(source, 2);
        var ingress = policy.tryAcquire(policy.batchKey(request), () -> {})) {
      Assert.assertEquals(
          ingress.submit(List.of(request, request)).join(), new int[] {count - 1, 0});
    }
  }

  @Test
  public void oversizedBatchReleasesTheReservedCapacity() {
    AtomicInteger releases = new AtomicInteger();
    CfrPolicy source =
        new CfrPolicy() {
          @Override
          public List<Prediction> evaluate(List<DecisionContext> batch) {
            throw new AssertionError("Oversized batch must fail before inference");
          }

          @Override
          public DecisionInferenceIngress.Attempt tryAcquireInferenceIngress() {
            return DecisionInferenceIngress.Attempt.acquired(
                new DecisionInferenceIngress(this, null, releases::incrementAndGet));
          }
        };
    DecisionRequest request = request();
    try (var policy = new CfrDuelPolicy(source, 2);
        var ingress = policy.tryAcquire(policy.batchKey(request), () -> {})) {
      Assert.expectThrows(
          IllegalArgumentException.class, () -> ingress.submit(List.of(request, request, request)));
    }
    Assert.assertEquals(releases.get(), 1);
  }

  private static DecisionRequest request() {
    var engine = new GameEngine(321);
    var point = ((GameStepResult.AwaitingDecisions) engine.stepHanchan()).decisions().getFirst();
    return new DecisionRequest(
        17,
        point.id(),
        point.player(),
        point.kind(),
        engine.observation(point.player()),
        point.legalActions());
  }
}
