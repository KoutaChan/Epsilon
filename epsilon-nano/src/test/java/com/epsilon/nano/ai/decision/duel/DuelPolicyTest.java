package com.epsilon.nano.ai.decision.duel;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.arena.DecisionBatchEncoder;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionBoundaryContext;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionEvaluatorFactory;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionGreedyEvaluator;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.runtime.DecisionInferenceIngress;
import com.epsilon.spi.DecisionRequest;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletionException;
import java.util.concurrent.TimeUnit;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 共通Duelから借りた公開観測が従来のエンジン入力と一致し、予約を一度だけ解放することを検証する。 */
public class DuelPolicyTest {
  @Test
  public void observedInputsMatchEngineEncodingAcrossSuccessiveBoundaries() throws Exception {
    GameEngine engine = new GameEngine(91826L);
    GameStepResult step = engine.stepHanchan();
    CheckingEvaluator evaluator = new CheckingEvaluator();
    try (var encoder = new DecisionBatchEncoder(2);
        var policy = new DuelPolicy(evaluator, encoder)) {
      int compared = 0;
      for (int boundary = 0; boundary < 64; boundary++) {
        while (step instanceof GameStepResult.RoundSettled) step = engine.stepHanchan();
        if (step instanceof GameStepResult.HanchanEnded) break;
        var awaiting = (GameStepResult.AwaitingDecisions) step;
        var selections = new EngineSelectionBuffer();
        for (EngineDecisionPoint point : awaiting.decisions()) {
          if (point.legalActions().size() > 1) {
            DecisionRequest request = request(engine, point);
            var bucket = DecisionBatchBuilder.selectInferenceBucket(point.legalActions());
            var builder = DecisionBatchBuilder.inference(1, bucket);
            builder.addInferenceRow(engine, point, DecisionBoundaryContext.uniform());
            evaluator.expected = builder.build();
            try (var ingress = policy.tryAcquire(policy.batchKey(request), () -> {})) {
              Assert.assertEquals(
                  ingress.submit(List.of(request)).get(10, TimeUnit.SECONDS), new int[] {0});
            }
            compared++;
          }
          selections.add(point.id(), point.legalActions().getFirst());
        }
        step = engine.commitDecisions(selections);
      }
      Assert.assertTrue(compared > 10);
      Assert.assertEquals(evaluator.released, compared);
    }
    Assert.assertFalse(evaluator.closed, "DuelSession owns the evaluator");
  }

  @Test
  public void unusedAndFailedEncodingReservationsAreReleasedOnce() {
    var engine = new GameEngine(91826L);
    var point = ((GameStepResult.AwaitingDecisions) engine.stepHanchan()).decisions().getFirst();
    var valid = request(engine, point);
    CheckingEvaluator evaluator = new CheckingEvaluator();
    try (var encoder = new DecisionBatchEncoder(1);
        var policy = new DuelPolicy(evaluator, encoder)) {
      var unused = policy.tryAcquire(policy.batchKey(valid), () -> {});
      unused.close();
      unused.close();
      Assert.assertEquals(evaluator.released, 1);
      var invalid =
          new DecisionRequest(
              0L, point.id(), point.player(), point.kind(), null, point.legalActions());
      try (var ingress = policy.tryAcquire(policy.batchKey(valid), () -> {})) {
        Assert.expectThrows(
            CompletionException.class, () -> ingress.submit(List.of(invalid)).join());
      }
      Assert.assertEquals(evaluator.released, 2);
      Assert.assertFalse(evaluator.closed);
    }
  }

  @Test(groups = "native")
  public void checkpointEvaluatorKeepsGreedyActionsThroughCommonPolicy() throws Exception {
    Path temporary = Files.createTempDirectory("duel-policy-");
    try {
      var config =
          EpsilonSettings.load(
              null,
              Map.of(
                  "epsilon.devices.learner", "cpu",
                  "epsilon.devices.inference", "cpu",
                  "epsilon.decision.inference.maxBatch", "4",
                  "epsilon.decision.inference.multiTransitionMaxBatch", "4"));
      try (var model =
          NetworkFactory.createDecisionModel(Device.cpu(), false, 32, EpsilonUtilityProfile.TOP)) {
        EpsilonDecisionCheckpointManager.saveInitial(model, temporary.resolve("checkpoint"));
      }
      GameEngine engine = new GameEngine(91826L);
      var point = ((GameStepResult.AwaitingDecisions) engine.stepHanchan()).decisions().getFirst();
      DecisionRequest request = request(engine, point);
      try (var execution = new DecisionExecutionContext();
          var handle =
              EpsilonDecisionEvaluatorFactory.openGreedyPolicyCheckpointEvaluator(
                  temporary.resolve("checkpoint"), 4, execution, config);
          var encoder = new DecisionBatchEncoder(2);
          var policy = new DuelPolicy(handle.evaluator(), encoder)) {
        var builder =
            DecisionBatchBuilder.inference(
                1, DecisionBatchBuilder.selectInferenceBucket(point.legalActions()));
        builder.addInferenceRow(engine, point, DecisionBoundaryContext.uniform());
        int[] expected = handle.evaluator().evaluateGreedyActionSlots(builder.build());
        try (var ingress = policy.tryAcquire(policy.batchKey(request), () -> {})) {
          Assert.assertNotNull(ingress);
          Assert.assertEquals(ingress.submit(List.of(request)).get(30, TimeUnit.SECONDS), expected);
        }
      }
    } finally {
      try (var paths = Files.walk(temporary)) {
        for (var path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
      }
    }
  }

  private static DecisionRequest request(GameEngine engine, EngineDecisionPoint point) {
    return new DecisionRequest(
        0L,
        point.id(),
        point.player(),
        point.kind(),
        engine.observation(point.player()),
        point.legalActions());
  }

  private static final class CheckingEvaluator implements EpsilonDecisionGreedyEvaluator {
    DecisionHostBatch expected;
    int released;
    boolean closed;

    public DecisionInferenceIngress.Attempt tryAcquireInferenceIngress() {
      return DecisionInferenceIngress.Attempt.acquired(
          new DecisionInferenceIngress(this, null, () -> released++));
    }

    public int[] evaluateGreedyActionSlots(DecisionHostBatch actual) {
      var a = actual.sliceRows(0, actual.size());
      var e = expected.sliceRows(0, expected.size());
      ShortBuffer ac = ShortBuffer.allocate(a.inputCategoricalElementCount());
      ShortBuffer ec = ShortBuffer.allocate(e.inputCategoricalElementCount());
      FloatBuffer an = FloatBuffer.allocate(a.inputNumericElementCount());
      FloatBuffer en = FloatBuffer.allocate(e.inputNumericElementCount());
      a.copyInputCategoriesTo(ac);
      e.copyInputCategoriesTo(ec);
      a.copyInputNumericsTo(an);
      e.copyInputNumericsTo(en);
      Assert.assertEquals(ac.array(), ec.array());
      Assert.assertEquals(an.array(), en.array());
      return new int[actual.size()];
    }

    public List<EpsilonDecisionInferenceServer.Prediction> evaluateBatch(DecisionHostBatch batch) {
      throw new AssertionError("Only greedy action slots may be requested");
    }

    public void close() {
      closed = true;
    }
  }
}
