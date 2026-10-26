package com.epsilon.prism.play;

import ai.djl.Device;
import com.epsilon.core.Action;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.core.Stage;
import com.epsilon.prism.core.Threats;
import com.epsilon.prism.model.PrismModel;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.spi.ReviewEvaluation;
import com.epsilon.spi.ReviewPolicy;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 牌譜解析の接続: 観測だけから求めた種別・類が収集と同じになり、白紙の版を開いて実際の判断を評価できる（CPU）。 */
public class PrismReviewPolicyProviderTest {
  @Test
  public void observationClassificationMatchesCollection() {
    int postCalls = 0;
    for (long game = 0; game < 4; game++) {
      var engine = new GameEngine(31 + game);
      var random = new SplittableRandom(7 + game);
      var selections = new EngineSelectionBuffer();
      GameStepResult step = engine.stepHanchan();
      while (!(step instanceof GameStepResult.HanchanEnded)) {
        if (!(step instanceof GameStepResult.AwaitingDecisions awaiting)) {
          step = engine.stepHanchan();
          continue;
        }
        selections.clear();
        var state = engine.getState();
        for (EngineDecisionPoint point : awaiting.decisions()) {
          List<Action> legal = point.legalActions();
          // 収集（GameRunner#prepare）の分類
          Kind kind = Kind.classify(point.kind(), legal);
          DecisionClass decisionClass =
              DecisionClass.of(
                  kind,
                  Stage.of(state.remainingWallTiles()),
                  Threats.count(state, point.player()));
          PolicyRequest request =
              SnapshotScorer.request(engine.observation(point.player()), legal);
          Assert.assertEquals(request.kind(), kind, point.toString());
          Assert.assertEquals(request.decisionClass(), decisionClass, point.toString());
          if (kind == Kind.POSTCALL) postCalls++;
          // 鳴きを多くして、鳴いた直後の打牌と脅威のもとの判断を十分に通す
          Action chosen = legal.get(random.nextInt(legal.size()));
          for (Action action : legal)
            if (action.type() == Action.Type.CHI || action.type() == Action.Type.PON)
              chosen = action;
          selections.add(point.id(), chosen);
        }
        step = engine.commitDecisions(selections);
      }
    }
    Assert.assertTrue(postCalls > 20, "post-call discards " + postCalls);
  }

  @Test(timeOut = 300_000)
  public void blankSnapshotEvaluatesRealDecisions() throws Exception {
    Path root = Files.createTempDirectory("prism-review");
    try {
      Path snapshot = root.resolve("1");
      PrismConfig config = PrismConfig.fromSettings(EpsilonSettings.defaults());
      try (var model =
          new PrismModel(
              config.modelConfig(Device.cpu(), 16), "m0", NetworkDevices.of(Device.cpu()))) {
        Snapshot.write(snapshot, model, PolicyParams.of(1));
      }
      var provider = new PrismReviewPolicyProvider();
      Assert.assertEquals(provider.seriesId(), "epsilon-prism");
      int evaluated = 0;
      try (ReviewPolicy policy = provider.openReview(snapshot, Map.of())) {
        var engine = new GameEngine(11);
        var selections = new EngineSelectionBuffer();
        GameStepResult step = engine.stepHanchan();
        while (evaluated < 8 && !(step instanceof GameStepResult.HanchanEnded)) {
          if (!(step instanceof GameStepResult.AwaitingDecisions awaiting)) {
            step = engine.stepHanchan();
            continue;
          }
          selections.clear();
          for (EngineDecisionPoint point : awaiting.decisions()) {
            List<Action> legal = point.legalActions();
            ReviewEvaluation evaluation =
                policy.evaluate(engine.observation(point.player()), legal);
            Assert.assertTrue(evaluation.valued());
            Assert.assertEquals(evaluation.actions(), legal.size());
            Assert.assertEquals(evaluation.values().length, legal.size());
            double sum = 0;
            int best = 0;
            for (int a = 0; a < legal.size(); a++) {
              sum += evaluation.probabilities()[a];
              if (evaluation.values()[a] > evaluation.values()[best]) best = a;
            }
            Assert.assertEquals(sum, 1, ReviewEvaluation.SUM_TOLERANCE);
            Assert.assertEquals(evaluation.losses()[best], 0f);
            if (legal.size() > 1) evaluated++;
            selections.add(point.id(), legal.get(best));
          }
          step = engine.commitDecisions(selections);
        }
      }
      Assert.assertTrue(evaluated >= 8, "evaluated " + evaluated);
    } finally {
      Snapshot.deleteTree(root);
    }
  }
}
