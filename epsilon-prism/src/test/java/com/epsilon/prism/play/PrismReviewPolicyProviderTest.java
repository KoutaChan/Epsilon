package com.epsilon.prism.play;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.core.Action;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.prism.backend.HipFixtures;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.core.Stage;
import com.epsilon.prism.core.Threats;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.policy.PolicyRequest;
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

/** 牌譜解析の接続: 観測だけから求めた種別・類が収集と同じになり、網の版を開いて実際の判断を評価できる（CPU）。 */
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
                  Threats.count(state, point.player()),
                  state.getKyokuIndex());
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

  /** 網の版を開いて実際の判断を評価する（CPU は FP32 の eager）。重みを動かした網で、手の値が 0 でないことも見る。 */
  @Test(timeOut = 300_000)
  public void snapshotEvaluatesRealDecisions() throws Exception {
    Path root = Files.createTempDirectory("prism-review");
    try {
      Path snapshot = root.resolve("1");
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        Snapshot.write(snapshot, HipFixtures.network(manager, 2, 17), PolicyParams.of(1));
      }
      Evaluated evaluated = evaluateRealDecisions(snapshot);
      Assert.assertTrue(evaluated.decisions() >= 8);
      Assert.assertTrue(evaluated.maxAbsValue() > 1e-3f, "values must be non-trivial: " + evaluated);
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  /** 評価した判断（手が 2 つ以上）の数と、手の値の絶対値の最大。 */
  private record Evaluated(int decisions, float maxAbsValue) {}

  /** 版を牌譜解析の口で開き、エンジンの半荘を最善手で進めながら 8 つ以上の判断を評価する（確率の和・最善手の損 0 を見る）。 */
  private static Evaluated evaluateRealDecisions(Path snapshot) {
    var provider = new PrismReviewPolicyProvider();
    Assert.assertEquals(provider.seriesId(), "epsilon-prism");
    int evaluated = 0;
    float maxAbs = 0;
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
            maxAbs = Math.max(maxAbs, Math.abs(evaluation.values()[a]));
          }
          Assert.assertEquals(sum, 1, ReviewEvaluation.SUM_TOLERANCE);
          Assert.assertEquals(evaluation.losses()[best], 0f);
          if (legal.size() > 1) evaluated++;
          selections.add(point.id(), legal.get(best));
        }
        step = engine.commitDecisions(selections);
      }
    }
    return new Evaluated(evaluated, maxAbs);
  }
}
