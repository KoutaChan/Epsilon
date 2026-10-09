package com.epsilon.cfr.evaluation;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.decision.duel.DuelEvaluation;
import com.epsilon.ai.decision.duel.EpsilonDecisionWallDuelEvaluator;
import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.config.CfrConfig;
import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import com.epsilon.cfr.model.CfrModel;
import com.epsilon.cfr.model.CfrModelConfig;
import com.epsilon.cfr.model.CfrObservationCodec;
import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.config.settings.DecisionChampionDuelSettings;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionDuelArenaSettings;
import com.epsilon.config.settings.DecisionEvalVsSettings;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 実モデルと本人の観測履歴を共通の対戦評価へ渡し、4席を入れ替えた対局の集計と判定を検証する。 */
@Test(groups = "native")
public class CfrDuelSessionTest {
  public void identicalAveragePoliciesUseTheCommonWallEvaluator() throws Exception {
    Path directory = Files.createTempDirectory("epsilon-cfr-duel-");
    Path checkpoint = directory.resolve("average");
    CfrConfig config = config(EpsilonUtilityProfile.TENHOU);
    try {
      try (CfrModel model =
          new CfrModel(
              new CfrModelConfig(
                  16,
                  16,
                  .001f,
                  0,
                  0,
                  123,
                  16,
                  DecisionComputePrecision.FLOAT32,
                  DecisionTensorTransfer.DIRECT_BUFFER,
                  EpsilonUtilityProfile.TENHOU),
              CfrModel.Role.AVERAGE,
              Device.cpu())) {
        model.saveModel(checkpoint);
      }
      try (CfrDuelSession session = CfrDuelSession.open(checkpoint, checkpoint, config)) {
        var result =
            EpsilonDecisionWallDuelEvaluator.evaluate(
                session,
                EpsilonDecisionWallDuelEvaluator.EvaluationPlan.fixed(1, .01),
                40000,
                1,
                0,
                0,
                EpsilonUtilityProfile.TENHOU,
                config.evaluation().evalVs());
        Assert.assertEquals(result.games(), 4);
        Assert.assertEquals(result.wallSeeds(), 1);
        Assert.assertEquals(
            result.decision(), EpsilonDecisionWallDuelEvaluator.Decision.UNRESOLVED);
        Assert.assertEquals(result.pairedRankDeltaMean(), 0d, 1e-12);
        Assert.assertEquals(result.pairedUtilityDeltaMean(), 0d, 1e-12);
        Assert.assertEquals(result.descriptiveResult().candidateAverageRank(), 2.5d);
        Assert.assertEquals(result.descriptiveResult().opponentAverageRank(), 2.5d);
        Assert.assertEquals(result.descriptiveResult().candidateTopRate(), .25d);
        Assert.assertEquals(result.descriptiveResult().candidateLastRate(), .25d);
      }
    } finally {
      delete(directory);
    }
  }

  public void placementUtilityFlowsIntoWallObservationsAndRejectsMismatchedModels()
      throws Exception {
    Path directory = Files.createTempDirectory("epsilon-cfr-duel-utility-");
    Path candidate = directory.resolve("candidate");
    Path opponent = directory.resolve("opponent");
    CfrConfig config = config(EpsilonUtilityProfile.PLACEMENT);
    try {
      var context = trainingContext();
      double[] candidatePolicy =
          saveModel(candidate, EpsilonUtilityProfile.PLACEMENT, 123, context, 0);
      double[] opponentPolicy =
          saveModel(
              opponent,
              EpsilonUtilityProfile.PLACEMENT,
              456,
              context,
              context.legalActions().size() - 1);
      Assert.assertTrue(
          candidatePolicy[0] > opponentPolicy[0] + .05,
          "The duel must use genuinely different learned policies");
      var walls = new ArrayList<DuelEvaluation.WallOutcome>();
      try (CfrDuelSession session = CfrDuelSession.open(candidate, opponent, config)) {
        var evaluation = session.evaluateWalls(2, 40000, 0, 4, walls::add);
        Assert.assertEquals(walls.size(), 2);
        for (var wall : walls)
          Assert.assertEquals(wall.pairedConfiguredUtilityDelta(), wall.pairedRankDelta(), 1e-12);
        double placement =
            evaluation.result()
                .utilityProfileAdvantages()[EpsilonUtilityProfile.PLACEMENT.ordinal()];
        double tenhou =
            evaluation.result().utilityProfileAdvantages()[EpsilonUtilityProfile.TENHOU.ordinal()];
        double streamed =
            walls.stream()
                .mapToDouble(DuelEvaluation.WallOutcome::pairedConfiguredUtilityDelta)
                .average()
                .orElseThrow();
        Assert.assertEquals(streamed, placement, 1e-12);
        Assert.assertTrue(
            Math.abs(placement - tenhou) > 1e-6,
            "The fixture must distinguish PLACEMENT from TENHOU");
      }
      var configMismatch =
          Assert.expectThrows(
              IOException.class,
              () -> CfrDuelSession.open(candidate, opponent, config(EpsilonUtilityProfile.TENHOU)));
      Assert.assertTrue(configMismatch.getMessage().contains("utility profile mismatch"));
      saveModel(
          opponent, EpsilonUtilityProfile.TENHOU, 456, context, context.legalActions().size() - 1);
      var opponentMismatch =
          Assert.expectThrows(
              IOException.class, () -> CfrDuelSession.open(candidate, opponent, config));
      Assert.assertTrue(opponentMismatch.getMessage().contains("utility profile mismatch"));
    } finally {
      delete(directory);
    }
  }

  private static double[] saveModel(
      Path checkpoint,
      EpsilonUtilityProfile profile,
      long seed,
      DecisionContext context,
      int preferredAction)
      throws IOException {
    try (CfrModel model =
        new CfrModel(
            new CfrModelConfig(
                16,
                16,
                .01f,
                0,
                0,
                seed,
                16,
                DecisionComputePrecision.FLOAT32,
                DecisionTensorTransfer.DIRECT_BUFFER,
                profile),
            CfrModel.Role.AVERAGE,
            Device.cpu())) {
      int[] actions =
          context.legalActions().stream().mapToInt(action -> action.toIndex()).toArray();
      double[] teacher = new double[actions.length];
      teacher[preferredAction] = 1;
      var row =
          new CfrTrainingRow(
              new CfrSampleId(1, context.episodeId(), context.decisionId(), context.seat()),
              actions,
              preferredAction,
              teacher,
              .5,
              0,
              0,
              CfrObservationCodec.encodePayload(context));
      for (int update = 0; update < 64; update++) model.train(List.of(row));
      double[] policy = model.evaluate(List.of(context)).getFirst().targetPolicy();
      Assert.assertTrue(
          policy[preferredAction] > 1d / actions.length + .05,
          "Teacher training must make the saved average policy nonuniform: preferredAction="
              + preferredAction
              + ", policy="
              + Arrays.toString(policy));
      model.saveModel(checkpoint);
      return policy;
    }
  }

  private static DecisionContext trainingContext() {
    var engine = new GameEngine(877);
    var point = ((GameStepResult.AwaitingDecisions) engine.stepHanchan()).decisions().getFirst();
    return new DecisionContext(
        1, point.id(), point.player(), engine.observation(point.player()), point.legalActions());
  }

  private static CfrConfig config(EpsilonUtilityProfile profile) {
    CfrConfig defaults = CfrConfig.defaults();
    return new CfrConfig(
        123,
        1,
        1,
        profile,
        new CfrConfig.Collection(
            "HAND_GRP",
            1,
            4,
            16,
            .1,
            false,
            16,
            2,
            4,
            new DecisionDuelArenaSettings(
                Math.min(32, Runtime.getRuntime().availableProcessors()), 2, 64),
            new InferenceBatchingSettings(0),
            new DecisionBranchComparisonSettings(false, false, true, 1, 8, .01)),
        defaults.learning(),
        defaults.replay(),
        new CfrConfig.Network(16, new DeviceSettings(true, "cpu", "cpu", "cpu")),
        defaults.boundary(),
        new CfrConfig.Evaluation(
            10,
            new DecisionChampionDuelSettings(1, 2, .01, .25),
            new DecisionEvalVsSettings(4, 16, 1, 40000),
            new DecisionDuelArenaSettings(1, 1, 4),
            new InferenceBatchingSettings(0)),
        CfrConfig.defaults().initialization());
  }

  private static void delete(Path directory) throws IOException {
    try (var paths = Files.walk(directory)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    }
  }
}
