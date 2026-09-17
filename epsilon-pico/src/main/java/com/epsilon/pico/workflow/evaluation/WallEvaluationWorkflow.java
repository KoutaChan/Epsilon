package com.epsilon.pico.workflow.evaluation;

import com.epsilon.ai.decision.duel.DuelEvaluation;
import com.epsilon.ai.decision.duel.EpsilonDecisionWallDuelEvaluator;
import com.epsilon.config.settings.DecisionEvalVsSettings;
import com.epsilon.pico.ai.decision.audit.EpsilonDecisionProductionAuditProtocol;
import com.epsilon.pico.ai.decision.duel.EpsilonDecisionDuelSession;
import com.epsilon.pico.config.settings.DecisionSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 本番用モデルの採用基準に従い、牌山を固定して対戦評価するコマンド。 */
public final class WallEvaluationWorkflow
    implements CommandWorkflow<WallEvaluationWorkflow.Result> {

  private static final Logger log = LoggerFactory.getLogger(WallEvaluationWorkflow.class);

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.EVAL_VS_WALL;
  }

  @Override
  public Result execute(WorkflowArguments arguments) throws Exception {
    Path reportFile = arguments.path(0, "reportFile").toAbsolutePath().normalize();
    int wallSeeds = arguments.integer(3, "wallSeeds");
    long seedBase = arguments.longValue(4, "seedBase");
    long duelSequence = arguments.longValue(5, "duelSequence");
    var plan =
        EpsilonDecisionWallDuelEvaluator.EvaluationPlan.fixed(
            wallSeeds, arguments.doubleValue(6, "alpha"));
    double promotionMargin = arguments.doubleValue(7, "promotionMargin");
    double harmfulMargin = arguments.doubleValue(8, "harmfulMargin");
    var config = arguments.settings();
    DecisionEvalVsSettings settings = config.bind(DecisionEvalVsSettings.class);
    EpsilonDecisionWallDuelEvaluator.Result evaluation;
    try (var context = new DecisionExecutionContext();
        var session =
            EpsilonDecisionDuelSession.open(
                arguments.path(1, "candidateCheckpoint"),
                arguments.path(2, "parentCheckpoint"),
                context,
                config)) {
      evaluation =
          EpsilonDecisionWallDuelEvaluator.evaluate(
              session,
              plan,
              seedBase,
              duelSequence,
              promotionMargin,
              harmfulMargin,
              config.bind(DecisionSettings.class).utilityProfile(),
              settings);
    }
    String rendered = render(evaluation, wallSeeds, seedBase, settings);
    Path reportParent = reportFile.getParent();
    if (reportParent != null) {
      Files.createDirectories(reportParent);
    }
    Files.writeString(reportFile, rendered);
    log.info("\nV20_DUEL_RESULT_BEGIN\n{}V20_DUEL_RESULT_END", rendered);
    return new Result(reportFile, evaluation);
  }

  @Override
  public String summarize(Result result) {
    return "report="
        + result.reportFile()
        + " wallSeeds="
        + result.evaluation().wallSeeds()
        + " selectionMode=POLICY_GREEDY decision="
        + result.evaluation().decision();
  }

  static String render(
      EpsilonDecisionWallDuelEvaluator.Result result,
      int requestedWallSeeds,
      long seedBase,
      DecisionEvalVsSettings settings) {
    DuelEvaluation.Result descriptive = result.descriptiveResult();
    return "candidateCheckpoint="
        + result.candidateCheckpoint()
        + System.lineSeparator()
        + "parentCheckpoint="
        + result.parentCheckpoint()
        + System.lineSeparator()
        + "protocolId="
        + EpsilonDecisionProductionAuditProtocol.ID
        + System.lineSeparator()
        + "auditId="
        + EpsilonDecisionProductionAuditProtocol.auditId(seedBase, result.duelSequence())
        + System.lineSeparator()
        + "selectionMode=POLICY_GREEDY"
        + System.lineSeparator()
        + "requestedWallSeeds="
        + requestedWallSeeds
        + System.lineSeparator()
        + "exactWallSeeds=true"
        + System.lineSeparator()
        + "continuousArena=true"
        + System.lineSeparator()
        + "confidenceMethod=FIXED_SAMPLE_GAUSSIAN"
        + System.lineSeparator()
        + "gamesInFlight="
        + settings.gamesInFlight()
        + System.lineSeparator()
        + "progressIntervalWallSeeds="
        + settings.progressIntervalWallSeeds()
        + System.lineSeparator()
        + "games="
        + result.games()
        + System.lineSeparator()
        + "wallSeeds="
        + result.wallSeeds()
        + System.lineSeparator()
        + "progressUpdates="
        + result.progressUpdates()
        + System.lineSeparator()
        + "duelSequence="
        + result.duelSequence()
        + System.lineSeparator()
        + "alpha="
        + result.alpha()
        + System.lineSeparator()
        + "seedBase="
        + seedBase
        + System.lineSeparator()
        + "promotionMargin="
        + result.promotionMargin()
        + System.lineSeparator()
        + "harmfulMargin="
        + result.harmfulMargin()
        + System.lineSeparator()
        + "decision="
        + result.decision()
        + System.lineSeparator()
        + "utilityProfile="
        + result.utilityProfile()
        + System.lineSeparator()
        + "pairedUtilityDeltaMean="
        + result.pairedUtilityDeltaMean()
        + System.lineSeparator()
        + "pairedUtilityDeltaLower="
        + result.pairedUtilityDeltaLower()
        + System.lineSeparator()
        + "pairedUtilityDeltaUpper="
        + result.pairedUtilityDeltaUpper()
        + System.lineSeparator()
        + "pairedRankDeltaMean="
        + result.pairedRankDeltaMean()
        + System.lineSeparator()
        + "pairedRankDeltaLower="
        + result.pairedRankDeltaLower()
        + System.lineSeparator()
        + "pairedRankDeltaUpper="
        + result.pairedRankDeltaUpper()
        + System.lineSeparator()
        + "pairedRankDeltaSe="
        + descriptive.pairedRankDeltaSe()
        + System.lineSeparator()
        + "candidateAverageRank="
        + descriptive.candidateAverageRank()
        + System.lineSeparator()
        + "opponentAverageRank="
        + descriptive.opponentAverageRank()
        + System.lineSeparator()
        + "candidateTopRate="
        + descriptive.candidateTopRate()
        + System.lineSeparator()
        + "opponentTopRate="
        + descriptive.opponentTopRate()
        + System.lineSeparator()
        + "candidateLastRate="
        + descriptive.candidateLastRate()
        + System.lineSeparator()
        + "opponentLastRate="
        + descriptive.opponentLastRate()
        + System.lineSeparator()
        + "scoreAdvantage="
        + descriptive.scoreAdvantage()
        + System.lineSeparator()
        + "utilityProfileAdvantages="
        + Arrays.toString(descriptive.utilityProfileAdvantages())
        + System.lineSeparator();
  }

  public record Result(Path reportFile, EpsilonDecisionWallDuelEvaluator.Result evaluation) {}
}
