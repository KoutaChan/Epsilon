package com.epsilon.major.workflow.evaluation;

import com.epsilon.ai.decision.duel.DuelEvaluation;
import com.epsilon.config.settings.DecisionEvalVsSettings;
import com.epsilon.major.ai.decision.duel.EpsilonDecisionDuelEvaluator;
import com.epsilon.runtime.DecisionExecutionContext;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/** 同じ牌山と席順の条件で、候補モデルと対戦相手を比較評価するコマンド。 */
public final class DecisionVersusWorkflow
    implements CommandWorkflow<DecisionVersusWorkflow.Result> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.EVAL_VS;
  }

  @Override
  public Result execute(WorkflowArguments arguments) throws Exception {
    Path candidate = arguments.optionalPath(0, Path.of("checkpoints/epsilon"));
    Path opponent = arguments.optionalPath(1, candidate);
    int games = arguments.optionalInteger(2, "games", 100);
    Path output = arguments.optionalPath(3, candidate.resolve("eval-vs.txt"));
    DuelEvaluation.Result evaluation;
    try (var context = new DecisionExecutionContext()) {
      evaluation =
          EpsilonDecisionDuelEvaluator.evaluate(
              candidate,
              opponent,
              games,
              arguments.settings().bind(DecisionEvalVsSettings.class).seedBase(),
              context,
              arguments.settings());
    }
    Path parent = output.toAbsolutePath().normalize().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    Files.writeString(output, render(candidate, opponent, evaluation));
    return new Result(output, evaluation);
  }

  @Override
  public String summarize(Result result) {
    DuelEvaluation.Result evaluation = result.evaluation();
    return "candidate="
        + evaluation.candidateCheckpoint()
        + " opponent="
        + evaluation.opponentCheckpoint()
        + " selectionMode=POLICY_GREEDY games="
        + evaluation.games()
        + " candidateRank="
        + evaluation.candidateAverageRank()
        + " opponentRank="
        + evaluation.opponentAverageRank()
        + " scoreAdvantage="
        + evaluation.scoreAdvantage()
        + " outFile="
        + result.output();
  }

  private static String render(Path candidate, Path opponent, DuelEvaluation.Result result) {
    return "candidateDir="
        + candidate
        + System.lineSeparator()
        + "opponentDir="
        + opponent
        + System.lineSeparator()
        + "candidateCheckpoint="
        + result.candidateCheckpoint()
        + System.lineSeparator()
        + "opponentCheckpoint="
        + result.opponentCheckpoint()
        + System.lineSeparator()
        + "selectionMode=POLICY_GREEDY"
        + System.lineSeparator()
        + "games="
        + result.games()
        + System.lineSeparator()
        + "wallSeeds="
        + result.wallSeeds()
        + System.lineSeparator()
        + "pairedRankDeltaMean="
        + result.pairedRankDeltaMean()
        + System.lineSeparator()
        + "pairedRankDeltaSe="
        + result.pairedRankDeltaSe()
        + System.lineSeparator()
        + "pairedRankDeltaLcb="
        + result.pairedRankDeltaLcb()
        + System.lineSeparator()
        + "candidateAverageRank="
        + result.candidateAverageRank()
        + System.lineSeparator()
        + "opponentAverageRank="
        + result.opponentAverageRank()
        + System.lineSeparator()
        + "candidateTopRate="
        + result.candidateTopRate()
        + System.lineSeparator()
        + "opponentTopRate="
        + result.opponentTopRate()
        + System.lineSeparator()
        + "candidateLastRate="
        + result.candidateLastRate()
        + System.lineSeparator()
        + "opponentLastRate="
        + result.opponentLastRate()
        + System.lineSeparator()
        + "scoreAdvantage="
        + result.scoreAdvantage()
        + System.lineSeparator()
        + "utilityProfileAdvantages="
        + Arrays.toString(result.utilityProfileAdvantages())
        + System.lineSeparator();
  }

  public record Result(Path output, DuelEvaluation.Result evaluation) {}
}
