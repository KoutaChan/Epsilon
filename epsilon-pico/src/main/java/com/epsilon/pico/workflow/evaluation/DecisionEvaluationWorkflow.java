package com.epsilon.pico.workflow.evaluation;

import ai.djl.Model;
import com.epsilon.pico.ai.decision.duel.EpsilonDecisionEvaluationRunner;
import com.epsilon.pico.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.pico.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/** Decision チェックポイントを通常対局で評価するワークフロー。 */
public final class DecisionEvaluationWorkflow
    implements CommandWorkflow<DecisionEvaluationWorkflow.Result> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.EVAL_DECISION;
  }

  @Override
  public Result execute(WorkflowArguments arguments) throws Exception {
    Path directory = arguments.optionalPath(0, Path.of("checkpoints/epsilon-pico"));
    int games = arguments.optionalInteger(1, "games", 100);
    Path tenhouLogDirectory = arguments.optionalPath(2, directory.resolve("eval-tenhou-logs"));
    Files.createDirectories(directory);
    Path checkpoint = EpsilonDecisionCheckpointManager.resolveExistingStrict(directory);
    if (checkpoint == null) {
      throw new IOException("Decision checkpoint not found: " + directory);
    }
    var inference =
        arguments.settings().bind(com.epsilon.pico.config.settings.DecisionInferenceSettings.class);
    var fusion =
        arguments
            .settings()
            .bind(com.epsilon.config.settings.DecisionInferenceFusionSettings.class);
    var devices = arguments.settings().bind(com.epsilon.config.settings.DeviceSettings.class);
    try (Model model =
            EpsilonDecisionCheckpointManager.loadForInference(
                checkpoint,
                com.epsilon.pico.ai.network.NetworkFactory.getInferenceDevices(devices).primary(),
                inference);
        EpsilonDecisionInferenceServer server =
            EpsilonDecisionInferenceServer.forFrozenModel(
                model, inference.maxBatch(), inference, fusion)) {
      EpsilonDecisionEvaluationRunner.Result evaluation =
          EpsilonDecisionEvaluationRunner.evaluate(
              server, games, tenhouLogDirectory, arguments.settings());
      Files.writeString(directory.resolve("eval-decision.txt"), render(checkpoint, evaluation));
      return new Result(checkpoint, evaluation);
    }
  }

  @Override
  public String summarize(Result result) {
    return "checkpoint="
        + result.checkpoint()
        + " selectionMode=POLICY_GREEDY games="
        + result.evaluation().games()
        + " lastFinalScores="
        + Arrays.toString(result.evaluation().lastFinalScores())
        + " tenhouLogDir="
        + result.evaluation().tenhouLogDir()
        + " tenhouLogFiles="
        + result.evaluation().tenhouLogFiles().size();
  }

  private static String render(Path checkpoint, EpsilonDecisionEvaluationRunner.Result evaluation) {
    return "checkpoint="
        + checkpoint
        + System.lineSeparator()
        + "selectionMode=POLICY_GREEDY"
        + System.lineSeparator()
        + "games="
        + evaluation.games()
        + System.lineSeparator()
        + "lastFinalScores="
        + Arrays.toString(evaluation.lastFinalScores())
        + System.lineSeparator()
        + "tenhouLogDir="
        + evaluation.tenhouLogDir()
        + System.lineSeparator()
        + "tenhouLogFiles="
        + evaluation.tenhouLogFiles().size()
        + System.lineSeparator();
  }

  public record Result(Path checkpoint, EpsilonDecisionEvaluationRunner.Result evaluation) {}
}
