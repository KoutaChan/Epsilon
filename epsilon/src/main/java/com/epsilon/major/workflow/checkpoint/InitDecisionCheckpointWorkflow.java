package com.epsilon.major.workflow.checkpoint;

import ai.djl.Model;
import com.epsilon.major.ai.decision.EpsilonDecisionConstants;
import com.epsilon.major.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.major.ai.network.NetworkFactory;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.nio.file.Path;

/** 新しい Decision チェックポイントを初期化するワークフロー。 */
public final class InitDecisionCheckpointWorkflow
    implements CommandWorkflow<InitDecisionCheckpointWorkflow.Result> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.INIT_DECISION_CHECKPOINT;
  }

  @Override
  public Result execute(WorkflowArguments arguments) throws Exception {
    Path directory = arguments.optionalPath(0, Path.of("checkpoints/epsilon"));
    var config =
        arguments.settings().bind(com.epsilon.major.config.settings.DecisionSettings.class);
    var devices = arguments.settings().bind(com.epsilon.config.settings.DeviceSettings.class);
    try (Model model =
        NetworkFactory.createDecisionModel(
            NetworkFactory.getLearnerDevice(devices),
            true,
            config.hidden(),
            config.utilityProfile())) {
      EpsilonDecisionCheckpointManager.saveInitial(model, directory);
    }
    return new Result(directory, EpsilonDecisionConstants.ARCHITECTURE_ID);
  }

  @Override
  public String summarize(Result result) {
    return "checkpointDir=" + result.directory() + " architecture=" + result.architecture();
  }

  public record Result(Path directory, String architecture) {}
}
