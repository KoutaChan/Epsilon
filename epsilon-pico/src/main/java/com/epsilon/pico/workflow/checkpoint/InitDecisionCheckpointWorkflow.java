package com.epsilon.pico.workflow.checkpoint;

import ai.djl.Model;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.pico.ai.decision.EpsilonDecisionConstants;
import com.epsilon.pico.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.pico.ai.network.NetworkFactory;
import com.epsilon.pico.config.settings.DecisionSettings;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.nio.file.Path;

/** 新しい Decision チェックポイントを初期化するコマンド。 */
public final class InitDecisionCheckpointWorkflow
    implements CommandWorkflow<InitDecisionCheckpointWorkflow.Result> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.INIT_DECISION_CHECKPOINT;
  }

  @Override
  public Result execute(WorkflowArguments arguments) throws Exception {
    Path directory = arguments.optionalPath(0, Path.of("checkpoints/epsilon-pico"));
    var config = arguments.settings().bind(DecisionSettings.class);
    var devices = arguments.settings().bind(DeviceSettings.class);
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
