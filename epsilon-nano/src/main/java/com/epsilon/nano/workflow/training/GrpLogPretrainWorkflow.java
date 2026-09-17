package com.epsilon.nano.workflow.training;

import com.epsilon.nano.ai.grp.EpsilonGrpTrainingRunner;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.nio.file.Path;

/** 牌譜から GRP を事前学習するワークフロー。 */
public final class GrpLogPretrainWorkflow
    implements CommandWorkflow<EpsilonGrpTrainingRunner.LogTrainSummary> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.PRETRAIN_GRP_LOGS;
  }

  @Override
  public EpsilonGrpTrainingRunner.LogTrainSummary execute(WorkflowArguments arguments)
      throws Exception {
    return EpsilonGrpTrainingRunner.trainFromLogs(
        arguments.optionalPath(0, Path.of("checkpoints/epsilon-nano")),
        arguments.optionalPath(1, Path.of("data/logs")),
        arguments.optionalInteger(2, "epochs", 1),
        arguments.optionalInteger(3, "maxFiles", 0),
        arguments.settings());
  }

  @Override
  public String summarize(EpsilonGrpTrainingRunner.LogTrainSummary result) {
    return "files="
        + result.files()
        + " skippedFiles="
        + result.skippedFiles()
        + " examples="
        + result.examples()
        + " batches="
        + result.batches()
        + " globalStep="
        + result.globalStep()
        + " iteration="
        + result.iteration()
        + " marginalNll="
        + result.marginalNll()
        + " rankAccuracy="
        + result.rankAccuracy()
        + " promotedEpochs="
        + result.promotedEpochs()
        + " rejectedEpochs="
        + result.rejectedEpochs()
        + " validationExamples="
        + result.validationExamples()
        + " validationMarginalNll="
        + result.validationMarginalNll()
        + " validationMarginalBrier="
        + result.validationMarginalBrier()
        + " validationMacroEce="
        + result.validationMacroEce()
        + " validationLastRankEce="
        + result.validationLastRankEce()
        + " validationRankAccuracy="
        + result.validationRankAccuracy()
        + " validationInvalidProbabilities="
        + result.validationInvalidProbabilities()
        + " validationMaxRowSumError="
        + result.validationMaxRowSumError()
        + " validationMaxColumnSumError="
        + result.validationMaxColumnSumError();
  }
}
