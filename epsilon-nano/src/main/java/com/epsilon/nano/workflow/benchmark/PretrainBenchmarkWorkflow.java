package com.epsilon.nano.workflow.benchmark;

import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.nano.ai.decision.benchmark.EpsilonDecisionPretrainBenchmark;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;

/** Decision の全パラメータを複数デバイスで事前学習する速度を測定するコマンド。 */
public final class PretrainBenchmarkWorkflow
    implements CommandWorkflow<EpsilonDecisionPretrainBenchmark.Report> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.BENCHMARK_DECISION_PRETRAIN;
  }

  @Override
  public EpsilonDecisionPretrainBenchmark.Report execute(WorkflowArguments arguments)
      throws Exception {
    return EpsilonDecisionPretrainBenchmark.run(
        arguments.optionalEnum(
            6,
            "tensorTransfer",
            DecisionTensorTransfer.class,
            DecisionTensorTransfer.DIRECT_BUFFER),
        arguments.optionalInteger(0, "optimizerBatchRows", 1_024),
        arguments.optionalInteger(1, "maxDeviceBatchRows", 512),
        arguments.optionalInteger(2, "warmupSteps", 3),
        arguments.optionalInteger(3, "measuredSteps", 10),
        arguments.optionalInteger(4, "legalActionBucket", 16),
        arguments.optionalInteger(5, "actionTransitionBucket", 8),
        arguments.settings());
  }

  @Override
  public String summarize(EpsilonDecisionPretrainBenchmark.Report report) {
    return "execution=FULL_MODEL_DATA_PARALLEL"
        + " tensorTransfer="
        + report.tensorTransfer()
        + " computePrecision="
        + report.computePrecision()
        + " availableDevices="
        + report.availableDevices()
        + " activeDevices="
        + report.activeDevices()
        + " optimizerBatchRows="
        + report.optimizerBatchRows()
        + " maxDeviceBatchRows="
        + report.maximumDeviceBatchRows()
        + " warmupSteps="
        + report.warmupSteps()
        + " measuredSteps="
        + report.measuredSteps()
        + " legalActionCapacity="
        + report.legalActionCapacity()
        + " actionTransitionCapacity="
        + report.actionTransitionCapacity()
        + " measuredRows="
        + report.measuredRows()
        + " elapsedMillis="
        + report.elapsedMillis()
        + " rowsPerSecond="
        + report.rowsPerSecond()
        + " loss="
        + report.loss()
        + " behaviorCloningLoss="
        + report.behaviorCloningLoss()
        + " valueLoss="
        + report.valueLoss();
  }
}
