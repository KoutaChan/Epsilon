package com.epsilon.pico.workflow.probe;

import com.epsilon.pico.ai.decision.probe.EpsilonDecisionCallCounterfactualOfflineProbe;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;

/** 鳴く場合と見送る場合の比較データを使い、モデルの中間表現と方策の差を調べるコマンド。 */
public final class CallCounterfactualProbeWorkflow
    implements CommandWorkflow<EpsilonDecisionCallCounterfactualOfflineProbe.ProbeReport> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.PROBE_DECISION_COUNTERFACTUAL_CALLS;
  }

  @Override
  public EpsilonDecisionCallCounterfactualOfflineProbe.ProbeReport execute(
      WorkflowArguments arguments) throws Exception {
    return EpsilonDecisionCallCounterfactualOfflineProbe.run(
        arguments.path(0, "reportFile"),
        arguments.path(1, "predictionTraceFile"),
        arguments.path(2, "grpCheckpointRoot"),
        arguments.path(3, "parentCheckpoint"),
        arguments.path(4, "candidateCheckpoint"),
        arguments.integer(5, "baseGames"),
        arguments.longValue(6, "seedBase"),
        arguments.integer(7, "gamesInFlight"),
        arguments.settings());
  }

  @Override
  public String summarize(EpsilonDecisionCallCounterfactualOfflineProbe.ProbeReport report) {
    return "roots="
        + report.dataset().roots()
        + " representation="
        + report.verdict().representationSignal()
        + " policy="
        + report.verdict().policySignal();
  }
}
