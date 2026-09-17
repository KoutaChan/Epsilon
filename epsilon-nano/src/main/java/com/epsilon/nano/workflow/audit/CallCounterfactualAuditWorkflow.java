package com.epsilon.nano.workflow.audit;

import com.epsilon.nano.ai.decision.audit.EpsilonDecisionCallCounterfactualAudit;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;

/** 同じ局面から見送りとチー・ポンに分岐し、その後の対局結果を比較するコマンド。 */
public final class CallCounterfactualAuditWorkflow
    implements CommandWorkflow<EpsilonDecisionCallCounterfactualAudit.AuditReport> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.AUDIT_DECISION_COUNTERFACTUAL_CALLS;
  }

  @Override
  public EpsilonDecisionCallCounterfactualAudit.AuditReport execute(WorkflowArguments arguments)
      throws Exception {
    return EpsilonDecisionCallCounterfactualAudit.run(
        arguments.path(0, "reportFile"),
        arguments.path(1, "traceFile"),
        arguments.path(2, "grpCheckpointRoot"),
        arguments.path(3, "parentCheckpoint"),
        arguments.path(4, "candidateCheckpoint"),
        arguments.integer(5, "baseGames"),
        arguments.longValue(6, "seedBase"),
        arguments.integer(7, "gamesInFlight"),
        arguments.settings());
  }

  @Override
  public String summarize(EpsilonDecisionCallCounterfactualAudit.AuditReport report) {
    return "baseGames="
        + report.baseGames()
        + " roots="
        + report.overall().roots()
        + " callMinusPassRank="
        + report.overall().callMinusPassRankMean()
        + " lower95="
        + report.overall().callMinusPassRankLower95()
        + " upper95="
        + report.overall().callMinusPassRankUpper95()
        + " signP="
        + report.overall().twoSidedSignP();
  }
}
