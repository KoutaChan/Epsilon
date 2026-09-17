package com.epsilon.pico.workflow.audit;

import com.epsilon.pico.ai.decision.audit.EpsilonDecisionPairedTrajectoryAudit;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;

/** 同じ対局データを更新前モデルと候補モデルで評価し、予測の差を調べるコマンド。 */
public final class PairedTrajectoryAuditWorkflow
    implements CommandWorkflow<EpsilonDecisionPairedTrajectoryAudit.AuditReport> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.AUDIT_DECISION_PAIRED_TRAJECTORY;
  }

  @Override
  public EpsilonDecisionPairedTrajectoryAudit.AuditReport execute(WorkflowArguments arguments)
      throws Exception {
    return EpsilonDecisionPairedTrajectoryAudit.run(
        arguments.path(0, "reportFile"),
        arguments.path(1, "traceFile"),
        arguments.path(2, "scratchDir"),
        arguments.path(3, "grpCheckpointRoot"),
        arguments.path(4, "parentCheckpoint"),
        arguments.path(5, "candidateCheckpoint"),
        arguments.integer(6, "gamesPerSource"),
        arguments.longValue(7, "seedBase"),
        arguments.settings());
  }

  @Override
  public String summarize(EpsilonDecisionPairedTrajectoryAudit.AuditReport report) {
    long samples = report.sources().stream().mapToLong(source -> source.overall().samples()).sum();
    return "gamesPerSource=" + report.gamesPerSource() + " samples=" + samples;
  }
}
