package com.epsilon.nano.workflow.audit;

import com.epsilon.nano.ai.decision.audit.EpsilonDecisionValueGrpAudit;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.util.List;

/** 固定した検証データで、Decision の価値予測と GRP の教師値の整合性を調べるコマンド。 */
public final class ValueGrpAuditWorkflow
    implements CommandWorkflow<EpsilonDecisionValueGrpAudit.AuditReport> {

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.AUDIT_DECISION_VALUE_GRP;
  }

  @Override
  public EpsilonDecisionValueGrpAudit.AuditReport execute(WorkflowArguments arguments)
      throws Exception {
    List<EpsilonDecisionValueGrpAudit.CheckpointSpec> checkpoints =
        arguments.remaining(5).stream()
            .map(EpsilonDecisionValueGrpAudit.CheckpointSpec::parse)
            .toList();
    return EpsilonDecisionValueGrpAudit.run(
        arguments.path(0, "outputFile"),
        arguments.path(1, "inputFileManifest"),
        arguments.path(2, "grpCheckpointRoot"),
        arguments.floatValue(3, "validationFraction"),
        arguments.integer(4, "maxValidationFiles"),
        checkpoints,
        arguments.settings());
  }

  @Override
  public String summarize(EpsilonDecisionValueGrpAudit.AuditReport report) {
    return "samples="
        + report.samples()
        + " grpTeacherSamples="
        + report.grpTeacherSamples()
        + " checkpoints="
        + report.checkpoints().size();
  }
}
