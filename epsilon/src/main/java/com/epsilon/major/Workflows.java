package com.epsilon.major;

import com.epsilon.major.workflow.audit.CallCounterfactualAuditWorkflow;
import com.epsilon.major.workflow.audit.PairedTrajectoryAuditWorkflow;
import com.epsilon.major.workflow.audit.ValueGrpAuditWorkflow;
import com.epsilon.major.workflow.benchmark.DuelBenchmarkWorkflow;
import com.epsilon.major.workflow.benchmark.InferenceReplayBenchmarkWorkflow;
import com.epsilon.major.workflow.benchmark.PretrainBenchmarkWorkflow;
import com.epsilon.major.workflow.benchmark.SelfPlayBenchmarkWorkflow;
import com.epsilon.major.workflow.checkpoint.InitDecisionCheckpointWorkflow;
import com.epsilon.major.workflow.evaluation.BeliefCalibrationWorkflow;
import com.epsilon.major.workflow.evaluation.DecisionEvaluationWorkflow;
import com.epsilon.major.workflow.evaluation.DecisionPretrainValidationWorkflow;
import com.epsilon.major.workflow.evaluation.DecisionVersusWorkflow;
import com.epsilon.major.workflow.evaluation.WallEvaluationWorkflow;
import com.epsilon.major.workflow.play.RiichiPlayWorkflow;
import com.epsilon.major.workflow.probe.CallCounterfactualProbeWorkflow;
import com.epsilon.major.workflow.promotion.DecisionExternalPromotionWorkflow;
import com.epsilon.major.workflow.training.BeliefLogPretrainWorkflow;
import com.epsilon.major.workflow.training.DecisionKlPlanWorkflow;
import com.epsilon.major.workflow.training.DecisionKlTrialWorkflow;
import com.epsilon.major.workflow.training.DecisionLogPretrainWorkflow;
import com.epsilon.major.workflow.training.DecisionTrainingWorkflow;
import com.epsilon.major.workflow.training.GrpLogPretrainWorkflow;
import com.epsilon.workflow.WorkflowRegistry;
import com.epsilon.workflow.benchmark.EngineBenchmarkWorkflow;
import java.util.List;

final class Workflows {
  private Workflows() {}

  /** コマンドラインから実行できるコマンドを、ヘルプの表示順に登録する。 */
  public static WorkflowRegistry create() {
    return WorkflowRegistry.of(
        List.of(
            new InitDecisionCheckpointWorkflow(),
            new GrpLogPretrainWorkflow(),
            new BeliefLogPretrainWorkflow(),
            new BeliefCalibrationWorkflow(),
            new DecisionLogPretrainWorkflow(),
            new DecisionPretrainValidationWorkflow(),
            new PretrainBenchmarkWorkflow(),
            new SelfPlayBenchmarkWorkflow(),
            new DuelBenchmarkWorkflow(),
            new InferenceReplayBenchmarkWorkflow(),
            new ValueGrpAuditWorkflow(),
            new PairedTrajectoryAuditWorkflow(),
            new CallCounterfactualAuditWorkflow(),
            new CallCounterfactualProbeWorkflow(),
            new DecisionTrainingWorkflow(),
            new DecisionKlPlanWorkflow(),
            new DecisionKlTrialWorkflow(),
            new DecisionEvaluationWorkflow(),
            new DecisionVersusWorkflow(),
            new WallEvaluationWorkflow(),
            new DecisionExternalPromotionWorkflow(),
            new RiichiPlayWorkflow(),
            new EngineBenchmarkWorkflow()));
  }
}
