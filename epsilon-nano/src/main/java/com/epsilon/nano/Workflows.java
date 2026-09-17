package com.epsilon.nano;

import com.epsilon.nano.workflow.audit.CallCounterfactualAuditWorkflow;
import com.epsilon.nano.workflow.audit.PairedTrajectoryAuditWorkflow;
import com.epsilon.nano.workflow.audit.ValueGrpAuditWorkflow;
import com.epsilon.nano.workflow.benchmark.DuelBenchmarkWorkflow;
import com.epsilon.nano.workflow.benchmark.InferenceReplayBenchmarkWorkflow;
import com.epsilon.nano.workflow.benchmark.PretrainBenchmarkWorkflow;
import com.epsilon.nano.workflow.benchmark.SelfPlayBenchmarkWorkflow;
import com.epsilon.nano.workflow.checkpoint.InitDecisionCheckpointWorkflow;
import com.epsilon.nano.workflow.evaluation.BeliefCalibrationWorkflow;
import com.epsilon.nano.workflow.evaluation.DecisionEvaluationWorkflow;
import com.epsilon.nano.workflow.evaluation.DecisionPretrainValidationWorkflow;
import com.epsilon.nano.workflow.evaluation.DecisionVersusWorkflow;
import com.epsilon.nano.workflow.evaluation.WallEvaluationWorkflow;
import com.epsilon.nano.workflow.play.RiichiPlayWorkflow;
import com.epsilon.nano.workflow.probe.CallCounterfactualProbeWorkflow;
import com.epsilon.nano.workflow.promotion.DecisionExternalPromotionWorkflow;
import com.epsilon.nano.workflow.training.BeliefLogPretrainWorkflow;
import com.epsilon.nano.workflow.training.DecisionKlPlanWorkflow;
import com.epsilon.nano.workflow.training.DecisionKlTrialWorkflow;
import com.epsilon.nano.workflow.training.DecisionLogPretrainWorkflow;
import com.epsilon.nano.workflow.training.DecisionTrainingWorkflow;
import com.epsilon.nano.workflow.training.GrpLogPretrainWorkflow;
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
