package com.epsilon.pico;

import com.epsilon.pico.workflow.audit.CallCounterfactualAuditWorkflow;
import com.epsilon.pico.workflow.audit.PairedTrajectoryAuditWorkflow;
import com.epsilon.pico.workflow.audit.ValueGrpAuditWorkflow;
import com.epsilon.pico.workflow.benchmark.DuelBenchmarkWorkflow;
import com.epsilon.pico.workflow.benchmark.InferenceProfileBenchmarkWorkflow;
import com.epsilon.pico.workflow.benchmark.InferenceReplayBenchmarkWorkflow;
import com.epsilon.pico.workflow.benchmark.PretrainBenchmarkWorkflow;
import com.epsilon.pico.workflow.benchmark.SelfPlayBenchmarkWorkflow;
import com.epsilon.pico.workflow.checkpoint.InitDecisionCheckpointWorkflow;
import com.epsilon.pico.workflow.evaluation.BeliefCalibrationWorkflow;
import com.epsilon.pico.workflow.evaluation.DecisionEvaluationWorkflow;
import com.epsilon.pico.workflow.evaluation.DecisionPretrainValidationWorkflow;
import com.epsilon.pico.workflow.evaluation.DecisionVersusWorkflow;
import com.epsilon.pico.workflow.evaluation.WallEvaluationWorkflow;
import com.epsilon.pico.workflow.play.RiichiPlayWorkflow;
import com.epsilon.pico.workflow.probe.CallCounterfactualProbeWorkflow;
import com.epsilon.pico.workflow.promotion.DecisionExternalPromotionWorkflow;
import com.epsilon.pico.workflow.training.BeliefLogPretrainWorkflow;
import com.epsilon.pico.workflow.training.DecisionKlPlanWorkflow;
import com.epsilon.pico.workflow.training.DecisionKlTrialWorkflow;
import com.epsilon.pico.workflow.training.DecisionLogPretrainWorkflow;
import com.epsilon.pico.workflow.training.DecisionTrainingWorkflow;
import com.epsilon.pico.workflow.training.GrpLogPretrainWorkflow;
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
            new InferenceProfileBenchmarkWorkflow(),
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
