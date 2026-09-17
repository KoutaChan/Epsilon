package com.epsilon.nano.workflow.promotion;

import com.epsilon.nano.ai.decision.training.EpsilonDecisionExternalPromotionGate;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.nio.file.Files;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 独立した固定牌山での検証に合格した候補モデルを、本番用モデルとして採用するコマンド。 */
public final class DecisionExternalPromotionWorkflow
    implements CommandWorkflow<DecisionExternalPromotionWorkflow.Result> {

  private static final Logger log =
      LoggerFactory.getLogger(DecisionExternalPromotionWorkflow.class);

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.PROMOTE_DECISION_EXTERNAL_GATE;
  }

  @Override
  public Result execute(WorkflowArguments arguments) throws Exception {
    Path resultFile = arguments.path(0, "resultFile").toAbsolutePath().normalize();
    Path checkpointRoot = arguments.path(1, "checkpointRoot");
    EpsilonDecisionExternalPromotionGate.PromotionResult promotion =
        EpsilonDecisionExternalPromotionGate.promote(
            checkpointRoot,
            arguments.path(2, "candidateCheckpoint"),
            arguments.path(3, "fixedWallReport"));
    Path parent = resultFile.getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    String rendered = promotion.render();
    Files.writeString(resultFile, rendered);
    log.info("\nV20_EXTERNAL_PROMOTION_BEGIN\n{}V20_EXTERNAL_PROMOTION_END", rendered);
    return new Result(resultFile, checkpointRoot, promotion);
  }

  @Override
  public String summarize(Result result) {
    return "checkpointRoot="
        + result.checkpointRoot()
        + " candidate="
        + result.promotion().candidate()
        + " iteration="
        + result.promotion().iteration()
        + " wallSeeds="
        + result.promotion().wallSeeds()
        + " resultFile="
        + result.resultFile();
  }

  public record Result(
      Path resultFile,
      Path checkpointRoot,
      EpsilonDecisionExternalPromotionGate.PromotionResult promotion) {}
}
