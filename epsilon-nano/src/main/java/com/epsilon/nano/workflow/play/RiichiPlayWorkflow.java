package com.epsilon.nano.workflow.play;

import ai.djl.Model;
import com.epsilon.ai.decision.DecisionSelectionMode;
import com.epsilon.client.riichi.RiichiClient;
import com.epsilon.config.settings.RiichiSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.arena.EpsilonDecisionPlayer;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionEvaluator;
import com.epsilon.nano.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.nano.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.io.IOException;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 一つの凍結Decision モデルを使い、RiichiLabで自動対局するワークフロー。 */
public final class RiichiPlayWorkflow implements CommandWorkflow<RiichiPlayWorkflow.Result> {

  private static final Logger log = LoggerFactory.getLogger(RiichiPlayWorkflow.class);

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.PLAY_RIICHI;
  }

  @Override
  public Result execute(WorkflowArguments arguments) throws Exception {
    RiichiSettings settings = arguments.settings().bind(RiichiSettings.class);
    if (settings.token().isBlank()) {
      throw new IllegalArgumentException(
          "Set epsilon.riichi.token in settings.toml before play-riichi");
    }
    Path directory = Path.of(settings.checkpointDir());
    Path checkpoint = EpsilonDecisionCheckpointManager.resolveExistingStrict(directory);
    if (checkpoint == null) {
      throw new IOException("Accepted Decision checkpoint not found: " + directory);
    }
    log.info("RiichiLab checkpoint: {}", checkpoint);
    var inference =
        arguments.settings().bind(com.epsilon.nano.config.settings.DecisionInferenceSettings.class);
    var fusion =
        arguments
            .settings()
            .bind(com.epsilon.config.settings.DecisionInferenceFusionSettings.class);
    var devices = arguments.settings().bind(com.epsilon.config.settings.DeviceSettings.class);
    try (Model model =
            EpsilonDecisionCheckpointManager.loadForInference(
                checkpoint, NetworkFactory.getInferenceDevices(devices).primary(), inference);
        EpsilonDecisionInferenceServer server =
            EpsilonDecisionInferenceServer.forFrozenModel(
                model, inference.maxBatch(), inference, fusion)) {
      SettingsLoader config = arguments.settings();
      warmUp(server, config);
      log.info("RiichiLab inference ready: policy=POLICY_GREEDY maxGames={}", settings.maxGames());
      return new Result(
          checkpoint,
          RiichiClient.play(
              settings.token(),
              settings.maxGames(),
              () ->
                  EpsilonDecisionPlayer.builder(server, config)
                      .selectionMode(DecisionSelectionMode.POLICY_GREEDY)
                      .build()));
    }
  }

  /** 配牌後の実際の合法手を評価し、初回順伝播を接続開始前に完了する。 */
  static void warmUp(EpsilonDecisionEvaluator evaluator) {
    warmUp(evaluator, EpsilonSettings.defaults());
  }

  static void warmUp(EpsilonDecisionEvaluator evaluator, SettingsLoader config) {
    GameEngine engine = new GameEngine(0L);
    var step = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
    var decision = step.decisions().getFirst();
    EpsilonDecisionPlayer.builder(evaluator, config)
        .selectionMode(DecisionSelectionMode.POLICY_GREEDY)
        .build()
        .selectAction(engine.getState(), decision.player(), decision.legalActions());
  }

  @Override
  public String summarize(Result result) {
    return "checkpoint="
        + result.checkpoint()
        + " completedGames="
        + result.play().completedGames()
        + " interruptedGames="
        + result.play().interruptedGames();
  }

  /** 使用チェックポイントと対局数の集計。 */
  public record Result(Path checkpoint, RiichiClient.Result play) {}
}
