package com.epsilon.nano.ai.decision.training;

import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.nano.ai.network.NetworkFactory;
import com.epsilon.nano.config.settings.DecisionSelectedPgCampaignSettings;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.nio.file.Path;

/** 保存中の学習器を保ったまま、現在の目標KLから次の減衰区間を設定する。 */
public final class DecisionKlTrainingPlan {
  private DecisionKlTrainingPlan() {}

  public static String replan(Path checkpointRoot, long endOptimizerStep, float endTargetKl)
      throws IOException {
    return replan(checkpointRoot, endOptimizerStep, endTargetKl, EpsilonSettings.defaults());
  }

  public static String replan(
      Path checkpointRoot, long endOptimizerStep, float endTargetKl, SettingsLoader config)
      throws IOException {
    EpsilonDecisionCheckpointManager.recoverWorkingCheckpoint(checkpointRoot);
    Path working = EpsilonDecisionCheckpointManager.working(checkpointRoot);
    var settings = config.bind(DecisionSelectedPgCampaignSettings.class);
    var resolved = DecisionSelectedPgLearnerContract.resolve(settings, config);
    var manifest = EpsilonDecisionCheckpointManager.requireValidCheckpoint(working);
    try (var context = new DecisionExecutionContext();
        var learner =
            DecisionLearner.openWorking(
                working,
                NetworkFactory.getLearnerDevice(
                    config.bind(com.epsilon.config.settings.DeviceSettings.class)),
                settings.optimizer(),
                resolved.learnerContractId(),
                context,
                config)) {
      learner.replanTargetKl(endOptimizerStep, endTargetKl);
      EpsilonDecisionCheckpointManager.saveWorking(
          learner, checkpointRoot, manifest.globalStep, manifest.iteration, manifest.selfPlayGames);
      var state = learner.controllerState();
      return "acceptedActorOptimizerSteps="
          + state.acceptedActorOptimizerSteps()
          + " currentTargetKl="
          + state.targetSamePathUpdateKl()
          + " endOptimizerStep="
          + endOptimizerStep
          + " endTargetKl="
          + endTargetKl;
    }
  }
}
