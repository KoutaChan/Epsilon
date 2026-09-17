package com.epsilon.major.client.tenhou;

import ai.djl.Model;
import com.epsilon.ai.decision.DecisionSelectionMode;
import com.epsilon.client.tenhou.TenhouClient;
import com.epsilon.config.settings.DecisionInferenceFusionSettings;
import com.epsilon.config.settings.DeviceSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.major.ai.decision.arena.EpsilonDecisionPlayer;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionInferenceServer;
import com.epsilon.major.ai.decision.training.EpsilonDecisionCheckpointManager;
import com.epsilon.major.ai.network.NetworkFactory;
import com.epsilon.major.config.settings.DecisionInferenceSettings;
import com.epsilon.major.config.settings.EpsilonSettings;
import com.epsilon.workflow.StartupSettings;
import java.io.IOException;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 系列の設定と推論モデルを開き、共通の天鳳クライアントへ接続する。 */
public final class TenhouMain {

  private static final Logger log = LoggerFactory.getLogger(TenhouMain.class);
  private static final String DEFAULT_CHECKPOINT = "checkpoints/epsilon";

  private TenhouMain() {}

  /** チェックポイントと推論サーバーは接続終了後に生成側で閉じる。 */
  public static void main(String[] args) throws IOException {
    var startup = StartupSettings.read("epsilon", EpsilonSettings.DEFAULT_RESOURCE, args);
    SettingsLoader settings = EpsilonSettings.load(startup.path(), startup.overrides());
    var options = TenhouClient.Options.parse(startup.arguments(), DEFAULT_CHECKPOINT);
    if (options == null) {
      TenhouClient.printUsage(DEFAULT_CHECKPOINT);
      System.exit(1);
      return;
    }

    try {
      Path checkpointRoot = Path.of(options.checkpointDir());
      Path checkpoint = EpsilonDecisionCheckpointManager.resolveExistingStrict(checkpointRoot);
      if (checkpoint == null) {
        throw new IOException("Accepted Decision checkpoint not found: " + checkpointRoot);
      }
      var inference = settings.bind(DecisionInferenceSettings.class);
      var fusion = settings.bind(DecisionInferenceFusionSettings.class);
      var devices = settings.bind(DeviceSettings.class);
      log.info("Loading models...");
      try (Model model =
              EpsilonDecisionCheckpointManager.loadForInference(
                  checkpoint, NetworkFactory.getInferenceDevices(devices).primary(), inference);
          EpsilonDecisionInferenceServer server =
              EpsilonDecisionInferenceServer.forFrozenModel(
                  model, inference.maxBatch(), inference, fusion)) {
        log.info("Models loaded");
        var player =
            EpsilonDecisionPlayer.builder(server, settings)
                .selectionMode(
                    options.samplePolicy()
                        ? DecisionSelectionMode.FULL_SUPPORT
                        : DecisionSelectionMode.POLICY_GREEDY)
                .build();
        TenhouClient.run(player, options);
      }
    } catch (Exception e) {
      log.error("Fatal error", e);
      System.exit(1);
    }
  }
}
