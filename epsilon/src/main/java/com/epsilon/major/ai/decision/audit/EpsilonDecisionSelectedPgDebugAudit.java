package com.epsilon.major.ai.decision.audit;

import ai.djl.Model;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.major.ai.decision.data.EpsilonDecisionTrainingSampleDescriptorReader;
import com.epsilon.major.ai.decision.training.DecisionOnlineLossConfig;
import com.epsilon.major.ai.decision.training.DecisionPolicySignalMultipliers;
import com.epsilon.major.ai.decision.training.DecisionPolicyTrustRegion;
import com.epsilon.major.config.settings.DecisionSelectedPgCampaignSettings;
import com.epsilon.major.config.settings.EpsilonSettings;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/**
 * 選択行動の方策勾配学習で、更新の基準となるモデルを検証する任意のデバッグ処理。
 *
 * <p>通常学習では空の結果を返して入力元を走査しない。監査計算は {@link EpsilonDecisionFinalPolicyAudit} に委譲する。
 */
public final class EpsilonDecisionSelectedPgDebugAudit {

  private EpsilonDecisionSelectedPgDebugAudit() {}

  public static Optional<Result> runIfEnabled(
      DecisionSelectedPgCampaignSettings settings,
      Model model,
      int expectedSamples,
      List<Path> fragmentPaths,
      EpsilonDecisionTrainingSampleDescriptorReader reader)
      throws IOException {
    return runIfEnabled(
        settings, model, expectedSamples, fragmentPaths, reader, EpsilonSettings.defaults());
  }

  /** 起動時に確定した設定スナップショットで監査する。 */
  public static Optional<Result> runIfEnabled(
      DecisionSelectedPgCampaignSettings settings,
      Model model,
      int expectedSamples,
      List<Path> fragmentPaths,
      EpsilonDecisionTrainingSampleDescriptorReader reader,
      SettingsLoader snapshot)
      throws IOException {
    if (!settings.debugFinalAuditEnabled()) {
      return Optional.empty();
    }

    DecisionOnlineLossConfig auditConfig =
        DecisionOnlineLossConfig.create(
            settings.policyUpdateClipRange(),
            settings.explorationCreditMix(),
            settings.entropyCoefficient());
    DecisionPolicySignalMultipliers policySignalMultipliers =
        new DecisionPolicySignalMultipliers(
            settings.dahaiWeight(), settings.riichiWeight(), settings.reactionWeight());
    EpsilonDecisionFinalPolicyAudit.Report report =
        EpsilonDecisionFinalPolicyAudit.evaluateStreaming(
            model,
            fragmentPaths,
            reader,
            settings.microBatchSize(),
            settings.finalAuditSampleLimit(),
            settings.maximumDeviceTransitionCells(),
            auditConfig,
            policySignalMultipliers,
            snapshot);
    if (report.sampleCount() != expectedSamples) {
      throw new IllegalStateException(
          "final policy audit did not cover the complete selected source: expected="
              + expectedSamples
              + " actual="
              + report.sampleCount());
    }
    return Optional.of(new Result(report));
  }

  public record Result(EpsilonDecisionFinalPolicyAudit.Report report) {

    public boolean passed(DecisionSelectedPgCampaignSettings settings) {
      return report.passed(new DecisionPolicyTrustRegion(settings.maximumOptimizerShardMeanKl()));
    }

    public String summary() {
      return "model{samples="
          + report.sampleCount()
          + ",meanRolloutPolicyKl="
          + report.meanRolloutPolicyKl()
          + ",maximumRolloutPolicyKl="
          + report.maximumRolloutPolicyKl()
          + ",observedMaximumPolicyUpdateRatio="
          + report.maximumObservedPolicyUpdateRatio()
          + "}";
    }
  }
}
