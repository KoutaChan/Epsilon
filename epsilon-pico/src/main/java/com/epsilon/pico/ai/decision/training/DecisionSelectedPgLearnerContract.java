package com.epsilon.pico.ai.decision.training;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionBranchComparisonSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.pico.ai.decision.EpsilonDecisionConstants;
import com.epsilon.pico.ai.decision.arena.DecisionAdaptiveExploration;
import com.epsilon.pico.ai.decision.arena.EpsilonDecisionPlayer;
import com.epsilon.pico.config.settings.DecisionSelectedPgCampaignSettings;
import com.epsilon.pico.config.settings.DecisionSelectedPgCampaignSettings.ActorKlControlSettings;
import com.epsilon.pico.config.settings.DecisionSettings;
import com.epsilon.pico.config.settings.EpsilonSettings;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;

/** AdamW の保存状態が、方策損失と対局データ収集の定義に適合するかを識別する。 */
final class DecisionSelectedPgLearnerContract {

  private static final String SCHEMA = "decision-selected-pg-actor-v12";

  private DecisionSelectedPgLearnerContract() {}

  /** 連続学習・trial・計画変更が共有する対局生成と保存契約を一度に解決する。 */
  static Resolved resolve(DecisionSelectedPgCampaignSettings settings) {
    return resolve(settings, EpsilonSettings.defaults());
  }

  static Resolved resolve(DecisionSelectedPgCampaignSettings settings, SettingsLoader config) {
    Objects.requireNonNull(settings, "settings");
    EpsilonDecisionPlayer.RolloutConfig actor =
        new EpsilonDecisionPlayer.RolloutConfig(
            config.bind(com.epsilon.config.settings.DecisionRolloutSettings.class).selectionMode(),
            config.bind(com.epsilon.config.settings.DecisionFullSupportSettings.class),
            settings.causalTraceLambda(),
            settings.explorationCreditMix());
    EpsilonDecisionPlayer.RolloutConfig opponent =
        new EpsilonDecisionPlayer.RolloutConfig(
            com.epsilon.ai.decision.DecisionSelectionMode.POLICY_GREEDY,
            com.epsilon.config.settings.DecisionFullSupportSettings.disabled(),
            settings.causalTraceLambda(),
            settings.explorationCreditMix());
    var optimizer = config.bind(DecisionSettings.class);
    return new Resolved(
        actor,
        opponent,
        fingerprint(
            settings,
            actor,
            optimizer.utilityProfile(),
            optimizer,
            config.bind(DecisionBranchComparisonSettings.class)));
  }

  record Resolved(
      EpsilonDecisionPlayer.RolloutConfig actorRollout,
      EpsilonDecisionPlayer.RolloutConfig opponentRollout,
      String learnerContractId) {}

  static String fingerprint(
      DecisionSelectedPgCampaignSettings campaign,
      EpsilonDecisionPlayer.RolloutConfig actorRollout,
      EpsilonUtilityProfile utilityProfile) {
    return fingerprint(campaign, actorRollout, utilityProfile, DecisionSettings.defaults());
  }

  static String fingerprint(
      DecisionSelectedPgCampaignSettings campaign,
      EpsilonDecisionPlayer.RolloutConfig actorRollout,
      EpsilonUtilityProfile utilityProfile,
      DecisionSettings optimizer) {
    return fingerprint(
        campaign,
        actorRollout,
        utilityProfile,
        optimizer,
        EpsilonSettings.defaults().bind(DecisionBranchComparisonSettings.class));
  }

  private static String fingerprint(
      DecisionSelectedPgCampaignSettings campaign,
      EpsilonDecisionPlayer.RolloutConfig actorRollout,
      EpsilonUtilityProfile utilityProfile,
      DecisionSettings optimizer,
      DecisionBranchComparisonSettings branch) {
    Objects.requireNonNull(campaign, "campaign");
    Objects.requireNonNull(actorRollout, "actorRollout");
    Objects.requireNonNull(utilityProfile, "utilityProfile");
    requireSame(
        "causalTraceLambda", campaign.causalTraceLambda(), actorRollout.causalTraceLambda());
    requireSame(
        "explorationCreditMix",
        campaign.explorationCreditMix(),
        actorRollout.explorationCreditMix());
    ActorKlControlSettings actorKlControl = campaign.optimizer().actorKlControl();
    // 目標計画はチェックポイントの制御状態が所有し、明示計画変更で変更できる。
    // 計画予算・終端倍率を互換性識別子へ含めると同じAdamWを保った延長を妨げるため、
    // ここでは更新回数と較正を含む不変の学習・制御意味と定義だけを固定する。
    String canonical =
        String.join(
            "\n",
            "schema=" + SCHEMA,
            "architecture=" + EpsilonDecisionConstants.ARCHITECTURE_ID,
            "utilityProfile=" + utilityProfile,
            "selectionMode=" + actorRollout.selectionMode(),
            "branchComparison=" + (branch.enabled() ? branch : "disabled"),
            "fullSupport=" + actorRollout.fullSupport(),
            "adaptiveHistogramBins=" + DecisionAdaptiveExploration.HISTOGRAM_BINS,
            "adaptiveMinimumCalibrationSamples="
                + DecisionAdaptiveExploration.MINIMUM_CALIBRATION_SAMPLES,
            "causalTraceLambda=" + Float.toHexString(actorRollout.causalTraceLambda()),
            "dahaiWeight=" + Float.toHexString(campaign.dahaiWeight()),
            "riichiWeight=" + Float.toHexString(campaign.riichiWeight()),
            "reactionWeight=" + Float.toHexString(campaign.reactionWeight()),
            "policyUpdateClipRange=" + Float.toHexString(campaign.policyUpdateClipRange()),
            "explorationCreditMix=" + Float.toHexString(actorRollout.explorationCreditMix()),
            "entropyCoefficient=" + Float.toHexString(campaign.entropyCoefficient()),
            "ppoEpochs=" + campaign.ppoEpochs(),
            "optimizerStepsPerEpoch=" + campaign.optimizerStepsPerEpoch(),
            "maximumOptimizerShardMeanKl="
                + Float.toHexString(campaign.maximumOptimizerShardMeanKl()),
            "valueLearningRate=" + Float.toHexString(campaign.optimizer().valueLearningRate()),
            "actorInitialLearningRate=" + Float.toHexString(actorKlControl.initialLearningRate()),
            "actorHardMinimumLearningRate="
                + Float.toHexString(actorKlControl.hardMinimumLearningRate()),
            "actorHardMaximumLearningRate="
                + Float.toHexString(actorKlControl.hardMaximumLearningRate()),
            "actorKlWarmupMacros=" + actorKlControl.warmupMacros(),
            "actorKlMedianWindowMacros=" + actorKlControl.medianWindowMacros(),
            "actorKlTargetMultiplier=" + Float.toHexString(actorKlControl.targetMultiplier()),
            "actorKlTargetClock=accepted-actor-optimizer-updates",
            "actorKlTargetSchedule=initial-hold10pct-then-cosine-explicit-replan",
            "actorKlDeadbandFactor=" + Float.toHexString(actorKlControl.deadbandFactor()),
            "actorKlMaximumAdjustmentFactorPerMacro="
                + Float.toHexString(actorKlControl.maximumAdjustmentFactorPerMacro()),
            "weightDecay=" + Float.toHexString(optimizer.weightDecay()),
            "gradientClip=" + Float.toHexString(optimizer.gradClip()));
    try {
      byte[] digest =
          MessageDigest.getInstance("SHA-256").digest(canonical.getBytes(StandardCharsets.UTF_8));
      return SCHEMA + ":" + HexFormat.of().formatHex(digest);
    } catch (NoSuchAlgorithmException impossible) {
      throw new AssertionError("SHA-256 is required by the Java platform", impossible);
    }
  }

  private static void requireSame(String name, float campaignValue, float rolloutValue) {
    if (Float.compare(campaignValue, rolloutValue) != 0) {
      throw new IllegalArgumentException(
          name
              + " must match between campaign and actor rollout: campaign="
              + campaignValue
              + " rollout="
              + rolloutValue);
    }
  }
}
