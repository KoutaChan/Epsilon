package com.epsilon.cfr.model;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import java.util.Objects;

/** 各モデルとともに保存する、ネットワーク構成と最適化の設定。 */
public record CfrModelConfig(
    int hiddenSize,
    int microBatchSize,
    float learningRate,
    float weightDecay,
    float gradientClip,
    long seed,
    int inferenceBatchSize,
    DecisionComputePrecision computePrecision,
    DecisionTensorTransfer tensorTransfer,
    EpsilonUtilityProfile utilityProfile) {
  public CfrModelConfig {
    Objects.requireNonNull(computePrecision, "computePrecision");
    Objects.requireNonNull(tensorTransfer, "tensorTransfer");
    Objects.requireNonNull(utilityProfile, "utilityProfile");
    if (!utilityProfile.rankBased())
      throw new IllegalArgumentException("CFR requires a rank-based utility profile");
    if (hiddenSize < 8 || hiddenSize % 4 != 0 || microBatchSize < 1)
      throw new IllegalArgumentException("Invalid CFR network dimensions or microbatch size");
    if (inferenceBatchSize < 1)
      throw new IllegalArgumentException("Invalid CFR inference batch size");
    if (!Float.isFinite(learningRate)
        || learningRate <= 0
        || !Float.isFinite(weightDecay)
        || weightDecay < 0
        || !Float.isFinite(gradientClip)
        || gradientClip < 0) throw new IllegalArgumentException("Invalid CFR optimizer settings");
  }

  public DecisionComputePrecision precision(Device device) {
    return device.isGpu() ? computePrecision : DecisionComputePrecision.FLOAT32;
  }
}
