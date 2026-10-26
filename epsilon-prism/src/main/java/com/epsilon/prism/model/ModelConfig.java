package com.epsilon.prism.model;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import java.util.Objects;

/** 網とともに保存する構成と最適化の設定。欄の名前と並びは版の {@code m0/model.json} の {@code config} の形式。 */
public record ModelConfig(
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
  public ModelConfig {
    Objects.requireNonNull(computePrecision, "computePrecision");
    Objects.requireNonNull(tensorTransfer, "tensorTransfer");
    Objects.requireNonNull(utilityProfile, "utilityProfile");
    if (!utilityProfile.rankBased())
      throw new IllegalArgumentException("Prism requires a rank-based utility profile");
    if (hiddenSize < 8 || hiddenSize % 4 != 0 || microBatchSize < 1 || inferenceBatchSize < 1)
      throw new IllegalArgumentException("Invalid Prism network dimensions or batch sizes");
    if (!Float.isFinite(learningRate)
        || learningRate <= 0
        || !Float.isFinite(weightDecay)
        || weightDecay < 0
        || !Float.isFinite(gradientClip)
        || gradientClip < 0) throw new IllegalArgumentException("Invalid Prism optimizer settings");
  }

  /** 推論・学習のバッチの大きさだけを変えた設定（版を別の run の設定で開くとき）。 */
  public ModelConfig withBatches(int microBatchSize, int inferenceBatchSize) {
    return new ModelConfig(
        hiddenSize,
        microBatchSize,
        learningRate,
        weightDecay,
        gradientClip,
        seed,
        inferenceBatchSize,
        computePrecision,
        tensorTransfer,
        utilityProfile);
  }

  public DecisionComputePrecision precision(Device device) {
    return device.isGpu() ? computePrecision : DecisionComputePrecision.FLOAT32;
  }

  /**
   * device で使う入力の転送の形。GPU では保存した形（ふつうは PINNED_BUFFER）、CPU では常に DIRECT_BUFFER（GPU で作った版を GPU を隠した CPU で
   * 読んでも、ページ固定のバッファを取らない）。
   */
  public DecisionTensorTransfer transfer(Device device) {
    return device.isGpu() ? tensorTransfer : DecisionTensorTransfer.DIRECT_BUFFER;
  }
}
