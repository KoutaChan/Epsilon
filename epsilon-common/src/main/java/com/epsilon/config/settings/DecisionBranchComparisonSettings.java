package com.epsilon.config.settings;

/**
 * 局境界までの限定二分岐学習を設定する。既定では無効とし、有効化後は九種九牌だけを比較する。
 *
 * @param enabled 分岐比較による収集と学習を有効にするか
 * @param winDecline 探索で和了を見逃した判断を比較するか
 * @param kyushu 九種九牌の宣言と続行を比較し、専用ゲートを比較結果だけで学習するか
 * @param maxBranchesPerGame 一対局で開始する追加枝の上限
 * @param maxInFlightBranches 同時に実行する追加枝の上限
 * @param maxExtraInferenceRatio 通常の推論行数に対する追加推論行数の上限比率
 */
@SettingsPrefix("epsilon.decision.rollout.branchComparison")
public record DecisionBranchComparisonSettings(
    @Setting("enabled") @Default("false") boolean enabled,
    @Setting("winDecline") @Default("false") boolean winDecline,
    @Setting("kyushu") @Default("true") boolean kyushu,
    @Setting("maxBranchesPerGame") @Default("1") int maxBranchesPerGame,
    @Setting("maxInFlightBranches") @Default("8") int maxInFlightBranches,
    @Setting("maxExtraInferenceRatio") @Default("0.01") double maxExtraInferenceRatio) {

  public DecisionBranchComparisonSettings {
    if (maxBranchesPerGame < 1
        || maxInFlightBranches < 1
        || !Double.isFinite(maxExtraInferenceRatio)
        || maxExtraInferenceRatio <= 0
        || maxExtraInferenceRatio > 1) {
      throw new IllegalArgumentException("Invalid branch comparison budget");
    }
  }
}
