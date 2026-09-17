package com.epsilon.config.settings;

/**
 * 各推論デバイスに独立した複製モデルを配置するかを指定する。
 *
 * @param enabled 各推論デバイスに独立した複製モデルを配置するなら {@code true}
 */
@SettingsPrefix("epsilon.decision.inference.replicas")
public record DecisionInferenceReplicasSettings(
    @Setting("enabled") @Default("true") boolean enabled) {}
