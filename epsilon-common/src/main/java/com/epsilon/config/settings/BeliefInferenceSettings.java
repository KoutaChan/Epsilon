package com.epsilon.config.settings;

/**
 * Belief 推論のバッチ化設定です。
 *
 * <p>{@link SettingsLoader}から生成した不変レコードを実行経路へ渡します。
 *
 * @param maxBatch 1回のBelief 順伝播へまとめる最大局面数
 */
@SettingsPrefix("epsilon.belief.inference")
public record BeliefInferenceSettings(@Setting("maxBatch") @Default("64") @Positive int maxBatch) {}
