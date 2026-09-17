package com.epsilon.config.settings;

/**
 * 公開情報から他家状態を推定するBeliefモデルの学習設定。
 *
 * @param hidden Belief ネットワークの隠れ層の幅
 * @param learningRate オプティマイザーの初期学習率
 * @param weightDecay オプティマイザーへ適用する重み減衰係数
 * @param gradClip 全勾配のノルムの上限
 */
@SettingsPrefix("epsilon.belief")
public record BeliefSettings(
    @Setting("hidden") @Default("128") @Positive int hidden,
    @Setting("learningRate") @Default("2e-4") @Positive float learningRate,
    @Setting("weightDecay") @Default("1e-4") @NonNegative float weightDecay,
    @Setting("gradClip") @Default("0.5") @Positive float gradClip) {}
