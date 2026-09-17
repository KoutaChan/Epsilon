package com.epsilon.config.settings;

/**
 * GRP モデル全体の学習・モデルサイズ設定です。
 *
 * <p>{@link SettingsLoader}から生成した不変レコードを実行経路へ渡します。
 *
 * @param enabled GRP推論・学習経路を有効にするなら {@code true}
 * @param hidden GRP ネットワークの隠れ層の幅
 * @param layers GRPの共通ネットワークの層数
 * @param batchSize 1 オプティマイザーステップへ渡す学習行数
 * @param learningRate オプティマイザーの初期学習率
 * @param weightDecay オプティマイザーへ適用する重み減衰係数
 * @param gradClip 全勾配のノルムの上限
 */
@SettingsPrefix("epsilon.grp")
public record GrpSettings(
    @Setting("enabled") @Default("true") boolean enabled,
    @Setting("hidden") @Default("64") @Positive int hidden,
    @Setting("layers") @Default("2") @Positive int layers,
    @Setting("batchSize") @Default("256") @Positive int batchSize,
    @Setting("learningRate") @Default("2e-4") @Positive float learningRate,
    @Setting("weightDecay") @Default("1e-4") @NonNegative float weightDecay,
    @Setting("gradClip") @Default("0.5") @Positive float gradClip) {}
