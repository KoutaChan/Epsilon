package com.epsilon.config.settings;

/**
 * 自己対戦の学習データを、使用前に読み込んで準備する件数と並列度の設定。
 *
 * @param prefetch 学習処理より先に準備しておくデータ断片数
 * @param prefetchWorkers データ断片を並列復号するワーカー数
 */
@SettingsPrefix("epsilon.decision.train.streaming")
public record DecisionTrainStreamingSettings(
    @Setting("prefetch") @Default("3") @Positive int prefetch,
    @Setting("prefetchWorkers") @Default("4") @Positive int prefetchWorkers) {}
