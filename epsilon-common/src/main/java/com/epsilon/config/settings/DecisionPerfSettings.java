package com.epsilon.config.settings;

/**
 * 収集・学習・評価などの処理の切り替え時に、GPUメモリの使用状況を記録する設定。
 *
 * <p>{@link SettingsLoader}から生成した不変レコードを実行経路へ渡します。
 *
 * @param memorySnapshots 処理の切り替え時に、GPUメモリの使用量とストリーム別の割り当て状況を出力するなら {@code true}
 */
@SettingsPrefix("epsilon.decision.perf")
public record DecisionPerfSettings(
    @Setting("memorySnapshots") @Default("false") boolean memorySnapshots) {}
