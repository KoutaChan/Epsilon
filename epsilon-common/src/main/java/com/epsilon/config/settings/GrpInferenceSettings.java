package com.epsilon.config.settings;

/**
 * GRP 推論のバッチ化設定です。
 *
 * <p>{@link SettingsLoader}から生成した不変レコードを実行経路へ渡します。
 *
 * @param maxBatch 1回のGRP 順伝播へまとめる最大局面数
 * @param asyncBatchingEnabled 呼び出し元以外のスレッドで要求をバッチ化するなら {@code true}
 * @param coalesceWaitMicros 要求をバッチへ合流させる最大待機時間（マイクロ秒）
 */
@SettingsPrefix("epsilon.grp.inference")
public record GrpInferenceSettings(
    @Setting("maxBatch") @Default("64") @Positive int maxBatch,
    @Setting("asyncBatchingEnabled") @Default("true") boolean asyncBatchingEnabled,
    @Setting("coalesceWaitMicros") @Default("0") @NonNegative int coalesceWaitMicros) {}
