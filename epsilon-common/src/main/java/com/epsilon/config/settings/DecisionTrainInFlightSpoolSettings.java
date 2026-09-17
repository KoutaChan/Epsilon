package com.epsilon.config.settings;

/**
 * 終了していない対局のデータを、一時ファイルへ保存するための設定。
 *
 * <p>{@code dirs}には保存先ディレクトリをカンマ区切りで指定する。複数の保存先がある場合は書き込み処理を順番に割り当て、一つの保存先にデータが集中することを防ぐ。
 *
 * @param dirs 未完対局データ本体を書き込むディレクトリのカンマ区切り一覧
 * @param asyncQueuePayloads 書き込み処理待ちキューへ保持できるデータ本体数
 * @param asyncWriters データ本体を並列保存する書き込み処理数
 * @param compressionLevel データ本体のZstandard圧縮レベル
 */
@SettingsPrefix("epsilon.decision.train.inFlightSpool")
public record DecisionTrainInFlightSpoolSettings(
    @Setting("dirs") @Default("train-inflight") String dirs,
    @Setting("asyncQueuePayloads") @Default("1024") @Positive int asyncQueuePayloads,
    @Setting("asyncWriters") @Default("4") @Positive int asyncWriters,
    @Setting("compressionLevel") @Default("1") int compressionLevel) {}
