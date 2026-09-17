package com.epsilon.config.settings;

/**
 * 終了した対局の学習データを、一時ファイルへ保存するための設定。
 *
 * @param dir 完了済みデータ断片を書き込むディレクトリ
 * @param keepFragments 消費後もデータ断片ファイルを残すなら {@code true}
 * @param asyncQueueGames 書き込み処理待ちキューへ保持できる完了対局数
 * @param asyncWriters データ断片を並列保存する書き込み処理数
 */
@SettingsPrefix("epsilon.decision.train.spool")
public record DecisionTrainSpoolSettings(
    @Setting("dir") @Default("train-fragments") String dir,
    @Setting("keepFragments") @Default("false") boolean keepFragments,
    @Setting("asyncQueueGames") @Default("1024") @Positive int asyncQueueGames,
    @Setting("asyncWriters") @Default("4") @Positive int asyncWriters) {}
