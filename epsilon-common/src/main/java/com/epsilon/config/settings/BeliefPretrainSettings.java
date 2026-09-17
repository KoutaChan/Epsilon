package com.epsilon.config.settings;

/**
 * 牌譜を使うBeliefモデルの事前学習で、データの読み込み量を指定する設定。
 *
 * @param logChunkFiles 一度に読み込む牌譜ファイル数
 * @param logSampleChunkSize 牌譜変換中にまとめて保持する最大サンプル数
 * @param batchSize 1 オプティマイザーステップへ渡す学習行数
 */
@SettingsPrefix("epsilon.belief.pretrain")
public record BeliefPretrainSettings(
    @Setting("logChunkFiles") @Default("8") @Positive int logChunkFiles,
    @Setting("logSampleChunkSize") @Default("4096") @Positive int logSampleChunkSize,
    @Setting("batchSize") @Default("256") @Positive int batchSize) {}
