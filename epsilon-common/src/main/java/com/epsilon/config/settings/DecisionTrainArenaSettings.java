package com.epsilon.config.settings;

/**
 * 自己対戦による学習データ収集で、対戦相手と対局処理の並列度を指定する。
 *
 * @param gamesInFlight 対戦環境が同時に進行させる対局数
 * @param cohorts 同じ推論バッチ形成規則を共有する対局グループ数
 * @param asyncInferencePipelineEnabled 対局進行と推論を並行して処理するなら {@code true}
 * @param streamingSchedulerEnabled 対局が終わるたびに空いた枠へ次の対局を補充するなら {@code true}
 * @param advanceWorkers 対局状態をCPU上で進めるワーカー数
 * @param advanceTasksPerWorker 各ワーカーへ同時発行できる対局進行タスク数
 * @param maximumAdvanceTaskSize 一つの対局進行タスクへまとめる対局数の上限
 * @param maximumOpponentSnapshotsPerInterval 一学習区間で対戦相手として使う過去モデル数の上限
 */
@SettingsPrefix("epsilon.decision.train.arena")
public record DecisionTrainArenaSettings(
    @Setting("gamesInFlight") @Default("32768") @Positive int gamesInFlight,
    @Setting("cohorts") @Default("1") @Positive int cohorts,
    @Setting("asyncInferencePipelineEnabled") @Default("true")
        boolean asyncInferencePipelineEnabled,
    @Setting("streamingSchedulerEnabled") @Default("true") boolean streamingSchedulerEnabled,
    @Setting("advanceWorkers") @Default("32") @Range(min = 1, max = 32) int advanceWorkers,
    @Setting("advanceTasksPerWorker") @Default("2") @Positive int advanceTasksPerWorker,
    @Setting("maximumAdvanceTaskSize") @Default("64") @Positive int maximumAdvanceTaskSize,
    @Setting("maximumOpponentSnapshotsPerInterval") @Default("1") @Positive
        int maximumOpponentSnapshotsPerInterval) {}
