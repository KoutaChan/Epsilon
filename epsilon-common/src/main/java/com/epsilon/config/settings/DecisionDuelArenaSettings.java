package com.epsilon.config.settings;

/**
 * 同じ牌山を使うモデル比較で、対局の進行と推論バッチの作成を並列化するための設定。
 *
 * <p>学習用の行動履歴や価値の教師データは生成せず、行動選択が完了するたびに対局を進める。ここではCPU処理の並列度を指定し、推論バッチの選択には共通の設定を使う。
 *
 * @param advanceWorkers GameEngineを並列に進めるワーカー数
 * @param advanceTasksPerWorker 各ワーカーへ同時発行する対局進行タスク数
 * @param maximumAdvanceTaskSize 一つの対局進行タスクへまとめるゲーム数上限
 */
@SettingsPrefix("epsilon.decision.evalVs.arena")
public record DecisionDuelArenaSettings(
    @Setting("advanceWorkers") @Default("32") @Range(min = 1, max = 64) int advanceWorkers,
    @Setting("advanceTasksPerWorker") @Default("2") @Positive int advanceTasksPerWorker,
    @Setting("maximumAdvanceTaskSize") @Default("64") @Positive int maximumAdvanceTaskSize) {}
