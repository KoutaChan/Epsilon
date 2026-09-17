package com.epsilon.config.settings;

/**
 * モデル比較コマンド{@code eval-vs}の並列度と乱数シードの設定。評価時は最大確率の合法手を選ぶ。
 *
 * @param gamesInFlight 対戦環境が同時に進行させる対局数
 * @param maximumInferenceBatch 比較対局の各複製モデルが一度に評価する最大行数。通常推論の上限以下へ制限する
 * @param progressIntervalWallSeeds 固定した牌山による評価の進捗を通知する牌山間隔
 * @param seedBase 席替えを組み合わせた比較評価で使う牌山シード系列の先頭値
 */
@SettingsPrefix("epsilon.decision.evalVs")
public record DecisionEvalVsSettings(
    @Setting("gamesInFlight") @Default("32768") @Positive int gamesInFlight,
    @Setting("maximumInferenceBatch") @Default("2048") @Positive int maximumInferenceBatch,
    @Setting("progressIntervalWallSeeds") @Default("2048") @Positive int progressIntervalWallSeeds,
    @Setting("seedBase") @Default("40000") long seedBase) {}
