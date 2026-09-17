package com.epsilon.config.settings;

/**
 * 自己対戦の相手として保持する、過去のモデルの保存数を指定する。
 *
 * @param max 対戦相手として保持する過去モデル数の上限。{@code 0} なら保持しない
 */
@SettingsPrefix("epsilon.decision.snapshotPool")
public record DecisionSnapshotPoolSettings(@Setting("max") @Default("32") int max) {}
