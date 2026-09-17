package com.epsilon.config.settings;

import com.epsilon.ai.decision.DecisionSelectionMode;

/**
 * 選択した行動だけを使う方策勾配学習で、自己対戦中の行動選択方式を指定する。
 *
 * @param selectionMode 自己対戦で行動を選ぶ方策。学習用の自己対戦データ収集では通常 {@link DecisionSelectionMode#FULL_SUPPORT}
 */
@SettingsPrefix("epsilon.decision.rollout")
public record DecisionRolloutSettings(
    @Setting("selectionMode") @Default("FULL_SUPPORT") DecisionSelectionMode selectionMode) {}
