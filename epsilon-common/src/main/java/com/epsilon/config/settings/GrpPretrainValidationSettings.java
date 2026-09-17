package com.epsilon.config.settings;

/**
 * GRPの事前学習で、検証用データの割合と必要な件数を指定する。
 *
 * @param fraction 入力元対局のうち検証へ固定割当する比率
 * @param minExamples 検証指標を確定するために必要なサンプル数
 * @param minGames 検証指標を確定するために必要な対局数
 */
@SettingsPrefix("epsilon.grp.pretrain.validation")
public record GrpPretrainValidationSettings(
    @Setting("fraction") @Default("0.10") @Range(min = 0.0, max = 0.5) float fraction,
    @Setting("minExamples") @Default("64") @Positive int minExamples,
    @Setting("minGames") @Default("8") @Positive int minGames) {}
