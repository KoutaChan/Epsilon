package com.epsilon.major.config.settings;

import com.epsilon.config.settings.Default;
import com.epsilon.config.settings.Positive;
import com.epsilon.config.settings.Range;
import com.epsilon.config.settings.Setting;
import com.epsilon.config.settings.SettingsPrefix;

/**
 * macro全体で固定する対戦相手の抽選と、履歴モデルの登録間隔。
 *
 * @param championProbability 現在のチャンピオンを選ぶ確率。残りは履歴から一様に抽選する
 * @param snapshotIntervalMacros 履歴へ登録する保存済みmacroの間隔
 */
@SettingsPrefix("epsilon.decision.train.opponents")
public record DecisionOpponentSettings(
    @Setting("championProbability") @Default("0.7") @Range(min = 0.0, max = 1.0)
        double championProbability,
    @Setting("snapshotIntervalMacros") @Default("10") @Positive int snapshotIntervalMacros) {

  public boolean shouldRegister(int completedMacros) {
    return completedMacros > 0 && completedMacros % snapshotIntervalMacros == 0;
  }
}
