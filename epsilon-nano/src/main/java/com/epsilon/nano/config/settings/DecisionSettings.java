package com.epsilon.nano.config.settings;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.Default;
import com.epsilon.config.settings.NonNegative;
import com.epsilon.config.settings.Positive;
import com.epsilon.config.settings.Setting;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.config.settings.SettingsPrefix;

/**
 * Decision ネットワークと学習オプティマイザーの基礎設定です。
 *
 * <p>設定の優先順位は {@link SettingsLoader} に従う。起動時に確定した設定からレコードを生成し、繰り返し実行する処理にはその参照を渡す。
 *
 * @param learningRate 自己対局学習 Decision オプティマイザーの初期学習率
 * @param weightDecay オプティマイザーへ適用する重み減衰係数
 * @param gradClip 全勾配のノルムの上限
 * @param hidden Decision ネットワークの共通表現幅
 * @param utilityProfile 価値教師値へ使う対局効用定義
 */
@SettingsPrefix("epsilon.decision")
public record DecisionSettings(
    @Setting("learningRate") @Default("2e-5") @Positive float learningRate,
    @Setting("weightDecay") @Default("1e-4") @NonNegative float weightDecay,
    @Setting("gradClip") @Default("0.5") @Positive float gradClip,
    @Setting("hidden") @Default("256") @Positive int hidden,
    @Setting("utilityProfile") @Default("TENHOU") EpsilonUtilityProfile utilityProfile) {

  /**
   * この設定項目に対応する、系列に同梱された既定値を返す。
   *
   * @return この系列の同梱既定値から生成した変更不可の設定
   */
  public static DecisionSettings defaults() {
    return EpsilonSettings.defaults().bind(DecisionSettings.class);
  }

  /** 指定された確定済み設定から、このレコードに対応する項目を読み込む。 */
  public static DecisionSettings load(SettingsLoader settings) {
    return settings.bind(DecisionSettings.class);
  }
}
