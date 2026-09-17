package com.epsilon.nano.config.settings;

import com.epsilon.config.settings.Default;
import com.epsilon.config.settings.Positive;
import com.epsilon.config.settings.Range;
import com.epsilon.config.settings.Setting;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.config.settings.SettingsPrefix;

/**
 * Decision 事前学習の検証データの分割設定です。
 *
 * <p>設定の優先順位は {@link SettingsLoader} に従う。起動時に確定した設定からレコードを生成し、繰り返し実行する処理にはその参照を渡す。
 *
 * @param fraction 入力元ファイルのうち検証へ固定割当する比率
 * @param maximumDeviceBatchRows 検証の一回のデバイス順伝播へ載せる最大行数
 * @param minSamples チェックポイント評価を有効とみなすために必要な検証サンプル数
 */
@SettingsPrefix("epsilon.decision.pretrain.validation")
public record DecisionPretrainValidationSettings(
    @Setting("fraction") @Default("0.10") @Range(min = 0.0, max = 0.5) float fraction,
    @Setting("maximumDeviceBatchRows") @Default("512") @Positive int maximumDeviceBatchRows,
    @Setting("minSamples") @Default("1024") @Positive int minSamples) {

  /**
   * この設定項目に対応する、系列に同梱された既定値を返す。
   *
   * @return この系列の同梱既定値から生成した変更不可の設定
   */
  public static DecisionPretrainValidationSettings defaults() {
    return EpsilonSettings.defaults().bind(DecisionPretrainValidationSettings.class);
  }

  /** 指定された確定済み設定から、このレコードに対応する項目を読み込む。 */
  public static DecisionPretrainValidationSettings load(SettingsLoader settings) {
    return settings.bind(DecisionPretrainValidationSettings.class);
  }
}
