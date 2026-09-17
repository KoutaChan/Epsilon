package com.epsilon.pico.config.settings;

import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.Default;
import com.epsilon.config.settings.NonNegative;
import com.epsilon.config.settings.Positive;
import com.epsilon.config.settings.Range;
import com.epsilon.config.settings.Setting;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.config.settings.SettingsPrefix;

/**
 * Decision 推論のバッチ処理と、非同期で再利用する実行枠を設定する。
 *
 * <p>設定の優先順位は {@link SettingsLoader} に従う。起動時に確定した設定を共有し、実行中に読み直さない。GPU
 * ではページロックしたバッファを使い、ホストからの転送、順伝播、ホストへの結果転送を実行枠ごとに並行して進める。
 *
 * <p>CPU では設定にかかわらず同期の逐次実行と FP32 を使い、パラメータと数値入力も FP32 で保持する。計算グラフを記録する方式は現在サポートしない。
 *
 * @param maxBatch 単一遷移候補だけを持つ要求の最大バッチ行数
 * @param multiTransitionMaxBatch 複数遷移を持つ要求の最大バッチ行数
 * @param slotsPerDevice 各デバイスが持つ再利用可能な実行枠数。実行枠ごとにページロックした入出力とデバイス作業領域を持つためVRAM使用量を決める
 * @param readyBatchesPerDevice デバイスへの割当て前にCPU側で保持する準備済みバッチ数の上限。元バッチの参照だけを保持し、追加VRAMを使わない
 * @param computePrecision ネットワーク順伝播の計算精度
 * @param frozenBfloat16Parameters 凍結パラメーターを演算精度と同じ低精度で保持するなら {@code true}
 * @param lowPrecisionInputNumerics 連続値入力も低精度へ変換するなら {@code true}
 * @param acceleratorGraph 計算グラフを記録する設定。現在は未対応のため {@code false} のみ許可する
 * @param acceleratorGraphRows 計算グラフの記録用に予約した行数。記録機能は現在未対応
 */
@SettingsPrefix("epsilon.decision.inference")
public record DecisionInferenceSettings(
    @Setting("maxBatch") @Default("4096") @Positive int maxBatch,
    @Setting("multiTransitionMaxBatch") @Default("2048") @Positive int multiTransitionMaxBatch,
    @Setting("slotsPerDevice") @Default("3") @Range(min = 1, max = 4) int slotsPerDevice,
    @Setting("readyBatchesPerDevice") @Default("6") @Positive int readyBatchesPerDevice,
    @Setting("computePrecision") @Default("BFLOAT16") DecisionComputePrecision computePrecision,
    @Setting("frozenBfloat16Parameters") @Default("true") boolean frozenBfloat16Parameters,
    @Setting("lowPrecisionInputNumerics") @Default("true") boolean lowPrecisionInputNumerics,
    @Setting("acceleratorGraph") @Default("false") boolean acceleratorGraph,
    @Setting("acceleratorGraphRows") @Default("0") @NonNegative int acceleratorGraphRows) {

  /** バッチ上限と非同期循環バッファ資源数の相関制約を検証する。 */
  public DecisionInferenceSettings {
    if (multiTransitionMaxBatch > maxBatch) {
      throw new IllegalArgumentException("multiTransitionMaxBatch must not exceed maxBatch");
    }
    if (readyBatchesPerDevice < slotsPerDevice) {
      throw new IllegalArgumentException("readyBatchesPerDevice must be at least slotsPerDevice");
    }
    if (acceleratorGraph) {
      throw new IllegalArgumentException("Decision inference does not support acceleratorGraph");
    }
  }

  /**
   * この設定項目に対応する、系列に同梱された既定値を返す。
   *
   * @return この系列の同梱既定値から生成した変更不可の設定
   */
  public static DecisionInferenceSettings defaults() {
    return EpsilonSettings.defaults().bind(DecisionInferenceSettings.class);
  }

  /** 指定された確定済み設定から、このレコードに対応する項目を読み込む。 */
  public static DecisionInferenceSettings load(SettingsLoader settings) {
    return settings.bind(DecisionInferenceSettings.class);
  }
}
