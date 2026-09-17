package com.epsilon.major.config.settings;

import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.Default;
import com.epsilon.config.settings.NonNegative;
import com.epsilon.config.settings.Positive;
import com.epsilon.config.settings.Range;
import com.epsilon.config.settings.Setting;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.config.settings.SettingsPrefix;

/**
 * Decision 推論サーバーの非同期循環バッファとバッチ処理設定です。
 *
 * <p>値の解決順序は {@link SettingsLoader} に従います。設定はスナップショットから生成し、頻繁に実行する処理ではレコード
 * 参照を渡します。推論入力は再利用可能なページ固定のバッファからデバイスへ転送し、実行枠ごとにH2D、順伝播、D2Hを
 * 重ねます。GPUの同期パイプラインと転送方式は選択できません。CPUは設定にかかわらず同期EAGER経路とFP32を使い、 チェックポイント パラメーターと連続値入力もFP32で保持します。
 *
 * @param maxBatch 単一遷移候補だけを持つ要求の最大バッチ行数
 * @param multiTransitionMaxBatch 複数遷移を持つ要求の最大バッチ行数
 * @param slotsPerDevice 各デバイスが持つ再利用可能な実行枠数。位置ごとにページ固定の入出力とデバイス作業領域を持つためVRAM使用量を決める
 * @param readyBatchesPerDevice デバイスへの割当て前にCPU側で保持する準備済みバッチ数の上限。元バッチの参照だけを保持し、追加VRAMを使わない
 * @param computePrecision ネットワーク順伝播の計算精度
 * @param frozenBfloat16Parameters 凍結パラメーターを演算精度と同じ低精度で保持するなら {@code true}
 * @param lowPrecisionInputNumerics 連続値入力も低精度へ変換するなら {@code true}
 * @param acceleratorGraph 未対応のグラフ実行設定。{@code false} のみ受け付ける
 * @param acceleratorGraphRows グラフ実行用の予約項目。現在はグラフ実行が無効なため使用しない
 * @param profileFile 推論計測結果を書き出すファイル。空文字列なら無効
 * @param profileMinimumRows 計測結果集計へ含める最小バッチ行数
 * @param profileAfterCalls 計測結果開始前にウォームアップとして除外する順伝播回数
 * @param profileTransitionCapacity 計測結果対象を絞る遷移容量。{@code 0} は制限なし
 */
@SettingsPrefix("epsilon.decision.inference")
public record DecisionInferenceSettings(
    @Setting("maxBatch") @Default("4096") @Positive int maxBatch,
    @Setting("multiTransitionMaxBatch") @Default("2048") @Positive int multiTransitionMaxBatch,
    @Setting("slotsPerDevice") @Default("2") @Range(min = 1, max = 4) int slotsPerDevice,
    @Setting("readyBatchesPerDevice") @Default("4") @Positive int readyBatchesPerDevice,
    @Setting("computePrecision") @Default("BFLOAT16") DecisionComputePrecision computePrecision,
    @Setting("frozenBfloat16Parameters") @Default("true") boolean frozenBfloat16Parameters,
    @Setting("lowPrecisionInputNumerics") @Default("true") boolean lowPrecisionInputNumerics,
    @Setting("acceleratorGraph") @Default("false") boolean acceleratorGraph,
    @Setting("acceleratorGraphRows") @Default("0") @NonNegative int acceleratorGraphRows,
    @Setting("profileFile") @Default("") String profileFile,
    @Setting("profileMinimumRows") @Default("512") @Positive int profileMinimumRows,
    @Setting("profileAfterCalls") @Default("200") @NonNegative int profileAfterCalls,
    @Setting("profileTransitionCapacity") @Default("0") @NonNegative
        int profileTransitionCapacity) {

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
