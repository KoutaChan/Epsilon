package com.epsilon.pico.config.settings;

import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.config.settings.Default;
import com.epsilon.config.settings.NonNegative;
import com.epsilon.config.settings.Positive;
import com.epsilon.config.settings.Setting;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.config.settings.SettingsPrefix;

/**
 * Decision の牌譜による事前学習で、データの読み込みと実行計画を設定する。
 *
 * <p>設定の優先順位は {@link SettingsLoader} に従う。起動時に確定した設定からレコードを生成し、繰り返し実行する処理にはその参照を渡す。
 *
 * @param compilerSampleBufferRows データセットの変換処理が並べ替え用に保持するサンプル行数
 * @param compilerReaderWorkers 入力元牌譜を並列に読むワーカー数
 * @param compilerPrefetchFiles 読み取り処理へ先読みする入力元ファイル数
 * @param optimizerBatchRows 1 パラメーター更新を構成する全デバイス合計行数
 * @param maximumDeviceBatchRows 1回のデバイス順伝播へ載せる最大行数
 * @param tensorTransfer ホストからデバイスへ入力を転送する方式
 * @param computePrecision 順伝播/入力の計算精度。BFLOAT16では実行パラメーター/勾配にも適用し、更新はFLOAT32 更新用に保持するパラメータで行う
 * @param datasetRowsPerShard 学習用に変換したデータセットの1 分割データ当たり行数
 * @param datasetPrefetchShards 学習中に先読みする学習用に変換した分割データ数
 * @param minimumRowsPerDataParallelDevice データ並列で1 デバイスへ割り当てる最小行数
 * @param progressLogIntervalSeconds 進捗指標を出力する間隔（秒）
 * @param learningRate オプティマイザーの初期学習率
 * @param lrDecayPerEpoch エポック終了ごとに学習率へ乗じる係数
 * @param behaviorCloningCoef 選択行動の負の対数尤度に掛ける損失係数
 * @param valueCoef HL-Gauss 価値交差エントロピーの係数
 * @param riichiDecisionWeight RIICHI対DAMA サンプルの方策損失重み
 * @param reactionDecisionWeight PASS・鳴き・RON応答サンプルの方策損失重み
 */
@SettingsPrefix("epsilon.decision.pretrain")
public record DecisionPretrainSettings(
    @Setting("compilerSampleBufferRows") @Default("16384") @Positive int compilerSampleBufferRows,
    @Setting("compilerReaderWorkers") @Default("8") @Positive int compilerReaderWorkers,
    @Setting("compilerPrefetchFiles") @Default("32") @Positive int compilerPrefetchFiles,
    @Setting("optimizerBatchRows") @Default("4096") @Positive int optimizerBatchRows,
    @Setting("maximumDeviceBatchRows") @Default("2048") @Positive int maximumDeviceBatchRows,
    @Setting("tensorTransfer") @Default("PINNED_BUFFER") DecisionTensorTransfer tensorTransfer,
    @Setting("computePrecision") @Default("BFLOAT16") DecisionComputePrecision computePrecision,
    @Setting("datasetRowsPerShard") @Default("32768") @Positive int datasetRowsPerShard,
    @Setting("datasetPrefetchShards") @Default("2") @Positive int datasetPrefetchShards,
    @Setting("minimumRowsPerDataParallelDevice") @Default("256") @Positive
        int minimumRowsPerDataParallelDevice,
    @Setting("progressLogIntervalSeconds") @Default("60") @Positive int progressLogIntervalSeconds,
    @Setting("learningRate") @Default("2e-4") @Positive float learningRate,
    @Setting("lrDecayPerEpoch") @Default("1.0") @Positive float lrDecayPerEpoch,
    @Setting("behaviorCloningCoef") @Default("1.0") @NonNegative float behaviorCloningCoef,
    @Setting("valueCoef") @Default("0.5") @NonNegative float valueCoef,
    @Setting("riichiDecisionWeight") @Default("2.0") @NonNegative float riichiDecisionWeight,
    @Setting("reactionDecisionWeight") @Default("2.0") @NonNegative float reactionDecisionWeight) {

  /** 方策または価値の少なくとも一方に正の損失係数があることを検証する。 */
  public DecisionPretrainSettings {
    if (behaviorCloningCoef == 0.0f && valueCoef == 0.0f) {
      throw new IllegalArgumentException("Decision pretrain requires a policy or value objective");
    }
  }

  /**
   * この設定項目に対応する、系列に同梱された既定値を返す。
   *
   * @return この系列の同梱既定値から生成した変更不可の設定
   */
  public static DecisionPretrainSettings defaults() {
    return EpsilonSettings.defaults().bind(DecisionPretrainSettings.class);
  }

  /**
   * 性能測定でバッチと転送方式だけを差し替え、損失/データ設定を実運用値に揃える。
   *
   * @param optimizerBatchRows 性能測定するオプティマイザーバッチ行数
   * @param deviceBatchRows 性能測定する1 デバイス順伝播の最大行数
   * @param transfer 性能測定するホスト-デバイス転送方式
   * @return 指定3項目だけを置き換えた設定
   */
  public DecisionPretrainSettings forBenchmark(
      int optimizerBatchRows, int deviceBatchRows, DecisionTensorTransfer transfer) {
    return SettingsLoader.validate(
        new DecisionPretrainSettings(
            compilerSampleBufferRows,
            compilerReaderWorkers,
            compilerPrefetchFiles,
            optimizerBatchRows,
            deviceBatchRows,
            transfer,
            computePrecision,
            datasetRowsPerShard,
            datasetPrefetchShards,
            minimumRowsPerDataParallelDevice,
            progressLogIntervalSeconds,
            learningRate,
            lrDecayPerEpoch,
            behaviorCloningCoef,
            valueCoef,
            riichiDecisionWeight,
            reactionDecisionWeight));
  }

  /**
   * 進捗ログ間隔を {@link System#nanoTime()} と同じ単位で返す。
   *
   * @return ログ間隔（ナノ秒）
   */
  public long progressLogIntervalNanos() {
    return progressLogIntervalSeconds * 1_000_000_000L;
  }

  /** 指定された確定済み設定から、このレコードに対応する項目を読み込む。 */
  public static DecisionPretrainSettings load(SettingsLoader settings) {
    return settings.bind(DecisionPretrainSettings.class);
  }
}
