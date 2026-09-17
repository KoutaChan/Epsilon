package com.epsilon.config.settings;

/**
 * Decision の自己対戦による学習の基本設定。
 *
 * <p>{@link SettingsLoader}から生成した不変レコードを実行経路へ渡します。
 *
 * @param sampleLimit 1回の学習更新へ投入するサンプル行数の上限
 * @param tensorTransfer ホストから学習処理デバイスへ入力を転送する方式
 * @param deviceTransferSchedule ホストバッチをデバイスへ転送するタイミング
 * @param computePrecision 順伝播の自動混合精度（autocast）で使う精度。損失・パラメーター・勾配・オプティマイザー状態はFLOAT32で扱う
 */
@SettingsPrefix("epsilon.decision.train")
public record DecisionTrainSettings(
    @Setting("sampleLimit") @Default("8192") @Positive int sampleLimit,
    @Setting("tensorTransfer") @Default("PINNED_BUFFER") DecisionTensorTransfer tensorTransfer,
    @Setting("deviceTransferSchedule") @Default("ON_DEMAND")
        DecisionTrainingDeviceTransferSchedule deviceTransferSchedule,
    @Setting("computePrecision") @Default("BFLOAT16") DecisionComputePrecision computePrecision) {

  public DecisionTrainSettings {
    if (deviceTransferSchedule == DecisionTrainingDeviceTransferSchedule.OVERLAPPED
        && tensorTransfer != DecisionTensorTransfer.PINNED_BUFFER) {
      throw new IllegalArgumentException(
          "OVERLAPPED Decision training requires tensorTransfer=PINNED_BUFFER");
    }
  }
}
