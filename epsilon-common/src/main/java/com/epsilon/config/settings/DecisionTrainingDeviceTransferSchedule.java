package com.epsilon.config.settings;

/**
 * Decision 学習処理がホストバッチをデバイスへ転送するタイミングです。
 *
 * <p>{@link DecisionTensorTransfer} はホスト側の転送バッファ形式を選び、この列挙型はデバイス転送を学習計算へ重ねるかを選びます。
 * 実行環境によって暗黙に方式を変えず、設定した経路をそのまま使用します。
 */
public enum DecisionTrainingDeviceTransferSchedule {
  /** バッチを使用する直前にデバイスへ転送する、単純な基準経路です。 */
  ON_DEMAND,

  /**
   * 次のバッチを専用ストリームで先行転送し、現在の学習計算と重ねる経路です。
   *
   * <p>{@code PINNED_BUFFER}とPyTorchのGPU学習ワーカーが必須で、条件を満たさない場合は起動時に直ちに例外を送出します。
   */
  OVERLAPPED
}
