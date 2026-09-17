package com.epsilon.config.settings;

/** 推論バッチの先頭行が到着してからの最大待機時間。容量が足りない場合は、容量が利用可能になり次第実行する。 */
@SettingsPrefix("epsilon.decision.inference")
public record InferenceBatchingSettings(@Default("1000000") long maxBatchWaitMicros) {
  public InferenceBatchingSettings {
    if (maxBatchWaitMicros < 0 || maxBatchWaitMicros > Long.MAX_VALUE / 1_000) {
      throw new IllegalArgumentException("maxBatchWaitMicros is outside the nanosecond range");
    }
  }

  public long maxBatchWaitNanos() {
    return maxBatchWaitMicros * 1_000;
  }
}
