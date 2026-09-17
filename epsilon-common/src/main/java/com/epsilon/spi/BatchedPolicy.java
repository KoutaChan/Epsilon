package com.epsilon.spi;

import java.util.List;
import java.util.concurrent.CompletableFuture;

/** テンソルやモデル構造を公開せず、入力行順の合法手位置を返す。 */
public interface BatchedPolicy extends AutoCloseable {
  /** 実行環境内だけで比較する分類キー。待機中は equals/hashCode が変化しない値を返す。 */
  Object batchKey(DecisionRequest request);

  int maxBatchSize(Object key);

  /** 符号化より前に容量を予約する。満杯なら null とし、容量が利用可能になる際に通知する。 実装は予約と通知登録の競合を処理する。通知は任意のスレッドから実行できる。 */
  Ingress tryAcquire(Object key, Runnable capacityAvailable);

  /** 実行可能なデバイスへ次のバッチを供給する必要がある場合にtrueを返す。 */
  default boolean needsInferenceWork() {
    return true;
  }

  /** 任意の累積バッチ計測。GPU カーネル時間ではなく、ホストで観測した各段階の経過時間です。 */
  default Timings timings() {
    return Timings.UNAVAILABLE;
  }

  record Timings(
      boolean available,
      long batches,
      long encoderQueueNanos,
      long encodingNanos,
      long inferenceRoundTripNanos) {
    public static final Timings UNAVAILABLE = new Timings(false, 0, 0, 0, 0);
  }

  /** バッチ固有の予約。submit に成功すると予約の所有権は完了処理へ移る。 */
  interface Ingress extends AutoCloseable {
    CompletableFuture<int[]> submit(List<DecisionRequest> requests);

    /** 未提出の予約だけを解放する。提出後の呼び出しは何もしない。 */
    @Override
    void close();
  }

  @Override
  void close();
}
