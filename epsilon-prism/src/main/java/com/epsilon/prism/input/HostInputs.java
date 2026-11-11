package com.epsilon.prism.input;

import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.function.IntFunction;

/**
 * 1 回の順伝播に使うホスト側の入力（学習のマイクロバッチと推論のバッチで使い回す）。
 *
 * @param input 入力のバッチ（入力の形式が作る 1 行のライターが書く）
 */
public record HostInputs(HostBatch input) {

  public HostInputs {
    Objects.requireNonNull(input, "input");
  }

  /**
   * 入力のバッチを確保する。
   *
   * @param format 入力の形式
   * @param capacity 1 つのバッチの上限
   * @param memory バイト数を受け取り、blob のバッファを確保する関数（ダイレクトバッファか、デバイスが読むバッチならページ固定のメモリ）
   * @return 確保した入力
   */
  public static HostInputs allocate(InputFormat format, BatchPlan.Capacity capacity, IntFunction<ByteBuffer> memory) {
    return new HostInputs(new HostBatch(format, capacity, memory.apply(BatchPlan.maxBytes(format, capacity))));
  }

  /**
   * バッチの計画を返す。
   *
   * @return 入力のバッチの計画
   */
  public BatchPlan plan() {
    return input.plan();
  }
}
