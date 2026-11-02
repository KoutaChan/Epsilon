package com.epsilon.prism.input;

import com.epsilon.prism.model.PrismShape;
import java.nio.ByteBuffer;
import java.util.Objects;
import java.util.function.IntFunction;

/**
 * 1 回の順伝播のホストの入力（学習の微小バッチと推論の塊で使い回す）。基準の入力の塊と、前線を使う網（{@link
 * PrismShape#frontier()}）なら同じ塊に付けた前線の塊を持つ。前線も塊の計画の遷移の位置に書くので、行と遷移の並びは
 * 基準の入力と同じ。
 *
 * @param input 基準の入力（Tokenizer が書く）
 * @param frontier 前線の塊（{@link FrontierFeatures} が書く。前線を使わない網は null）
 */
public record HostInputs(HostBatch input, FrontierBatch frontier) {

  /**
   * 組を確かめる。
   *
   * @throws IllegalArgumentException 前線の塊が別の塊に付いているとき
   */
  public HostInputs {
    Objects.requireNonNull(input, "input");
    if (frontier != null && frontier.host() != input) {
      throw new IllegalArgumentException("the frontier batch belongs to another host batch");
    }
  }

  /**
   * 塊を取る。
   *
   * @param capacity 1 塊の上限
   * @param frontier 前線を使う網なら true
   * @param memory バイト数から blob の領域を取る（ダイレクトの領域か、デバイスが読む塊ならページ固定の領域）
   * @return 塊
   */
  public static HostInputs allocate(
      BatchPlan.Capacity capacity, boolean frontier, IntFunction<ByteBuffer> memory) {
    HostBatch input = new HostBatch(capacity, memory.apply(BatchPlan.maxBytes(capacity)));
    return new HostInputs(
        input,
        frontier ? new FrontierBatch(input, memory.apply(FrontierBatch.maxBytes(capacity))) : null);
  }

  /**
   * 塊の計画を返す。
   *
   * @return 基準の入力の計画（前線の塊も同じ計画を読む）
   */
  public BatchPlan plan() {
    return input.plan();
  }
}
