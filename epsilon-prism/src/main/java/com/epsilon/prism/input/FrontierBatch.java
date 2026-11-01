package com.epsilon.prism.input;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * 前線の入力の 1 塊。{@link FrontierField} の [遷移, {@link FrontierField#WIDTH}] の BF16 を、blob の先頭から 1 区画だけ置く（リトル
 * エンディアン）。同じ塊の {@link HostBatch} に付け、その計画（{@link BatchPlan}）の遷移の位置に書くので、行の並びと遷移の番号は網の入力と
 * 同じになる。H2D は blob の先頭から {@link #bytes} B を 1 回送る。
 *
 * <p>blob は外から渡す {@link ByteBuffer}。符号化のスレッドは行ごとに
 * 別の位置へ絶対位置で書くので、複数のスレッドが同じ塊の別の行を同時に書いてよい。
 */
public final class FrontierBatch {

  /** 1 遷移のバイト数。 */
  public static final int TRANSITION_BYTES = FrontierField.WIDTH * FrontierField.ELEMENT.bytes();

  private final HostBatch host;
  private final ByteBuffer blob;

  /**
   * 塊を作る。
   *
   * @param host 付ける網の入力の塊（行と遷移の位置をその計画から読む）
   * @param blob 前線を書く領域（{@link #maxBytes} 以上。この塊が持ち、バイト順をリトルエンディアンにする）
   * @throws IllegalArgumentException 領域が足りないとき
   */
  public FrontierBatch(HostBatch host, ByteBuffer blob) {
    this.host = Objects.requireNonNull(host, "host");
    int required = maxBytes(host.plan().capacity());
    if (blob.capacity() < required) {
      throw new IllegalArgumentException(
          "blob holds " + blob.capacity() + " bytes but the capacity needs " + required);
    }
    this.blob = blob.order(ByteOrder.LITTLE_ENDIAN);
  }

  /**
   * 上限の遷移で詰めたときの blob の大きさを返す。
   *
   * @param capacity 1 塊の上限
   * @return バイト数
   */
  public static int maxBytes(BatchPlan.Capacity capacity) {
    return Math.multiplyExact(capacity.transitions(), TRANSITION_BYTES);
  }

  /**
   * 付けた網の入力の塊を返す。
   *
   * @return 塊
   */
  public HostBatch host() {
    return host;
  }

  /**
   * 前線の領域を返す。位置（position・limit）は使わず、書き手は絶対位置で書く。
   *
   * @return blob
   */
  public ByteBuffer blob() {
    return blob;
  }

  /**
   * 送るバイト数を返す。
   *
   * @return 計画の遷移の数 × {@link #TRANSITION_BYTES}
   * @throws IllegalStateException 計画を封じる前のとき
   */
  public int bytes() {
    BatchPlan plan = host.plan();
    if (!plan.sealed()) {
      throw new IllegalStateException("batch plan is not sealed");
    }
    return plan.count(InputSchema.Axis.TRANSITION) * TRANSITION_BYTES;
  }
}
