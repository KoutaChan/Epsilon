package com.epsilon.prism.input;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * 1 塊の網の入力。UINT8 の blob 1 本に、{@link InputSchema.Tensor} の順の区画を {@link BatchPlan} の位置で置く（リトルエンディアン）。H2D は
 * blob の先頭から {@link #bytes} B を 1 回送る。
 *
 * <p>blob は外から渡す {@link ByteBuffer}（ダイレクトの領域か、ページ固定の領域）。符号化のスレッドは行ごとに別の
 * 位置へ絶対位置で書くので、複数のスレッドが同じ塊の別の行を同時に書いてよい（{@link InputWriter} はスレッドごとに blob の複製を持つ）。
 *
 * <p>blob のほかに、送らないホストだけの配列として、手ごとの規則事前 p0 の特徴（打牌の後の向聴・受け入れ。{@link PriorFeatures}）を持つ。
 */
public final class HostBatch {

  private final ByteBuffer blob;
  private final BatchPlan plan;
  private final float[] priorShanten;
  private final float[] priorUkeire;

  /**
   * 塊を作る。
   *
   * @param capacity 1 塊の上限
   * @param blob 入力を書く領域（{@link BatchPlan#maxBytes} 以上。この塊が持ち、バイト順をリトルエンディアンにする）
   * @throws IllegalArgumentException 領域が足りないとき
   */
  public HostBatch(BatchPlan.Capacity capacity, ByteBuffer blob) {
    plan = new BatchPlan(capacity);
    int required = BatchPlan.maxBytes(capacity);
    if (blob.capacity() < required) {
      throw new IllegalArgumentException(
          "blob holds " + blob.capacity() + " bytes but the capacity needs " + required);
    }
    this.blob = blob.order(ByteOrder.LITTLE_ENDIAN);
    priorShanten = new float[capacity.actions()];
    priorUkeire = new float[capacity.actions()];
  }

  /**
   * 行の位置と区画の位置を返す。
   *
   * @return 計画
   */
  public BatchPlan plan() {
    return plan;
  }

  /**
   * 入力の領域を返す。位置（position・limit）は使わず、書き手は絶対位置で書く。
   *
   * @return blob
   */
  public ByteBuffer blob() {
    return blob;
  }

  /**
   * 送るバイト数を返す。
   *
   * @return 計画で使う blob の大きさ
   */
  public int bytes() {
    return plan.bytes();
  }

  /**
   * 手の打牌の後の向聴を、p0 が読む float の値で返す（打牌でない手は 0）。
   *
   * @param action 塊の中の手の番号
   * @return 向聴
   */
  public float priorShanten(int action) {
    return priorShanten[Objects.checkIndex(action, plan.count(InputSchema.Axis.ACTION))];
  }

  /**
   * 手の打牌の後の受け入れの生きた枚数を、p0 が読む float の値で返す（打牌でない手は 0）。
   *
   * @param action 塊の中の手の番号
   * @return 受け入れの枚数
   */
  public float priorUkeire(int action) {
    return priorUkeire[Objects.checkIndex(action, plan.count(InputSchema.Axis.ACTION))];
  }

  /** p0 の特徴を書く（{@link InputWriter} だけが呼ぶ）。 */
  void prior(int action, float shanten, float ukeire) {
    priorShanten[action] = shanten;
    priorUkeire[action] = ukeire;
  }
}
