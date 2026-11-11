package com.epsilon.prism.input;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

/**
 * 1 バッチ分のネットワークの入力。UINT8 の blob 1 つに、{@link InputBlob.Tensor} の順に各テンソルの領域を {@link BatchPlan} が決めた位置に
 * 置く（リトルエンディアン）。H2D の転送は、blob の先頭から {@link #bytes} B を 1 回で送る。
 *
 * <p>blob は外から渡す {@link ByteBuffer}（ダイレクトバッファか、ページ固定のメモリ）。符号化のスレッドは行ごとに別の位置へ絶対位置で
 * 書くので、複数のスレッドが同じバッチの別の行を同時に書いてよい（入力のライターは、中身を共有する blob の複製をスレッドごとに持つ）。
 *
 * <p>blob のほかに、デバイスへは送らないホストだけの配列として、合法手ごとのルールベースの事前分布 p0 の特徴（打牌の後の向聴と受け入れ。
 * {@link PriorFeatures}）を持つ。
 */
public final class HostBatch {

  private final ByteBuffer blob;
  private final BatchPlan plan;
  private final float[] priorShanten;
  private final float[] priorUkeire;

  /**
   * バッチを作る。
   *
   * @param format 入力の形式
   * @param capacity 1 つのバッチの上限
   * @param blob 入力を書くバッファ（{@link BatchPlan#maxBytes} 以上。このバッチが持ち、バイト順をリトルエンディアンにする）
   * @throws IllegalArgumentException バッファが足りないとき
   */
  public HostBatch(InputFormat format, BatchPlan.Capacity capacity, ByteBuffer blob) {
    plan = new BatchPlan(format, capacity);
    int required = BatchPlan.maxBytes(format, capacity);
    if (blob.capacity() < required) {
      throw new IllegalArgumentException(
          "blob holds " + blob.capacity() + " bytes but the capacity needs " + required);
    }
    this.blob = blob.order(ByteOrder.LITTLE_ENDIAN);
    priorShanten = new float[capacity.actions()];
    priorUkeire = new float[capacity.actions()];
  }

  /**
   * 行の位置と各テンソルの領域の位置を決める計画を返す。
   *
   * @return 計画
   */
  public BatchPlan plan() {
    return plan;
  }

  /**
   * 入力のバッファを返す。バッファの position・limit は使わず、ライターは絶対位置で書く。
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
   * その手で打牌した後の向聴を、p0 が読む float の値で返す（打牌でない手は 0）。
   *
   * @param action バッチの中の合法手の番号
   * @return 向聴
   */
  public float priorShanten(int action) {
    return priorShanten[Objects.checkIndex(action, plan.count(InputBlob.Axis.ACTION))];
  }

  /**
   * その手で打牌した後の受け入れの残り枚数を、p0 が読む float の値で返す（打牌でない手は 0）。
   *
   * @param action バッチの中の合法手の番号
   * @return 受け入れの枚数
   */
  public float priorUkeire(int action) {
    return priorUkeire[Objects.checkIndex(action, plan.count(InputBlob.Axis.ACTION))];
  }

  /** 合法手ごとの p0 の特徴を書く。呼ぶのは入力のライターだけ。 */
  public void prior(int action, float shanten, float ukeire) {
    priorShanten[action] = shanten;
    priorUkeire[action] = ukeire;
  }
}
