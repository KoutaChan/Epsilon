package com.epsilon.prism.hip;

import java.util.Objects;

/**
 * 1 つの計画の形。容量で、{@link HipPlan#create} が一度だけ確保する作業領域と重みのバッファの大きさが決まる。
 *
 * @param architecture ネットワークの形の記述
 * @param capacity 1 バッチの容量
 * @param weightBytes WeightPack の容量（バイト）
 */
public record HipShape(HipArchitecture architecture, Capacity capacity, long weightBytes) {

  /**
   * 1 バッチの容量。
   *
   * @param rows 行の数
   * @param events イベントのトークンの数
   * @param transitions 遷移の数（行の数以上）
   * @param actions 合法手の数（行の数以上）
   * @param inputBytes blob のバイト数
   */
  public record Capacity(int rows, int events, int transitions, int actions, long inputBytes) {

    /**
     * 容量を確かめる。
     *
     * @throws IllegalArgumentException 行の数か blob のバイト数が正でないとき、イベントの数が負のとき、遷移か合法手の数が行の数より
     *     少ないとき
     */
    public Capacity {
      if (rows <= 0 || events < 0 || transitions < rows || actions < rows || inputBytes <= 0) {
        throw new IllegalArgumentException(
            "invalid capacity: rows="
                + rows
                + " events="
                + events
                + " transitions="
                + transitions
                + " actions="
                + actions
                + " inputBytes="
                + inputBytes);
      }
    }
  }

  /**
   * 形を確かめる。
   *
   * @throws NullPointerException 形の記述か容量が null のとき
   * @throws IllegalArgumentException WeightPack の容量が 1 バイト〜{@link Integer#MAX_VALUE} バイトの範囲を外れるとき
   */
  public HipShape {
    Objects.requireNonNull(architecture, "architecture");
    Objects.requireNonNull(capacity, "capacity");
    if (weightBytes <= 0 || weightBytes > Integer.MAX_VALUE) {
      throw new IllegalArgumentException(
          "weight pack capacity must be 1.." + Integer.MAX_VALUE + " bytes, got " + weightBytes);
    }
  }

  /** 行ごとの局・席・牌のトークンの数 F。 */
  public int fixedTokens() {
    return architecture.counts().fixed();
  }

  /**
   * バッチがこの計画の容量に収まることを確かめる。
   *
   * @param batch バッチ
   * @throws IllegalArgumentException 容量を超えるとき
   */
  public void requireFits(HipBatch batch) {
    if (batch.rows() > capacity.rows()
        || batch.events() > capacity.events()
        || batch.transitions() > capacity.transitions()
        || batch.actions() > capacity.actions()
        || batch.bytes() > capacity.inputBytes()) {
      throw new IllegalArgumentException(batch + " exceeds the plan capacity " + capacity);
    }
  }
}
