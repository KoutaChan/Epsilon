package com.epsilon.prism.hip;

/**
 * 塊の入力（UINT8 の blob 1 本）の区画。名前・並び・軸・要素の型は網の入力の転送の形（{@code InputSchema.Tensor}）と同じで、網の側
 * （{@code HipBackend}）が名前で照らす。区画の幅（1 個あたりの要素の数）は WeightPack の頭が持つ（{@link HipPack}）。
 *
 * <p>トークンの軸は型ごとの区画を続けたもの（局 R・席 4R・牌 34R・出来事 N_E）。作業域の残差の流れも同じ並びにする。
 */
public enum HipSection {
  ROUND_CATEGORY(Axis.ROW, HipElement.U8),
  ROUND_SCALAR(Axis.ROW, HipElement.BF16),
  SEAT_CATEGORY(Axis.SEAT, HipElement.U8),
  SEAT_SCALAR(Axis.SEAT, HipElement.BF16),
  TILE_CATEGORY(Axis.TILE, HipElement.U8),
  EVENT_CATEGORY(Axis.EVENT, HipElement.U8),
  EVENT_SCALAR(Axis.EVENT, HipElement.BF16),
  /** 関係の属性（トークンの軸）。 */
  RELATION(Axis.TOKEN, HipElement.U8),
  TRANSITION_CATEGORY(Axis.TRANSITION, HipElement.U8),
  TRANSITION_SCALAR(Axis.TRANSITION, HipElement.BF16),
  /** 遷移ごとの鍵の符号。 */
  TRANSITION_CODE(Axis.TRANSITION, HipElement.U8),
  /** B の道（FP32）。 */
  DISCARD_EXACT(Axis.TRANSITION, HipElement.FP32),
  /** 手の索引。 */
  ACTION_COLUMN(Axis.ACTION, HipElement.I32),
  /** 門の道（FP32）。 */
  ACTION_EXACT(Axis.ACTION, HipElement.FP32),
  /** 行の索引。 */
  ROW_COLUMN(Axis.ROW, HipElement.I32);

  /** 区画の行の軸。 */
  public enum Axis {
    /** 判断の行。 */
    ROW,
    /** 行ごとの席。 */
    SEAT,
    /** 行ごとの牌。 */
    TILE,
    EVENT,
    /** 局・席・牌・出来事の全てのトークン。 */
    TOKEN,
    TRANSITION,
    /** 合法手。 */
    ACTION
  }

  private final Axis axis;
  private final HipElement element;

  HipSection(Axis axis, HipElement element) {
    this.axis = axis;
    this.element = element;
  }

  public Axis axis() {
    return axis;
  }

  public HipElement element() {
    return element;
  }

  /**
   * 塊の中のこの区画の行の数。
   *
   * @param batch 塊の数
   * @param seats 行ごとの席の数
   * @param tiles 行ごとの牌の数
   * @return 行の数
   */
  public long count(HipBatch batch, int seats, int tiles) {
    return switch (axis) {
      case ROW -> batch.rows();
      case SEAT -> (long) seats * batch.rows();
      case TILE -> (long) tiles * batch.rows();
      case EVENT -> batch.events();
      case TOKEN -> (long) (1 + seats + tiles) * batch.rows() + batch.events();
      case TRANSITION -> batch.transitions();
      case ACTION -> batch.actions();
    };
  }
}
