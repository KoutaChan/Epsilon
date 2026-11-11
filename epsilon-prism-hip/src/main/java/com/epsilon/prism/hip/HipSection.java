package com.epsilon.prism.hip;

/**
 * バッチの入力（UINT8 の blob 1 本）の中の領域。名前・並び・軸・要素の型は、ネットワークが入力を送るときの形
 * （{@code InputSchema.Tensor}）と同じで、ネットワークの側（{@code HipBackend}）が名前で照合する。領域の幅（1 行あたりの要素の数）は
 * WeightPack のヘッダーが持つ（{@link HipPack}）。
 *
 * <p>トークンの軸は、型ごとの領域をつなげたもの（局 R・席 S·R・牌 T·R・イベント N_E。S・T は形の記述にある席と牌の数）。作業領域の
 * 残差ストリームも同じ並びにする。
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
  /** 遷移ごとに、局・席・牌の各キーに付けるコード。 */
  TRANSITION_CODE(Axis.TRANSITION, HipElement.U8),
  /** 打牌の値 B のヘッドが FP32 のまま読むフィールド。 */
  DISCARD_EXACT(Axis.TRANSITION, HipElement.FP32),
  /** 手のインデックス。 */
  ACTION_COLUMN(Axis.ACTION, HipElement.I32),
  /** ゲートのヘッドが FP32 のまま読むフィールド。 */
  ACTION_EXACT(Axis.ACTION, HipElement.FP32),
  /** 行のインデックス。 */
  ROW_COLUMN(Axis.ROW, HipElement.I32);

  /** 領域の 1 行が何に当たるか。 */
  public enum Axis {
    /** 判断の行。 */
    ROW,
    /** 行ごとの席。 */
    SEAT,
    /** 行ごとの牌。 */
    TILE,
    EVENT,
    /** 局・席・牌・イベントの全てのトークン。 */
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
   * バッチの中で、この領域が持つ行の数を返す。
   *
   * @param batch バッチ
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
