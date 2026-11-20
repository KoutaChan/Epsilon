package com.epsilon.prism.kernel;

import com.epsilon.prism.kernel.ModelSchema.Section;

/**
 * バッチの入力（UINT8 の blob 1 本）の中の領域。名前・並び・軸・要素の型は、ネットワークが入力を送るときの形
 * （{@code InputSchema.Tensor}）と同じで、入力を渡す側が列名を照合する。領域の幅（1 行あたりの要素の数）は
 * WeightPack のヘッダーが持つ（{@link PackedWeightLayout}）。
 *
 * <p>トークンの軸は、型ごとの領域をつなげたもの（局 R・席 S·R・牌 T·R・イベント N_E。S・T は形の記述にある席と牌の数）。作業領域の
 * 残差ストリームも同じ並びにする。
 */
public enum InputSection {
  ROUND_CATEGORY,
  ROUND_SCALAR,
  SEAT_CATEGORY,
  SEAT_SCALAR,
  TILE_CATEGORY,
  EVENT_CATEGORY,
  EVENT_SCALAR,
  /** 関係の属性（トークンの軸）。 */
  RELATION,
  TRANSITION_CATEGORY,
  TRANSITION_SCALAR,
  /** 遷移ごとに、局・席・牌の各キーに付けるコード。 */
  TRANSITION_CODE,
  /** 打牌の値 B のヘッドが FP32 のまま読むフィールド。 */
  DISCARD_EXACT,
  /** 手のインデックス。 */
  ACTION_COLUMN,
  /** ゲートのヘッドが FP32 のまま読むフィールド。 */
  ACTION_EXACT,
  /** 行のインデックス。 */
  ROW_COLUMN;

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
  private final ElementType element;

  InputSection() {
    Section section = Section.valueOf(name());
    axis = Axis.valueOf(section.axis().name());
    element = ElementType.valueOf(section.element().name());
  }

  public Axis axis() {
    return axis;
  }

  public ElementType element() {
    return element;
  }
}
