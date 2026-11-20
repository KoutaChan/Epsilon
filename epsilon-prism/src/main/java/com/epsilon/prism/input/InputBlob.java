package com.epsilon.prism.input;

import com.epsilon.prism.kernel.ModelSchema.Section;

/**
 * 1 バッチの入力の並べ方。各領域（{@link Tensor}）の名前・軸・要素の型と、行と手のインデックスの列は、ルールに依らない。領域の幅は
 * ルールごとの形式（{@link InputFormat#width}）が決める。カーネルの入力領域とは名前で対応させるので、名前は変えない。
 */
public final class InputBlob {

  private InputBlob() {}

  /** テンソルの行の軸。 */
  public enum Axis {
    /** 判断ごとの行。 */
    ROW,
    /** 行ごとに、席の数だけ並ぶ席のトークン。 */
    SEAT,
    /** 行ごとに 34 個並ぶ牌のトークン。 */
    TILE,
    /** 行ごとに、0 個からイベントの容量まで並ぶイベントのトークン。 */
    EVENT,
    /** 局・席・牌・イベントの全てのトークン。 */
    TOKEN,
    TRANSITION,
    ACTION
  }

  /** 要素の型。 */
  public enum Element {
    U8(1),
    /** bfloat16（最近接偶数で丸める）。 */
    BF16(2),
    FP32(4),
    I32(4);

    private final int bytes;

    Element(int bytes) {
      this.bytes = bytes;
    }

    /**
     * 1 要素のバイト数を返す。
     *
     * @return バイト数
     */
    public int bytes() {
      return bytes;
    }
  }

  /** 1 バッチで送るテンソル。領域はこの順に並べ、各領域の先頭を {@link BatchPlan#TENSOR_ALIGNMENT} バイトの境界にそろえる。 */
  public enum Tensor {
    ROUND_CATEGORY,
    ROUND_SCALAR,
    SEAT_CATEGORY,
    SEAT_SCALAR,
    TILE_CATEGORY,
    EVENT_CATEGORY,
    EVENT_SCALAR,
    RELATION,
    TRANSITION_CATEGORY,
    TRANSITION_SCALAR,
    /** 遷移ごとに、局・席・牌の各キーに付けるコード。 */
    TRANSITION_CODE,
    /** 打牌の値のヘッドが FP32 のまま読むフィールド。 */
    DISCARD_EXACT,
    ACTION_COLUMN,
    /** ゲートのヘッドが FP32 のまま読むフィールド。 */
    ACTION_EXACT,
    ROW_COLUMN;

    private final Axis axis;
    private final Element element;

    Tensor() {
      Section section = Section.valueOf(name());
      axis = Axis.valueOf(section.axis().name());
      element = Element.valueOf(section.element().name());
    }

    /**
     * 行の軸を返す。
     *
     * @return 軸
     */
    public Axis axis() {
      return axis;
    }

    /**
     * 要素の型を返す。
     *
     * @return 要素の型
     */
    public Element element() {
      return element;
    }
  }

  /** 手ごとのインデックスの列（INT32。埋め込みには通さない）。番号は 0 始まりで、無いときは −1。 */
  public enum ActionColumn {
    ROW,
    FIRST_TRANSITION,
    /** 遷移の数（同値な打牌のうち、代表でない手は 0）。 */
    TRANSITIONS,
    /** 価値グラフでの手の扱いの番号（形式の {@code roles()} の順）。 */
    ROLE,
    /** 基準にする手（立直して x を切る手ならダマで x を切る手、応答なら PASS、ほかは −1）。 */
    BASE,
    /** 同値な打牌の代表（代表の手では自分自身）。 */
    REPRESENTATIVE
  }

  /** 行ごとのインデックスの列（INT32。埋め込みには通さない）。 */
  public enum RowColumn {
    FIRST_EVENT,
    EVENTS,
    FIRST_ACTION,
    ACTIONS,
    FIRST_TRANSITION,
    TRANSITIONS,
    /** 判断の種類（ルールの語彙の中の位置）。価値グラフの形を選ぶのに使うので、局のフィールドと重複して持つ。 */
    DECISION_KIND
  }

  /** 価値グラフでの手の扱いごとの、s の求め方。推論カーネルも同じ求め方をする。 */
  public enum Composition {
    /** s = B(t)。 */
    DISCARD_VALUE,
    /** s = A(s[ダマ]) + Δ_RIICHI（基準はダマの遷移）。 */
    RIICHI_GATE,
    /** s = Δ（基準は PASS の遷移、PASS ≡ 0）。 */
    RESPONSE_GATE,
    /** s = A(M) + Δ（基準は ē）。 */
    TURN_GATE,
    /** s = 0。 */
    ZERO
  }
}
