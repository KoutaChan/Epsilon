package com.epsilon.prism.hip;

/**
 * WeightPack のヘッダーに書く、入力の列の番号（int32）。並びは native の {@code Schema} と同じ。カーネルは列の番号をここから読み、
 * コードに直接書かない。数は {@link HipArchitecture} が持つ。値はネットワークの側（{@code WeightLayout}）が入力の宣言から作る。
 */
public enum HipSchema {
  /** 行のインデックスの列: 最初のイベント。 */
  ROW_FIRST_EVENT,
  /** 行のインデックスの列: イベントの数。 */
  ROW_EVENTS,
  /** 行のインデックスの列: 最初の手。 */
  ROW_FIRST_ACTION,
  /** 行のインデックスの列: 手の数。 */
  ROW_ACTIONS,
  /** 行のインデックスの列: 最初の遷移。 */
  ROW_FIRST_TRANSITION,
  /** 行のインデックスの列: 遷移の数。 */
  ROW_TRANSITIONS,
  /** 手のインデックスの列: 行。 */
  ACTION_ROW,
  /** 手のインデックスの列: 最初の遷移（同値な打牌のうち、代表でない手は −1）。 */
  ACTION_FIRST_TRANSITION,
  /** 手のインデックスの列: 遷移の数（同値な打牌のうち、代表でない手は 0）。 */
  ACTION_TRANSITIONS,
  /** 手のインデックスの列: 価値グラフでの手の扱い（その ordinal）。 */
  ACTION_ROLE,
  /** 手のインデックスの列: 基準にする手（無ければ −1）。 */
  ACTION_BASE,
  /** 手のインデックスの列: 同値な打牌の代表。 */
  ACTION_REPRESENTATIVE,
  /** 関係の属性の列: 牌。 */
  RELATION_TILE,
  /** 関係の属性の列: 持ち主。 */
  RELATION_OWNER,
  /** 関係の属性の列: イベントの番号。 */
  RELATION_EVENT,
  /** 関係の属性の列: 持ち主が立直を宣言したイベントの番号。 */
  RELATION_RIICHI,
  /** 関係の属性の列: 持ち主の最後の打牌のイベントの番号。 */
  RELATION_LAST_DISCARD,
  /** 関係の属性の列: 繋がるイベント。 */
  RELATION_LINK,
  /** 関係の属性の列: 相手から見た安全の最初の列（相手の数だけ続く）。 */
  RELATION_SAFETY,
  /** 遷移のフィールドの列: 鳴いた後に捨てる牌。 */
  TRANSITION_CALL_DISCARD,
  /** 遷移のフィールドの列: 主牌。 */
  TRANSITION_MAIN_TILE,
  /** 関係のバイアスの表の行の数（7 区分 × 16）。コードの割り当て方で決まる定数として照合する。 */
  RELATION_ROWS
}
