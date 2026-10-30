package com.epsilon.prism.hip;

/**
 * WeightPack の頭が持つ入力の約束（int32、native の {@code Schema} と同じ順）。カーネルは列の番号・数をここから読み、手で書かない。
 * 値は網の側（{@code WeightLayout}）が入力の宣言から作る。
 */
public enum HipSchema {
  /** 行ごとの席のトークンの数（4）。 */
  SEATS,
  /** 行ごとの牌のトークンの数（34）。牌の種類の数と同じで、牌の関係の表は TILES² 個。 */
  TILES,
  /** 行の索引の列: 最初の出来事。 */
  ROW_FIRST_EVENT,
  /** 行の索引の列: 出来事の数。 */
  ROW_EVENTS,
  /** 行の索引の列: 最初の手。 */
  ROW_FIRST_ACTION,
  /** 行の索引の列: 手の数。 */
  ROW_ACTIONS,
  /** 行の索引の列: 最初の遷移。 */
  ROW_FIRST_TRANSITION,
  /** 行の索引の列: 遷移の数。 */
  ROW_TRANSITIONS,
  /** 手の索引の列: 行。 */
  ACTION_ROW,
  /** 手の索引の列: 最初の遷移（代表でない双子は −1）。 */
  ACTION_FIRST_TRANSITION,
  /** 手の索引の列: 遷移の数（代表でない双子は 0）。 */
  ACTION_TRANSITIONS,
  /** 手の索引の列: 役割（役割の ordinal）。 */
  ACTION_ROLE,
  /** 手の索引の列: 基準の手（なしは −1）。 */
  ACTION_BASE,
  /** 手の索引の列: 双子の代表。 */
  ACTION_REPRESENTATIVE,
  /** 関係の属性の列: 牌。 */
  RELATION_TILE,
  /** 関係の属性の列: 持ち主。 */
  RELATION_OWNER,
  /** 関係の属性の列: 出来事の番号。 */
  RELATION_EVENT,
  /** 関係の属性の列: 持ち主の立直宣言の番号。 */
  RELATION_RIICHI,
  /** 関係の属性の列: 持ち主の最後の打牌の番号。 */
  RELATION_LAST_DISCARD,
  /** 関係の属性の列: 繋がる出来事。 */
  RELATION_LINK,
  /** 関係の属性の列: 相手から見た安全の最初の列。 */
  RELATION_SAFETY,
  /** 相手から見た安全の列の数（3）。 */
  RELATION_SAFETIES,
  /** 遷移の欄の列: 鳴いた後に捨てる牌。 */
  TRANSITION_CALL_DISCARD,
  /** 遷移の欄の列: 主牌。 */
  TRANSITION_MAIN_TILE,
  /** 鍵の符号の数（E_k・E_v の行）。 */
  CODES,
  /** 関係の偏りの表の行（区画 7 × 16）。 */
  RELATION_ROWS,
  /** 門の種類の数。 */
  GATE_KINDS,
  /** 手の役割の数（{@link HipPack#MAX_ROLES} 以下）。 */
  ROLES,
  /** 集約の温度 τ（float のビット）。 */
  TAU_BITS,
  /** 注意のロジットの尺度 1/√16（float のビット）。 */
  SCALE_BITS
}
