import type { ActionType } from "../api/types";
import type { MessageKey } from "./catalog";

// 保存形式の行動名と表示文言の対応を、型検査できる一覧で定義する。
export const actionMessageKeys = {
  DAHAI: "action.dahai",
  CHI: "action.chi",
  PON: "action.pon",
  DAIMINKAN: "action.daiminkan",
  ANKAN: "action.ankan",
  KAKAN: "action.kakan",
  RIICHI_DAHAI: "action.riichi",
  TSUMO_AGARI: "action.tsumo",
  RON_AGARI: "action.ron",
  PASS: "action.pass",
  KYUSHU_KYUHAI: "action.kyushu",
} as const satisfies Record<ActionType, MessageKey>;
