import type {
  AnalysisStarted,
  HistoryEntry,
  ImportedRecord,
  ModelInfo,
} from "../../../protocol/model.ts";
import type { ResultDocument } from "../../../protocol/result.ts";

// オーナーの画面の遷移。URL は使わないので、読み込み直すと最初の画面へ戻る。

/** 画面の種類。解析結果の画面（review）だけは、シェル（ヘッダー・トースト・ダイアログ）ごと置き換えて表示する。 */
export type Screen =
  | { name: "home" }
  | { name: "confirm"; record: ImportedRecord; seat: number }
  | {
      name: "progress";
      record: ImportedRecord;
      seat: number;
      jobId: string;
      /** 解析を受け付けた側が返したモデル。 */
      model: ModelInfo;
    }
  | { name: "review"; document: ResultDocument; seat: number };

/** 共有ダイアログの対象。履歴の項目でも解析結果の文書でもよい。 */
export type ShareTarget = Pick<HistoryEntry, "resultId" | "record">;

export type Modal =
  | { name: "storage" }
  | { name: "settings" }
  | { name: "delete"; entry: HistoryEntry }
  | { name: "share"; target: ShareTarget };

export interface Navigation {
  screen: Screen;
  modal: Modal | null;
}

export type NavigationAction =
  /** 牌譜を取り込んだ。確認の画面へ進み、最初に見る席は 0 にする。 */
  | { type: "imported"; record: ImportedRecord }
  | { type: "seatChosen"; seat: number }
  | { type: "started"; started: AnalysisStarted }
  /** 解析が終わり、文書を読んだ。解析中の画面にいるときだけ、選んでいた席で解析結果を開く。 */
  | { type: "completed"; document: ResultDocument }
  /** 履歴から解析結果を開いた。最初に見る席は 0。 */
  | { type: "opened"; document: ResultDocument }
  /** 戻る。どの画面からも最初の画面へ戻る。解析に渡した牌譜は解析が終わると外れるので、解析中の画面から確認の画面へは戻らない。 */
  | { type: "back" }
  /** デスクトップの保存先を変えた。読み込んだ牌譜と解析のジョブは前の保存先のものなので、確認・解析中の画面からは最初の画面へ戻る。 */
  | { type: "folderChanged" }
  | { type: "modalOpened"; modal: Modal }
  | { type: "modalClosed" };

const HOME: Screen = { name: "home" };

export const INITIAL_NAVIGATION: Navigation = { screen: HOME, modal: null };

export function navigate(
  state: Navigation,
  action: NavigationAction,
): Navigation {
  const { screen } = state;
  switch (action.type) {
    case "imported":
      return {
        ...state,
        screen: { name: "confirm", record: action.record, seat: 0 },
      };
    case "seatChosen":
      return screen.name === "confirm"
        ? { ...state, screen: { ...screen, seat: action.seat } }
        : state;
    case "started":
      return screen.name === "confirm"
        ? {
            ...state,
            screen: {
              name: "progress",
              record: screen.record,
              seat: screen.seat,
              jobId: action.started.jobId,
              model: action.started.model,
            },
          }
        : state;
    case "completed":
      return screen.name === "progress"
        ? {
            screen: {
              name: "review",
              document: action.document,
              seat: screen.seat,
            },
            modal: null,
          }
        : state;
    case "opened":
      return {
        screen: { name: "review", document: action.document, seat: 0 },
        modal: null,
      };
    case "back":
      return { ...state, screen: HOME };
    case "folderChanged":
      return screen.name === "confirm" || screen.name === "progress"
        ? { ...state, screen: HOME }
        : state;
    case "modalOpened":
      return { ...state, modal: action.modal };
    case "modalClosed":
      return { ...state, modal: null };
  }
}
