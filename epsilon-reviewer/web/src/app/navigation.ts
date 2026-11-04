import type {
  AnalysisStarted,
  HistoryEntry,
  ImportedRecord,
  ModelInfo,
} from "../../../protocol/model.ts";
import type { ResultDocument } from "../../../protocol/result.ts";

// 持ち主の画面の遷移。URL は使わず、読み込み直すと最初の画面へ戻る。

/** 画面。review はシェル（頭・知らせ・窓）を全て置き換える。 */
export type Screen =
  | { name: "home" }
  | { name: "confirm"; record: ImportedRecord; seat: number }
  | {
      name: "progress";
      record: ImportedRecord;
      seat: number;
      jobId: string;
      /** 解析を受けた側が返したモデル。 */
      model: ModelInfo;
    }
  | { name: "review"; document: ResultDocument; seat: number };

/** 共有の窓の対象。履歴の項目と解析結果の文書のどちらでもよい。 */
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
  /** 牌譜を読み込んだ（最初に見る席は 0）。 */
  | { type: "imported"; record: ImportedRecord }
  | { type: "seatChosen"; seat: number }
  | { type: "started"; started: AnalysisStarted }
  /** 解析が終わり文書を読んだ。解析の画面にいる時だけ、その席で開く。 */
  | { type: "completed"; document: ResultDocument }
  /** 履歴から開いた。 */
  | { type: "opened"; document: ResultDocument }
  /** 解析の画面は確かめへ、確かめと解析結果は最初の画面へ。 */
  | { type: "back" }
  /** デスクトップの保存先を変えた。読み込んだ牌譜と解析の仕事は前の保存先のもの。 */
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
      return {
        ...state,
        screen:
          screen.name === "progress"
            ? { name: "confirm", record: screen.record, seat: screen.seat }
            : HOME,
      };
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
