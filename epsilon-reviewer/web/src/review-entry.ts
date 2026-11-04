import type { ScenarioAnswer, ScenarioQuery } from "../../protocol/model.ts";
import type { ResultDocument } from "../../protocol/result.ts";

// シェル（取り込み・履歴・共有）と再生（雀卓・もしも）の境目。シェルは文書と行き先を渡し、再生は文書を読んで画面を作る。

/** もしもの 1 判断を評価する先。Web の持ち主・共有リンク・デスクトップで行き先が違い、再生の側は区別しない。 */
export interface ScenarioClient {
  evaluate(query: ScenarioQuery, signal: AbortSignal): Promise<ScenarioAnswer>;
}

/** 戻るボタンの文言。history は「履歴へ戻る」（replay.home）、home は「ホームへ」（common.home）。 */
export type ReviewBack = "history" | "home";

/** 解析結果の画面（features/replay/ReviewView.tsx の ReviewView）の入口。 */
export interface ReviewEntryProps {
  /** 取り込んだままの文書。読み方の確かめは再生の側が 1 回だけ行う。 */
  document: ResultDocument;
  /** 最初に見る席（0..3）。 */
  initialSeat: number;
  /** もしもの評価の先。null は結果のモデルが今のカタログで使えないこと（もしもは押せない）。 */
  scenario: ScenarioClient | null;
  /** 共有リンクの画面を開く。共有リンクの閲覧とデスクトップでは null（共有の項目を出さない）。 */
  onShare: (() => void) | null;
  back: ReviewBack;
  onBack: () => void;
}
