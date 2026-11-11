import type { ScenarioAnswer, ScenarioQuery } from "../../protocol/model.ts";
import type { ResultDocument } from "../../protocol/result.ts";

// シェル（取り込み・履歴・共有）と再生の画面（雀卓・もしも）の間で受け渡す型。シェルは文書と、もしも・共有・戻るの処理を渡す。再生の画面は、受け取った文書を読んで画面を作る。

/** もしもの判断を 1 つ評価するクライアント。Web のオーナー・共有リンク・デスクトップで送り先は違うが、再生の画面は区別しない。 */
export interface ScenarioClient {
  evaluate(query: ScenarioQuery, signal: AbortSignal): Promise<ScenarioAnswer>;
}

/** 戻るボタンの文言。history は「履歴へ戻る」（replay.home）、home は「ホームへ」（common.home）。 */
export type ReviewBack = "history" | "home";

/** 解析結果の画面（features/replay/ReviewView.tsx の ReviewView）に渡すもの。 */
export interface ReviewEntryProps {
  /** 取り込んだままの文書。形式が正しいかは、再生の画面で 1 回だけ確かめる。 */
  document: ResultDocument;
  /** 最初に見る席（0..3）。 */
  initialSeat: number;
  /** もしもを評価するクライアント。解析に使ったモデルが今のカタログで使えないときは null で、もしもは押せない。 */
  scenario: ScenarioClient | null;
  /** 共有リンクの画面を開く。共有リンクで見ているときとデスクトップでは null で、共有の項目を出さない。 */
  onShare: (() => void) | null;
  back: ReviewBack;
  onBack: () => void;
}
