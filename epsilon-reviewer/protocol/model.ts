import type { ErrorBody } from "./errors.ts";
import type { CandidateSet, TableState } from "./result.ts";

// 操作の要求と応答の中身。書く側は値の無い欄を書かない（null は書かない）。

/** モデルの版を指す鍵。 */
export interface ModelKey {
  modelId: string;
  revision: string;
}

/** 解析結果と履歴に残すモデルの表示。 */
export interface ModelInfo extends ModelKey {
  displayName: string;
  version: string;
}

/** カタログの 1 モデル。使えない理由は Java の記録に残し、画面には送らない。 */
export interface CatalogModel extends ModelInfo {
  available: boolean;
}

export interface ModelCatalog {
  /** models.json の順。 */
  models: readonly CatalogModel[];
  defaultModel?: ModelKey;
}

/** 牌譜の形式。 */
export type RecordSource = "tenhou" | "mjai" | "majsoul";

/** 牌譜の概要。読み込み・履歴・解析結果で同じ形。 */
export interface RecordSummary {
  source: RecordSource;
  /** 4 席の名前。名前の無い席は空の文字列。 */
  names: readonly string[];
  /** 4 席の最終の持ち点。牌譜に無ければ書かない。 */
  finalScores?: readonly number[];
  roundCount: number;
}

/** 読み込んで解析を待つ牌譜。recordId は 30 分だけ使える。 */
export interface ImportedRecord {
  recordId: string;
  record: RecordSummary;
}

export interface HistoryEntry {
  resultId: string;
  createdAt: string;
  record: RecordSummary;
  model: ModelInfo;
}

/** 履歴。新しい解析が先。 */
export interface HistoryList {
  results: readonly HistoryEntry[];
}

export interface AnalysisStarted {
  jobId: string;
  model: ModelInfo;
}

export const JOB_STATUSES = [
  "queued",
  "running",
  "saving",
  "completed",
  "failed",
  "cancelled",
] as const;
export type JobStatus = (typeof JOB_STATUSES)[number];

/** 解析の進み。progress は 0..100 で、completed になるまで 99 を超えない。 */
export interface JobSnapshot {
  jobId: string;
  status: JobStatus;
  progress: number;
  /** completed のときだけ。 */
  resultId?: string;
  /** failed のときだけ（analysis_failed・model_unavailable・shutting_down）。 */
  error?: ErrorBody;
}

export interface ShareLink {
  url: string;
}

/**
 * 河の 1 枚の変更。index は判断の局面の river[] の位置（鳴かれた牌も数える）。鳴かれた牌は変えられない。
 * 判断の席の河は牌だけを変え、tsumogiri は記録のまま送る（その牌が入った配牌かツモも同じ牌に替える）。
 */
export interface RiverEdit {
  seat: number;
  index: number;
  tile: string;
  tsumogiri: boolean;
}

/** 開いているドラ表示牌の変更。index 0 が局の始まりの表示牌、1 からが槓ドラ。 */
export interface DoraEdit {
  index: number;
  tile: string;
}

/**
 * 判断の席の手牌の 1 枚の入れ替え。index は判断の局面の hand[] の位置、hand.length はツモ牌。
 * 牌の数は変わらない。リーチの後は使えない。
 */
export interface HandEdit {
  index: number;
  tile: string;
}

/**
 * 判断の局面に当てる変更。点数・本場・供託は判断の局面で見えている値。省いた欄は元のまま。
 * 画面は変えた所だけを、河は (席, 位置)、ドラと手牌は位置の順で送る。
 */
export interface ScenarioEdits {
  rivers?: readonly RiverEdit[];
  dora?: readonly DoraEdit[];
  hand?: readonly HandEdit[];
  /** 4 席の点数。 */
  scores?: readonly number[];
  honba?: number;
  kyotaku?: number;
}

/** もしもの 1 判断。席は判断の席（見ている席）。 */
export interface ScenarioQuery {
  /** ResultDocument.rounds の位置。 */
  round: number;
  /** 判断の resolve（解かれた出来事の位置）。 */
  eventIndex: number;
  seat: number;
  edits: ScenarioEdits;
  /** 応答の判断で鳴いたとするチー・ポンの候補の行動 ID。あれば鳴いた後の打牌を評価する。 */
  call?: number;
}

/** もしもの答え。table は鳴いた後の局面で、call のときだけ（他は画面が変更から作る）。 */
export interface ScenarioAnswer extends CandidateSet {
  table?: TableState;
}

/** 要求の欄の無い操作。 */
export type NoFields = Record<string, never>;
