import type { ErrorBody } from "./errors.ts";
import type { CandidateSet, TableState } from "./result.ts";

// 操作の要求と応答の中身。書く側は、値の無いフィールドを書かない（null も書かない）。

/** モデルのリビジョンを指すキー。 */
export interface ModelKey {
  modelId: string;
  revision: string;
}

/** 解析結果と履歴に残す、モデルの表示用の情報。 */
export interface ModelInfo extends ModelKey {
  displayName: string;
  version: string;
}

/** カタログの 1 つのモデル。使えない理由は Java のログに残し、画面には送らない。 */
export interface CatalogModel extends ModelInfo {
  /** モデルが読む牌譜の人数（4 人か 3 人）。 */
  seats: number;
  available: boolean;
}

export interface ModelCatalog {
  /** models.json に書いた順。 */
  models: readonly CatalogModel[];
  defaultModel?: ModelKey;
}

/** 牌譜の形式。 */
export type RecordSource = "tenhou" | "mjai" | "majsoul";

/** 牌譜の概要。読み込み、履歴、解析結果のどれでも同じ形。 */
export interface RecordSummary {
  source: RecordSource;
  /** 席ごとの名前。人数（4 人か 3 人）と同じ数だけ並び、名前の無い席は空の文字列。 */
  names: readonly string[];
  /** 席ごとの最終の持ち点。前のバージョンが保存した結果には無いことがある。 */
  finalScores?: readonly number[];
  roundCount: number;
}

/** 読み込んで、解析を待っている牌譜。recordId が使えるのは 30 分だけ。 */
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

/** 解析の履歴。新しい解析が先に並ぶ。 */
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

/** 解析の進み具合。progress は 0..100 で、completed になるまでは 99 を超えない。 */
export interface JobSnapshot {
  jobId: string;
  status: JobStatus;
  progress: number;
  /** completed のときだけ書く。 */
  resultId?: string;
  /** failed のときだけ書く（analysis_failed・model_unavailable・shutting_down のどれか）。 */
  error?: ErrorBody;
}

export interface ShareLink {
  url: string;
}

/**
 * 河の 1 枚を変える。index は判断の局面での river[] の位置で、鳴かれた牌も数える。鳴かれた牌は変えられない。
 * 判断する席の河では牌だけを変え、tsumogiri は牌譜の値のまま送る。このとき、その牌が入った配牌かツモも同じ牌に替える。
 */
export interface RiverEdit {
  seat: number;
  index: number;
  tile: string;
  tsumogiri: boolean;
}

/** めくれているドラ表示牌を変える。index 0 が局の始めの表示牌で、1 から後が槓ドラの表示牌。 */
export interface DoraEdit {
  index: number;
  tile: string;
}

/**
 * 判断する席の手牌を 1 枚入れ替える。index は判断の局面での hand[] の位置で、hand.length ならツモ牌を指す。
 * 手牌の枚数は変わらない。リーチの後は使えない。
 */
export interface HandEdit {
  index: number;
  tile: string;
}

/**
 * 判断の局面に加える変更。点数・本場・供託は、判断の局面で見えている値で書く。省いたフィールドは元のまま。
 * 画面は変えた所だけを送る。河は (席, 位置) の順に、ドラと手牌は位置の順に並べる。
 */
export interface ScenarioEdits {
  rivers?: readonly RiverEdit[];
  dora?: readonly DoraEdit[];
  hand?: readonly HandEdit[];
  /** 全ての席の点数。 */
  scores?: readonly number[];
  honba?: number;
  kyotaku?: number;
}

/** もしもで評価する 1 つの判断。seat は判断する席（画面で見ている席）。 */
export interface ScenarioQuery {
  /** ResultDocument.rounds の中の位置。 */
  round: number;
  /** その判断の resolve（判断の結果として起きたイベントの位置）。 */
  eventIndex: number;
  seat: number;
  edits: ScenarioEdits;
  /** 他家の打牌にチーかポンをしたことにするときの、その候補の行動 ID。あれば、鳴いた後の打牌を評価する。 */
  call?: number;
}

/** もしもの結果。table は鳴いた後の局面で、call のときだけ書く（それ以外は画面が変更から局面を作る）。 */
export interface ScenarioAnswer extends CandidateSet {
  table?: TableState;
}

/** フィールドを持たない要求の型。 */
export type NoFields = Record<string, never>;
