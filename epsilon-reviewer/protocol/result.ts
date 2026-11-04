import type { ModelInfo, RecordSummary } from "./model.ts";

// 解析結果の文書。Java が保存の時に 1 回だけ書き、gzip のまま送り、画面が 1 回だけ読む。
// 書く側は省ける欄を書かない（null も書かない）。省いた欄の意味は各欄の説明のとおり。
// 例外は TableState（全欄を書く）と PlayerDelta.draw（null は「ツモ牌が無くなった」）。

/** 文書の形式の番号。これ以外は開かない。 */
export const RESULT_FORMAT = 6;

export interface ResultDocument {
  format: typeof RESULT_FORMAT;
  /** 小文字の UUID。共有リンクの id も同じ値。 */
  resultId: string;
  /** 解析した時刻（ISO-8601、UTC、ミリ秒まで）。 */
  createdAt: string;
  model: ModelInfo;
  record: RecordSummary;
  /**
   * 価値を推定するモデルの表示の温度。あれば候補の値は最善手との価値差（損失）で、表示の確率は
   * candidateProbabilities で作る。無ければ候補の値は確率。
   */
  temperature?: number;
  rounds: readonly RoundDocument[];
}

/** 1 局。手順は牌譜の順で、判断はそれが解く出来事の前に並ぶ。 */
export interface RoundDocument {
  /** (場風) × 4 + (局 − 1)。0 が東 1 局。 */
  roundIndex: number;
  honba: number;
  /** 親の席（0..3）。 */
  dealer: number;
  /** 局の最初の出来事（start_kyoku）の牌譜の中の位置。出来事の手順は 1 つずつ続く位置を持つ。 */
  firstEvent: number;
  /** 局の最初の手順の局面。 */
  start: TableState;
  steps: readonly Step[];
}

/** 局の 1 手順。event を持てば出来事、seat を持てば判断。 */
export type Step = EventStep | DecisionStep;

/** 牌譜の出来事。位置は firstEvent からの出来事の手順の数で決まる。 */
export interface EventStep {
  event: EventKind;
  /** 行った席。無ければ席の無い出来事（-1 として扱う）。 */
  actor?: number;
  /** 直前の手順の局面からの差分。無ければ局面は変わらない。 */
  delta?: StateDelta;
  /** hora と ryukyoku だけが持つ。 */
  outcome?: RoundOutcome;
}

/** 1 席の判断。候補は表示の順（最善が先）。 */
export interface DecisionStep extends CandidateSet {
  seat: number;
  /** 判断を起こした出来事の位置。無ければこの判断の前の最後の出来事の位置。 */
  cause?: number;
  /** 判断が解かれた出来事の位置。無ければ cause + 1。牌譜の終わりで開いたままの判断は出来事の数。 */
  resolve?: number;
  /** 直前の手順の局面からの差分。無ければ局面は変わらない。 */
  delta?: StateDelta;
}

/** 判断の候補。もしもの答えも同じ形。 */
export interface CandidateSet {
  candidates: readonly CandidateRow[];
  /** 牌譜で選んだ候補の位置。無ければ記録に無い判断（未観測の応答・もしもで選べなくなった手）。 */
  chosen?: number;
  /**
   * 表示の確率（候補の順）。温度から作れない評価のときだけ書く。価値を推定しないモデルでは書かない
   * （値がそのまま確率）。
   */
  probabilities?: readonly number[];
}

/**
 * 1 候補。値は float32 の最短の十進表記（損失か確率）。牌は CALL_TYPES の手だけが持ち、他の手の種類・ツモ切り・
 * 打牌の牌は actionOf(actionId) で決まる。
 */
export type CandidateRow =
  | readonly [actionId: number, value: number]
  | readonly [actionId: number, value: number, tiles: readonly string[]];

/** 出来事の種類（牌譜の mjai の名前）。start_game は手順にならない。 */
export const EVENT_KINDS = [
  "start_kyoku",
  "tsumo",
  "dahai",
  "chi",
  "pon",
  "daiminkan",
  "ankan",
  "kakan",
  "reach",
  "reach_accepted",
  "dora",
  "hora",
  "ryukyoku",
  "end_kyoku",
  "end_game",
  "none",
] as const;
export type EventKind = (typeof EVENT_KINDS)[number];

/** 局面。players の位置が絶対の席。 */
export interface TableState {
  players: readonly PlayerState[];
  /** 開いたドラ表示牌（開いた順）。 */
  dora: readonly string[];
  honba: number;
  /** 卓の上の供託の本数。 */
  kyotaku: number;
  /** 残りのツモの数。 */
  remaining: number;
  /** 手番の席。 */
  activeSeat: number;
}

export interface PlayerState {
  score: number;
  riichi: boolean;
  /** 手牌（ツモ牌を除く）。tileOrder の順。 */
  hand: readonly string[];
  /** 見えているツモ牌。他家の牌が見えない局面では他家は null。 */
  draw: string | null;
  river: readonly RiverTile[];
  melds: readonly Meld[];
}

/** 河の 1 枚。印は真のときだけ書く。 */
export interface RiverTile {
  tile: string;
  tsumogiri?: true;
  /** リーチの宣言牌。 */
  riichi?: true;
  /** 鳴かれた牌（河に残して数える）。 */
  called?: true;
}

export interface Meld {
  type: "CHI" | "PON" | "DAIMINKAN" | "ANKAN" | "KAKAN";
  /** CHI・PON は 3 枚、槓は 4 枚。表示の順。 */
  tiles: readonly string[];
  /** 鳴いた元の絶対の席。暗槓は -1。 */
  from: number;
  /** 横向きの牌の位置。暗槓は -1。 */
  calledIndex: number;
  /** 加槓で加えた牌の位置（3）。他は -1。 */
  addedIndex: number;
}

/**
 * 配列の 1 か所の置き換え [位置, 消す数, ...入れる値]。前の配列の位置から消す数だけ消し、そこへ値を入れる。
 */
export type Splice<T> = readonly [
  index: number,
  removeCount: number,
  ...values: T[],
];

/** 局面の差分。変わった欄だけを書く。 */
export interface StateDelta {
  /** 変わった席だけ、席の順。 */
  players?: readonly PlayerDelta[];
  dora?: Splice<string>;
  honba?: number;
  kyotaku?: number;
  remaining?: number;
  activeSeat?: number;
}

export interface PlayerDelta {
  seat: number;
  score?: number;
  riichi?: boolean;
  hand?: Splice<string>;
  /** 新しいツモ牌。null はツモ牌が無くなったこと。 */
  draw?: string | null;
  river?: Splice<RiverTile>;
  melds?: Splice<Meld>;
}

/** 局の終わり。流局の理由と点の移動は持たない（画面は持ち点の差を出す）。 */
export type RoundOutcome =
  | {
      type: "hora";
      actor: number;
      /** ロンされた席。ツモは actor と同じ。 */
      target: number;
      winningTile: string;
      /** 採点の内訳が牌譜にも計算にも無ければ書かない。 */
      details?: WinDetails;
    }
  | { type: "ryukyoku" };

/**
 * 採点の内訳。牌譜に無い時は、天鳳の規則で計算した点の移動が牌譜と全ての席で一致した時だけ計算の値を持つ。
 * 分からない欄は書かない。
 */
export interface WinDetails {
  han?: number;
  fu?: number;
  points?: number;
  /** 役満の倍数。 */
  yakuman?: number;
  yaku?: readonly Yaku[];
  /** 裏ドラ表示牌。空の配列は「裏ドラを見た記録があり 0 枚」、書かないのは「記録が無い」。 */
  uraIndicators?: readonly string[];
}

export interface Yaku {
  code: string;
  han: number;
  yakuman: number;
}

/** 行動の種類（Java の Action.Type と同じ名前と並び）。 */
export const ACTION_TYPES = [
  "DAHAI",
  "CHI",
  "PON",
  "DAIMINKAN",
  "ANKAN",
  "KAKAN",
  "RIICHI_DAHAI",
  "TSUMO_AGARI",
  "RON_AGARI",
  "PASS",
  "KYUSHU_KYUHAI",
] as const;
export type ActionType = (typeof ACTION_TYPES)[number];

/** 候補の行に牌を書く手の種類。他の手の牌は行動 ID で決まる。 */
export const CALL_TYPES: ReadonlySet<ActionType> = new Set<ActionType>([
  "CHI",
  "PON",
  "DAIMINKAN",
  "ANKAN",
  "KAKAN",
]);

/** 行動 ID だけで決まる手の形。 */
export interface ActionShape {
  type: ActionType;
  tsumogiri: boolean;
  /** 打牌とリーチの打牌の牌（赤は r 付き）。他の手は null。 */
  tile: string | null;
}

const HONOR_CODES = ["E", "S", "W", "N", "P", "F", "C"] as const;
const RED_FIVES = ["5mr", "5pr", "5sr"] as const;

/** 牌種の番号（0..33。萬・筒・索の 1..9、東南西北白發中）の牌の記号。 */
export function tileTypeCode(type: number): string {
  if (type >= 27) return HONOR_CODES[type - 27]!;
  return `${(type % 9) + 1}${"mps"[Math.floor(type / 9)]}`;
}

/**
 * 行動 ID（Java の Action.toIndex、0..371）を手の形にする。並びは Action.java の固定の配置
 * （打牌 0..73、チー 74..154、ポン 155..191、大明槓 192..225、暗槓 226..259、加槓 260..293、
 * リーチ 294..367、ツモ 368、ロン 369、見送り 370、九種九牌 371）。
 */
export function actionOf(actionId: number): ActionShape {
  if (!Number.isInteger(actionId) || actionId < 0 || actionId > 371)
    throw new RangeError(`Invalid action ID: ${actionId}`);
  if (actionId < 74) return discard("DAHAI", actionId);
  if (actionId < 155) return { type: "CHI", tsumogiri: false, tile: null };
  if (actionId < 192) return { type: "PON", tsumogiri: false, tile: null };
  if (actionId < 226)
    return { type: "DAIMINKAN", tsumogiri: false, tile: null };
  if (actionId < 260) return { type: "ANKAN", tsumogiri: false, tile: null };
  if (actionId < 294) return { type: "KAKAN", tsumogiri: false, tile: null };
  if (actionId < 368) return discard("RIICHI_DAHAI", actionId - 294);
  const type = (["TSUMO_AGARI", "RON_AGARI", "PASS", "KYUSHU_KYUHAI"] as const)[
    actionId - 368
  ]!;
  return { type, tsumogiri: false, tile: null };
}

/** 打牌の 74 の並び（手出し 34、ツモ切り 34、赤の手出し 3、赤のツモ切り 3）の 1 つ。 */
function discard(type: "DAHAI" | "RIICHI_DAHAI", offset: number): ActionShape {
  if (offset < 34)
    return { type, tsumogiri: false, tile: tileTypeCode(offset) };
  if (offset < 68)
    return { type, tsumogiri: true, tile: tileTypeCode(offset - 34) };
  if (offset < 71)
    return { type, tsumogiri: false, tile: RED_FIVES[offset - 68]! };
  return { type, tsumogiri: true, tile: RED_FIVES[offset - 71]! };
}

/** 候補の牌。CALL_TYPES の手は行の牌、打牌は行動 ID の牌、他は無し。 */
export function candidateTiles(row: CandidateRow): readonly string[] {
  if (row.length === 3) return row[2];
  const { tile } = actionOf(row[0]);
  return tile === null ? [] : [tile];
}

/**
 * 損失から表示の確率を作る: p_i = exp(-loss_i / temperature) / Σ_j exp(-loss_j / temperature)。
 * 和は候補の順に binary64 で取り、各値を float32 に丸める。Java の書く側はこの値と方策の確率の差が
 * 1e-6 を超える判断だけ probabilities を書く。
 */
export function displayProbabilities(
  losses: readonly number[],
  temperature: number,
): number[] {
  const weights = losses.map((loss) => Math.exp(-loss / temperature));
  let total = 0;
  for (const weight of weights) total += weight;
  return weights.map((weight) => Math.fround(weight / total));
}

/** 判断の候補の表示の確率（候補の順）。temperature は文書の temperature。 */
export function candidateProbabilities(
  set: CandidateSet,
  temperature: number | undefined,
): readonly number[] {
  if (set.probabilities) return set.probabilities;
  const values = set.candidates.map((row) => row[1]);
  return temperature === undefined
    ? values
    : displayProbabilities(values, temperature);
}

/** 判断の候補の損失（候補の順）。価値を推定しないモデルは null。 */
export function candidateLosses(
  set: CandidateSet,
  temperature: number | undefined,
): readonly number[] | null {
  return temperature === undefined ? null : set.candidates.map((row) => row[1]);
}

const TILE_PATTERN = /^(?:[1-9][mps]|5[mps]r|[ESWNPFC])$/;

/** 牌の記号か（萬筒索の 1..9、赤は 5 だけ r 付き、字牌は ESWNPFC）。 */
export function isTileCode(value: unknown): value is string {
  return typeof value === "string" && TILE_PATTERN.test(value);
}

/** 手牌の並びの順の値。萬・筒・索の 1..9、東南西北白發中の順で、赤の 5 は同じ 5 の前。 */
export function tileOrder(tile: string): number {
  const red = tile.length === 3;
  const honor = HONOR_CODES.indexOf(tile as (typeof HONOR_CODES)[number]);
  const type =
    honor >= 0 ? 27 + honor : "mps".indexOf(tile[1]!) * 9 + Number(tile[0]) - 1;
  return type * 2 + (red ? 0 : 1);
}

/** 書き出した解析結果のファイルの名前。 */
export function resultFileName(resultId: string): string {
  return `${resultId}.epsilon-reviewer.json.gz`;
}
