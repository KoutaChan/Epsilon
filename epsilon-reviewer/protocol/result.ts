import type { ModelInfo, RecordSummary } from "./model.ts";

// 解析結果の文書。Java が保存するときに 1 回だけ書き、gzip のまま送り、画面が 1 回だけ読む。
// 書く側は、省けるフィールドを書かない（null も書かない）。省いたときの意味は各フィールドの説明に書く。
// 例外は TableState（全てのフィールドを書く）と PlayerDelta.draw（null は「ツモ牌が無くなった」の意味）。

/** 文書の形式の番号。これ以外の番号の文書は開かない。 */
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
   * 価値を推定するモデルで、表示に使う温度。あれば、候補の値は最善手との価値の差（損失）で、表示する確率は
   * candidateProbabilities で作る。無ければ、候補の値は確率そのもの。
   */
  temperature?: number;
  rounds: readonly RoundDocument[];
}

/**
 * 1 局。手順は牌譜の順に並び、判断はその判断の結果として起きるイベントの前に置く。人数（4 人か 3 人）は
 * record.names の数。
 */
export interface RoundDocument {
  /** (場風) × 人数 + (局 − 1)。0 が東 1 局。 */
  roundIndex: number;
  honba: number;
  /** 親の席（0..人数 − 1）。 */
  dealer: number;
  /** 局の最初のイベント（start_kyoku）が、牌譜の何番目のイベントか。後に続くイベントの位置は、ここから 1 つずつ増える。 */
  firstEvent: number;
  /** 局の最初の手順の局面。 */
  start: TableState;
  steps: readonly Step[];
}

/** 局の 1 つの手順。event を持てばイベント、seat を持てば判断。 */
export type Step = EventStep | DecisionStep;

/** 牌譜のイベント。位置は、firstEvent に、この局でこれより前にあるイベントの手順の数を足した値。 */
export interface EventStep {
  event: EventKind;
  /** そのイベントの席。無ければ席の無いイベント（-1 として扱う）。 */
  actor?: number;
  /** 直前の手順の局面からの差分。無ければ局面は変わらない。 */
  delta?: StateDelta;
  /** hora と ryukyoku だけが持つ。 */
  outcome?: RoundOutcome;
}

/** 1 つの席の判断。候補は表示の順（最善手が先）に並ぶ。 */
export interface DecisionStep extends CandidateSet {
  seat: number;
  /** 判断のきっかけになったイベントの位置。無ければ、この判断の前にある最後のイベントの位置。 */
  cause?: number;
  /**
   * 判断の結果として起きたイベントの位置。無ければ cause + 1。牌譜が終わっても決着していない判断では、
   * イベントの数。
   */
  resolve?: number;
  /** 直前の手順の局面からの差分。前のバージョンが書いた文書にだけある。無ければ局面は変わらない。 */
  delta?: StateDelta;
}

/** 判断の候補。もしもの結果も同じ形。 */
export interface CandidateSet {
  candidates: readonly CandidateRow[];
  /**
   * 牌譜で選んだ候補の位置。無ければ、牌譜に選んだ手の無い判断（見えていない応答や、もしもで選べなくなった
   * 手）。
   */
  chosen?: number;
  /**
   * 表示する確率（候補の順）。前のバージョンが、温度から作れない評価にだけ書いた。あれば、温度から作らずに
   * これを使う。
   */
  probabilities?: readonly number[];
}

/**
 * 1 つの候補。値（損失か確率）は float32 を最短の十進数で書いたもの。牌を持つのは CALL_TYPES の手だけ。
 * 手の種類、ツモ切りか、打牌の牌は、行動 ID から actionOf で求める。
 */
export type CandidateRow =
  | readonly [actionId: number, value: number]
  | readonly [actionId: number, value: number, tiles: readonly string[]];

/** イベントの種類（mjai の名前）。start_game は手順に入れない。 */
export const EVENT_KINDS = [
  "start_kyoku",
  "tsumo",
  "dahai",
  "chi",
  "pon",
  "daiminkan",
  "ankan",
  "kakan",
  "kita",
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

/** 局面。players の添字がそのまま席の番号で、人数だけ並ぶ。 */
export interface TableState {
  players: readonly PlayerState[];
  /** めくれたドラ表示牌（めくれた順）。 */
  dora: readonly string[];
  honba: number;
  /** 卓に出ている供託の本数。 */
  kyotaku: number;
  /** 残りのツモの数。 */
  remaining: number;
  /** 手番の席。 */
  activeSeat: number;
}

export interface PlayerState {
  score: number;
  riichi: boolean;
  /** ツモ牌を除いた手牌。tileOrder の順に並ぶ。 */
  hand: readonly string[];
  /** 見えているツモ牌。他家の牌が見えない局面では、他家の値は null。 */
  draw: string | null;
  river: readonly RiverTile[];
  /** 副露を並べ、その後に三人打ちで抜いた北を 1 枚ずつ（NUKI）並べる。 */
  melds: readonly Meld[];
}

/** 河の 1 枚。印のフィールドは true のときだけ書く。 */
export interface RiverTile {
  tile: string;
  tsumogiri?: true;
  /** リーチの宣言牌。 */
  riichi?: true;
  /** 鳴かれた牌（河に残したまま数える）。 */
  called?: true;
}

export interface Meld {
  type: "CHI" | "PON" | "DAIMINKAN" | "ANKAN" | "KAKAN" | "NUKI";
  /** チー・ポンは 3 枚、槓は 4 枚、抜いた北（NUKI）は 1 枚。表示する順に並ぶ。 */
  tiles: readonly string[];
  /** 鳴いた牌を出した席の番号。暗槓と抜いた北では -1。 */
  from: number;
  /** 横に倒す牌の位置。暗槓と抜いた北では -1。 */
  calledIndex: number;
  /** 加槓で加えた牌の位置（3）。加槓以外では -1。 */
  addedIndex: number;
}

/**
 * 配列の 1 か所を置き換える [位置, 消す数, ...入れる値]。前の配列のその位置から消す数だけ要素を消し、
 * そこへ値を入れる。
 */
export type Splice<T> = readonly [
  index: number,
  removeCount: number,
  ...values: T[],
];

/** 局面の差分。変わったフィールドだけを書く。 */
export interface StateDelta {
  /** 変わった席だけを、席の順に並べる。 */
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
  /** 新しいツモ牌。null はツモ牌が無くなったことを表す。 */
  draw?: string | null;
  river?: Splice<RiverTile>;
  melds?: Splice<Meld>;
}

/** 局の結果。流局の理由と点の移動は持たない（画面は持ち点の差を表示する）。 */
export type RoundOutcome =
  | {
      type: "hora";
      actor: number;
      /** ロンされた席。ツモ和了なら actor と同じ。 */
      target: number;
      winningTile: string;
      /** 点数の内訳が牌譜に無く、計算でも求められなければ書かない。 */
      details?: WinDetails;
    }
  | { type: "ryukyoku" };

/**
 * 和了の点数の内訳。牌譜に内訳が無いときは、天鳳のルールで計算した点の移動が全ての席で牌譜と一致したときだけ、
 * 計算した値を書く。分からないフィールドは書かない。
 */
export interface WinDetails {
  han?: number;
  fu?: number;
  points?: number;
  /** 役満の倍数。 */
  yakuman?: number;
  yaku?: readonly Yaku[];
  /** 裏ドラ表示牌。空の配列は「裏ドラを見た記録があり 0 枚」、書かないときは「記録が無い」の意味。 */
  uraIndicators?: readonly string[];
}

export interface Yaku {
  code: string;
  han: number;
  yakuman: number;
}

/** 行動の種類。名前と並び順は Java の Action.Type と同じ。 */
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
  "NUKI",
] as const;
export type ActionType = (typeof ACTION_TYPES)[number];

/** 候補の行に牌を書く手の種類。ほかの手の牌は行動 ID で決まる。 */
export const CALL_TYPES: ReadonlySet<ActionType> = new Set<ActionType>([
  "CHI",
  "PON",
  "DAIMINKAN",
  "ANKAN",
  "KAKAN",
]);

/** 行動 ID だけで決まる手の中身。 */
export interface ActionShape {
  type: ActionType;
  tsumogiri: boolean;
  /** 打牌とリーチの打牌で切る牌（赤は r 付き）。ほかの手では null。 */
  tile: string | null;
}

const HONOR_CODES = ["E", "S", "W", "N", "P", "F", "C"] as const;
const RED_FIVES = ["5mr", "5pr", "5sr"] as const;

/** 牌種の番号（0..33。萬・筒・索の 1..9、東南西北白發中の順）を牌の記号にする。 */
export function tileTypeCode(type: number): string {
  if (type >= 27) return HONOR_CODES[type - 27]!;
  return `${(type % 9) + 1}${"mps"[Math.floor(type / 9)]}`;
}

/** 行動 ID の最大値（三人打ちの北抜き）。 */
export const MAXIMUM_ACTION_ID = 372;

/**
 * 行動 ID（Java の ActionSection の番号、0..372）を手の中身にする。番号の割り当ては ActionSection.java と同じ
 * 固定の並び（打牌 0..73、チー 74..154、ポン 155..191、大明槓 192..225、暗槓 226..259、加槓 260..293、
 * リーチ 294..367、ツモ 368、ロン 369、見送り 370、九種九牌 371、北抜き 372）。
 */
export function actionOf(actionId: number): ActionShape {
  if (!Number.isInteger(actionId) || actionId < 0 || actionId > MAXIMUM_ACTION_ID)
    throw new RangeError(`Invalid action ID: ${actionId}`);
  if (actionId < 74) return discard("DAHAI", actionId);
  if (actionId < 155) return { type: "CHI", tsumogiri: false, tile: null };
  if (actionId < 192) return { type: "PON", tsumogiri: false, tile: null };
  if (actionId < 226)
    return { type: "DAIMINKAN", tsumogiri: false, tile: null };
  if (actionId < 260) return { type: "ANKAN", tsumogiri: false, tile: null };
  if (actionId < 294) return { type: "KAKAN", tsumogiri: false, tile: null };
  if (actionId < 368) return discard("RIICHI_DAHAI", actionId - 294);
  const type = (
    ["TSUMO_AGARI", "RON_AGARI", "PASS", "KYUSHU_KYUHAI", "NUKI"] as const
  )[actionId - 368]!;
  return { type, tsumogiri: false, tile: null };
}

/** 打牌の 74 通り（手出し 34、ツモ切り 34、赤の手出し 3、赤のツモ切り 3）の中の 1 つを手の中身にする。 */
function discard(type: "DAHAI" | "RIICHI_DAHAI", offset: number): ActionShape {
  if (offset < 34)
    return { type, tsumogiri: false, tile: tileTypeCode(offset) };
  if (offset < 68)
    return { type, tsumogiri: true, tile: tileTypeCode(offset - 34) };
  if (offset < 71)
    return { type, tsumogiri: false, tile: RED_FIVES[offset - 68]! };
  return { type, tsumogiri: true, tile: RED_FIVES[offset - 71]! };
}

/** 候補の牌を返す。CALL_TYPES の手は行に書いた牌、打牌は行動 ID で決まる牌、ほかの手は空。 */
export function candidateTiles(row: CandidateRow): readonly string[] {
  if (row.length === 3) return row[2];
  const { tile } = actionOf(row[0]);
  return tile === null ? [] : [tile];
}

/**
 * 損失から表示する確率を作る: p_i = exp(-loss_i / temperature) / Σ_j exp(-loss_j / temperature)。
 * 和は候補の順に binary64 で足し、各値を float32 に丸める。probabilities を持つのは前のバージョンが書いた
 * 文書だけで、方策の確率とこの値の差が 1e-6 を超える判断に書いてある。
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

/** 判断の候補を表示する確率を、候補の順に返す。temperature には文書の temperature を渡す。 */
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

/** 判断の候補の損失を、候補の順に返す。価値を推定しないモデルなら null。 */
export function candidateLosses(
  set: CandidateSet,
  temperature: number | undefined,
): readonly number[] | null {
  return temperature === undefined ? null : set.candidates.map((row) => row[1]);
}

const TILE_PATTERN = /^(?:[1-9][mps]|5[mps]r|[ESWNPFC])$/;

/** 牌の記号かを返す（萬筒索の 1..9、赤は 5 にだけ r を付ける、字牌は ESWNPFC）。 */
export function isTileCode(value: unknown): value is string {
  return typeof value === "string" && TILE_PATTERN.test(value);
}

/** 手牌を並べるときの順の値。萬・筒・索の 1..9、東南西北白發中の順で、赤の 5 は同じ色の 5 の前に来る。 */
export function tileOrder(tile: string): number {
  const red = tile.length === 3;
  const honor = HONOR_CODES.indexOf(tile as (typeof HONOR_CODES)[number]);
  const type =
    honor >= 0 ? 27 + honor : "mps".indexOf(tile[1]!) * 9 + Number(tile[0]) - 1;
  return type * 2 + (red ? 0 : 1);
}

/**
 * その記号の牌が山に何枚あるかを返す（Java の TileCopies.count と同じ）。各色の 5 は 4 枚のうち 1 枚が赤なので、
 * 赤の 5 は 1 枚、赤でない 5 は 3 枚、ほかは 4 枚。三人打ち（seats が 3）には 2〜8 萬と赤の 5 萬が無い。
 */
export function tileCopies(tile: string, seats = 4): number {
  if (seats === 3 && tile[1] === "m" && tile[0] !== "1" && tile[0] !== "9")
    return 0;
  if (tile.length === 3) return 1;
  return tile[0] === "5" ? 3 : 4;
}

/** 解析結果を書き出すときのファイル名。 */
export function resultFileName(resultId: string): string {
  return `${resultId}.epsilon-reviewer.json.gz`;
}
