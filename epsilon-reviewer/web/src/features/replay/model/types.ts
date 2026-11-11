import type {
  ModelInfo,
  RecordSummary,
} from "../../../../../protocol/model.ts";
import type {
  ActionType,
  CandidateSet,
  EventKind,
  PlayerState,
  RoundOutcome,
  StateDelta,
  TableState,
} from "../../../../../protocol/result.ts";

// 読み込んだ解析結果。文書の配列やオブジェクトは、写さずにそのまま使う。局面は、見るときに stateReader が作る。

export interface ReplayModel {
  readonly resultId: string;
  readonly createdAt: string;
  readonly model: ModelInfo;
  readonly record: RecordSummary;
  /** 価値を推定するモデルで、表示に使う温度。温度が無ければ、候補の値は確率。 */
  readonly temperature: number | undefined;
  readonly rounds: readonly ReplayRound[];
}

export interface ReplayRound {
  /** ReplayModel.rounds の中で何番目の局か。もしもの要求で局を指すのに使う（要求の round）。 */
  readonly index: number;
  /** (場風) × 席の数 + (局 − 1)。 */
  readonly roundIndex: number;
  readonly honba: number;
  readonly dealer: number;
  /** 席の数（4 人か 3 人）。 */
  readonly seats: number;
  readonly start: TableState;
  /** 局面の差分の列。局面 k（k = 0..deltas.length）は、start に deltas[0..k-1] を順に適用したもの。 */
  readonly deltas: readonly StateDelta[];
  readonly steps: readonly ReplayStep[];
}

export type HoraOutcome = Extract<RoundOutcome, { type: "hora" }>;
export type DrawOutcome = Extract<RoundOutcome, { type: "ryukyoku" }>;
/** 局の結果を持たないイベントの種類。 */
export type PlainEventKind = Exclude<EventKind, "hora" | "ryukyoku">;

interface StepBase {
  /** 局の中で何番目の手順か。 */
  readonly index: number;
  /** 局面の番号（局の始めから適用した差分の数）。 */
  readonly frame: number;
}

interface EventBase extends StepBase {
  readonly decision: false;
  /** 牌譜の中で何番目のイベントか。 */
  readonly eventIndex: number;
  /** 動いた席。席の無いイベントでは -1。 */
  readonly actor: number;
}

/** 牌譜のイベント。局の結果を持つのは和了（hora）と流局（ryukyoku）だけ。 */
export type EventStep =
  | (EventBase & { readonly eventType: "hora"; readonly outcome: HoraOutcome })
  | (EventBase & {
      readonly eventType: "ryukyoku";
      readonly outcome: DrawOutcome;
    })
  | (EventBase & {
      readonly eventType: PlainEventKind;
      readonly outcome: null;
    });

/** 1 つの席の判断。 */
export interface DecisionStep extends StepBase {
  readonly decision: true;
  /** 判断の結果が出たイベントが、牌譜の中で何番目か。 */
  readonly eventIndex: number;
  /** 判断のきっかけになったイベントが、牌譜の中で何番目か。 */
  readonly causeEventIndex: number;
  /** 判断する席。 */
  readonly actor: number;
  readonly set: CandidateSet;
}

export type ReplayStep = EventStep | DecisionStep;

export interface PlayerSnapshot extends PlayerState {
  readonly seat: number;
  readonly name: string;
}

export interface TableSnapshot extends Omit<TableState, "players"> {
  readonly players: readonly PlayerSnapshot[];
}

/** 手順と、そのときの局面。 */
export type Snapshot = ReplayStep & TableSnapshot;
export type EventSnapshot = EventStep & TableSnapshot;

/** 表示用の候補。値は、文書にある候補の行から作る。 */
export interface Candidate {
  readonly actionId: number;
  readonly type: ActionType;
  readonly tiles: readonly string[];
  readonly probability: number;
  /** 最善手との価値の差。価値を推定しないモデルでは null。 */
  readonly valueLoss: number | null;
  readonly chosen: boolean;
  readonly tsumogiri: boolean;
}
