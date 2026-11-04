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

// 読んだ解析結果。文書の配列と物は借りて写さず、局面は stateReader が見る時に作る。

export interface ReplayModel {
  readonly resultId: string;
  readonly createdAt: string;
  readonly model: ModelInfo;
  readonly record: RecordSummary;
  /** 価値を推定するモデルの表示の温度。無ければ候補の値は確率。 */
  readonly temperature: number | undefined;
  readonly rounds: readonly ReplayRound[];
}

export interface ReplayRound {
  /** ReplayModel.rounds の位置（もしもの要求の round）。 */
  readonly index: number;
  /** (場風) × 4 + (局 − 1)。 */
  readonly roundIndex: number;
  readonly honba: number;
  readonly dealer: number;
  readonly start: TableState;
  /** 局面の差分の並び。局面 k（k = 0..deltas.length）は start に deltas[0..k-1] を当てたもの。 */
  readonly deltas: readonly StateDelta[];
  readonly steps: readonly ReplayStep[];
}

export type HoraOutcome = Extract<RoundOutcome, { type: "hora" }>;
export type DrawOutcome = Extract<RoundOutcome, { type: "ryukyoku" }>;
/** 局の結果を持たない出来事。 */
export type PlainEventKind = Exclude<EventKind, "hora" | "ryukyoku">;

interface StepBase {
  /** 局の手順の位置。 */
  readonly index: number;
  /** 局面の番号（局の始めから当てた差分の数）。 */
  readonly frame: number;
}

interface EventBase extends StepBase {
  readonly decision: false;
  /** 牌譜の出来事の位置。 */
  readonly eventIndex: number;
  /** 行った席。席の無い出来事は -1。 */
  readonly actor: number;
}

/** 牌譜の出来事。hora と ryukyoku だけが局の結果を持つ。 */
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

/** 1 席の判断。eventIndex は解かれた出来事、causeEventIndex は起こした出来事の位置。 */
export interface DecisionStep extends StepBase {
  readonly decision: true;
  readonly eventIndex: number;
  readonly causeEventIndex: number;
  /** 判断の席。 */
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

/** 手順と、その時の局面。 */
export type Snapshot = ReplayStep & TableSnapshot;
export type EventSnapshot = EventStep & TableSnapshot;

/** 表示の候補。値は文書の行から作る。 */
export interface Candidate {
  readonly actionId: number;
  readonly type: ActionType;
  readonly tiles: readonly string[];
  readonly probability: number;
  /** 最善手との価値差。価値を推定しないモデルでは null。 */
  readonly valueLoss: number | null;
  readonly chosen: boolean;
  readonly tsumogiri: boolean;
}
