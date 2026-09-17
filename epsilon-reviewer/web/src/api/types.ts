export interface Model {
  modelId: string;
  revision: string;
  displayName: string;
  version: string;
  series: string;
}
export interface AvailableModel extends Model {
  availability: "available" | "unavailable";
  reason: string | null;
  reasonCode: string | null;
}
export interface ModelCatalog {
  models: readonly AvailableModel[];
  defaultModel: Pick<Model, "modelId" | "revision"> | null;
}
export interface RecordMetadata {
  source: string;
  fileName?: string;
  names: readonly string[];
  finalScores: readonly number[] | null;
  finalScoresSource?: string;
  roundCount: number;
}
export interface ImportedRecord {
  recordId: string;
  metadata: RecordMetadata;
}
export interface HistoryEntry {
  resultId: string;
  createdAt: string;
  metadata: RecordMetadata;
  model: Model;
}
export interface ShareLink {
  url: string;
}
export interface RiverTile {
  tile: string;
  tsumogiri: boolean;
  riichi: boolean;
  called: boolean;
}
export interface Meld {
  type: "CHI" | "PON" | "DAIMINKAN" | "ANKAN" | "KAKAN";
  tiles: readonly string[];
  from: number;
  calledIndex: number;
  addedIndex: number;
}
export interface PlayerSnapshot extends PlayerState {
  seat: number;
  name: string;
}
export type ActionType =
  | "DAHAI"
  | "CHI"
  | "PON"
  | "DAIMINKAN"
  | "ANKAN"
  | "KAKAN"
  | "RIICHI_DAHAI"
  | "TSUMO_AGARI"
  | "RON_AGARI"
  | "PASS"
  | "KYUSHU_KYUHAI";
export interface Candidate {
  actionId: number;
  type: ActionType;
  tiles: readonly string[];
  probability: number;
  chosen: boolean;
  tsumogiri: boolean;
}
export interface WinDetails {
  han: number | null;
  fu: number | null;
  points: number | null;
  yakuman: number | null;
  yaku: readonly { code: string; han: number; yakuman: number }[] | null;
  uraIndicators: readonly string[] | null;
}
export type RoundOutcome =
  | {
      type: "hora";
      actor: number;
      target: number;
      winningTile: string;
      deltas: readonly number[] | null;
      details: WinDetails | null;
    }
  | {
      type: "ryukyoku";
      actor: null;
      target: null;
      winningTile: null;
      deltas: readonly number[] | null;
      details: null;
    };
interface StepFields {
  index: number;
  eventIndex: number;
  stateId: number;
  eventType: string;
  actor: number;
  candidates: readonly Candidate[];
  outcome: RoundOutcome | null;
}
export type ReviewStep = StepFields &
  (
    | { decision: true; causeEventIndex: number }
    | { decision: false; causeEventIndex: null }
  );
export interface PlayerState {
  score: number;
  riichi: boolean;
  hand: readonly string[];
  draw: string | null;
  river: readonly RiverTile[];
  melds: readonly Meld[];
}
export interface TableState {
  players: readonly PlayerState[];
  dora: readonly string[];
  honba: number;
  kyotaku: number;
  remaining: number;
  activeSeat: number;
}
export interface TableSnapshot extends Omit<TableState, "players"> {
  players: readonly PlayerSnapshot[];
}
export type Snapshot = ReviewStep & TableSnapshot;
export interface ArrayPatch<T> {
  index: number;
  removeCount: number;
  values: readonly T[];
}
export interface PlayerDelta {
  seat: number;
  score: number | null;
  riichi: boolean | null;
  hand: ArrayPatch<string> | null;
  draw: { tile: string | null } | null;
  river: ArrayPatch<RiverTile> | null;
  melds: ArrayPatch<Meld> | null;
}
export interface StateDelta {
  players: readonly PlayerDelta[];
  dora: ArrayPatch<string> | null;
  honba: number | null;
  kyotaku: number | null;
  remaining: number | null;
  activeSeat: number | null;
}
export interface StateEntry {
  stateId: number;
  checkpoint: TableState | null;
  delta: StateDelta | null;
}
export interface ReviewRound {
  id: string;
  roundIndex: number;
  honba: number;
  dealer: number;
  states: readonly StateEntry[];
  steps: readonly ReviewStep[];
}
export interface ReviewResult extends HistoryEntry {
  formatVersion: 4;
  rounds: readonly ReviewRound[];
}
export interface Job {
  jobId: string;
  status:
    "queued" | "running" | "saving" | "completed" | "failed" | "cancelled";
  progress: number;
  resultId: string | null;
  error: { code: string; message: string } | null;
}
export interface BridgeRequest {
  path: string;
  method: "GET" | "POST" | "DELETE";
  body?: unknown;
  fileName?: string;
}
export interface DesktopSettings {
  serverUrl: string;
  resultsDirectory: string;
  language: string;
}
declare global {
  interface Window {
    epsilonReviewer?: {
      request: (
        request: BridgeRequest,
      ) => Promise<{ status: number; body: unknown }>;
      getSettings: () => Promise<DesktopSettings>;
      setServerUrl: (url: string) => Promise<DesktopSettings>;
      setLanguage: (language: string) => Promise<DesktopSettings>;
      chooseResultsDirectory: () => Promise<DesktopSettings>;
    };
  }
}
