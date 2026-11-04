import type {
  ModelInfo,
  RecordSummary,
} from "../../../../../protocol/model.ts";
import {
  CALL_TYPES,
  EVENT_KINDS,
  RESULT_FORMAT,
  actionOf,
  isTileCode,
  type CandidateSet,
  type EventKind,
  type Meld,
  type RiverTile,
  type RoundOutcome,
  type StateDelta,
  type TableState,
} from "../../../../../protocol/result.ts";
import { ReviewFormatError } from "./formatError.ts";
import type {
  DecisionStep,
  EventStep,
  ReplayModel,
  ReplayRound,
  ReplayStep,
} from "./types.ts";

// 解析結果の文書を 1 回だけ流して確かめ、手順に牌譜の位置と局面の番号を付ける。局面はここでは作らない。

type Fields = Readonly<Record<string, unknown>>;
interface Lengths {
  readonly players: { hand: number; river: number; melds: number }[];
  dora: number;
}

const EVENTS: ReadonlySet<string> = new Set(EVENT_KINDS);
const MELD_SIZES: Readonly<Record<Meld["type"], number>> = {
  CHI: 3,
  PON: 3,
  DAIMINKAN: 4,
  ANKAN: 4,
  KAKAN: 4,
};
const MAXIMUM_ACTION_ID = 371;

/** @throws ReviewFormatError 解析結果の文書として読めない */
export function decodeResult(value: unknown): ReplayModel {
  const document = object(value);
  if (document.format !== RESULT_FORMAT) fail("Unsupported result format.");
  const temperature =
    document.temperature === undefined
      ? undefined
      : positive(document.temperature);
  const rounds = array(document.rounds);
  if (rounds.length === 0) fail("The result has no rounds.");
  return {
    resultId: text(document.resultId),
    createdAt: text(document.createdAt),
    model: modelInfo(document.model),
    record: recordSummary(document.record),
    temperature,
    rounds: rounds.map((round, index) =>
      decodeRound(round, index, temperature !== undefined),
    ),
  };
}

function decodeRound(
  value: unknown,
  index: number,
  valued: boolean,
): ReplayRound {
  const round = object(value);
  const firstEvent = count(round.firstEvent);
  const start = tableState(round.start);
  const lengths: Lengths = {
    players: start.players.map((player) => ({
      hand: player.hand.length,
      river: player.river.length,
      melds: player.melds.length,
    })),
    dora: start.dora.length,
  };
  const source = array(round.steps);
  if (source.length === 0) fail("A round has no steps.");
  const steps: ReplayStep[] = [];
  const deltas: StateDelta[] = [];
  let events = 0;
  let position = 0;
  try {
    for (; position < source.length; position++) {
      const step = object(source[position]);
      if (step.delta !== undefined)
        deltas.push(stateDelta(step.delta, lengths));
      if (step.event !== undefined) {
        steps.push(
          eventStep(step, position, firstEvent + events, deltas.length),
        );
        events++;
      } else {
        if (events === 0) fail("A round must start with an event.");
        steps.push(
          decisionStep(
            step,
            position,
            firstEvent,
            firstEvent + events - 1,
            deltas.length,
            valued,
          ),
        );
      }
    }
  } catch (error) {
    if (error instanceof ReviewFormatError)
      fail(`rounds[${index}].steps[${position}]: ${error.message}`);
    throw error;
  }
  return {
    index,
    roundIndex: count(round.roundIndex),
    honba: count(round.honba),
    dealer: seat(round.dealer),
    start,
    deltas,
    steps,
  };
}

function eventStep(
  step: Fields,
  index: number,
  eventIndex: number,
  frame: number,
): EventStep {
  if (step.seat !== undefined)
    fail("A step cannot be both an event and a decision.");
  const eventType = eventKind(step.event);
  const outcome =
    step.outcome === undefined ? null : roundOutcome(step.outcome);
  const settles = eventType === "hora" || eventType === "ryukyoku";
  if (settles ? outcome?.type !== eventType : outcome !== null)
    fail("Only hora and ryukyoku events carry their own outcome.");
  const actor = step.actor === undefined ? -1 : seat(step.actor);
  // 種類と結果の組は上で確かめた。
  return {
    index,
    decision: false,
    eventIndex,
    eventType,
    actor,
    frame,
    outcome,
  } as EventStep;
}

function decisionStep(
  step: Fields,
  index: number,
  firstEvent: number,
  lastEvent: number,
  frame: number,
  valued: boolean,
): DecisionStep {
  const actor = seat(step.seat);
  const cause = step.cause === undefined ? lastEvent : count(step.cause);
  if (cause < firstEvent || cause > lastEvent)
    fail("A decision cause must be an earlier event of its round.");
  const resolve = step.resolve === undefined ? cause + 1 : count(step.resolve);
  candidateSet(step, valued);
  return {
    index,
    decision: true,
    eventIndex: resolve,
    causeEventIndex: cause,
    actor,
    frame,
    set: step as unknown as CandidateSet,
  };
}

function candidateSet(set: Fields, valued: boolean): void {
  const rows = array(set.candidates);
  if (rows.length === 0) fail("A decision has no candidates.");
  for (const row of rows) candidateRow(row, valued);
  if (set.chosen !== undefined && count(set.chosen) >= rows.length)
    fail("The chosen candidate is out of range.");
  if (set.probabilities !== undefined) {
    const probabilities = array(set.probabilities);
    if (probabilities.length !== rows.length)
      fail("The probabilities do not match the candidates.");
    for (const probability of probabilities) unit(probability);
  }
}

function candidateRow(value: unknown, valued: boolean): void {
  const row = array(value);
  const actionId = count(row[0]);
  if (actionId > MAXIMUM_ACTION_ID) fail("A candidate has an invalid action.");
  if (valued) loss(row[1]);
  else unit(row[1]);
  const call = CALL_TYPES.has(actionOf(actionId).type);
  if (row.length !== (call ? 3 : 2))
    fail("Only call candidates carry their tiles.");
  if (call && tileList(row[2]).length === 0)
    fail("A call candidate has no tiles.");
}

function tableState(value: unknown): TableState {
  const table = object(value);
  const players = array(table.players);
  if (players.length !== 4) fail("A table must have four players.");
  for (const player of players) playerState(player);
  tileList(table.dora);
  count(table.honba);
  count(table.kyotaku);
  count(table.remaining);
  seat(table.activeSeat);
  return value as TableState;
}

function playerState(value: unknown): void {
  const player = object(value);
  integer(player.score);
  boolean(player.riichi);
  tileList(player.hand);
  if (player.draw !== null) tile(player.draw);
  for (const discard of array(player.river)) riverTile(discard);
  for (const entry of array(player.melds)) meld(entry);
}

function stateDelta(value: unknown, lengths: Lengths): StateDelta {
  const delta = object(value);
  if (delta.players !== undefined) {
    let previous = -1;
    for (const entry of array(delta.players)) {
      const change = object(entry);
      const at = seat(change.seat);
      if (at <= previous)
        fail("Player changes must follow the seat order once each.");
      previous = at;
      if (change.score !== undefined) integer(change.score);
      if (change.riichi !== undefined) boolean(change.riichi);
      if (change.draw !== undefined && change.draw !== null) tile(change.draw);
      const length = lengths.players[at]!;
      if (change.hand !== undefined)
        length.hand = splice(change.hand, length.hand, tile);
      if (change.river !== undefined)
        length.river = splice(change.river, length.river, riverTile);
      if (change.melds !== undefined)
        length.melds = splice(change.melds, length.melds, meld);
    }
  }
  if (delta.dora !== undefined)
    lengths.dora = splice(delta.dora, lengths.dora, tile);
  if (delta.honba !== undefined) count(delta.honba);
  if (delta.kyotaku !== undefined) count(delta.kyotaku);
  if (delta.remaining !== undefined) count(delta.remaining);
  if (delta.activeSeat !== undefined) seat(delta.activeSeat);
  return value as StateDelta;
}

/** 範囲を前の配列の長さで確かめ、置き換えた後の長さを返す。 */
function splice(
  value: unknown,
  length: number,
  item: (value: unknown) => unknown,
): number {
  const change = array(value);
  const start = count(change[0]);
  const removeCount = count(change[1]);
  if (start + removeCount > length)
    fail("A splice is outside the previous array.");
  for (let position = 2; position < change.length; position++)
    item(change[position]);
  return length - removeCount + change.length - 2;
}

function riverTile(value: unknown): RiverTile {
  const discard = object(value);
  tile(discard.tile);
  for (const mark of [discard.tsumogiri, discard.riichi, discard.called])
    if (mark !== undefined && mark !== true)
      fail("A river mark is written only when true.");
  return value as RiverTile;
}

function meld(value: unknown): Meld {
  const entry = object(value);
  const type = entry.type;
  if (typeof type !== "string" || !Object.hasOwn(MELD_SIZES, type))
    fail("Unsupported meld.");
  const size = MELD_SIZES[type as Meld["type"]];
  if (tileList(entry.tiles).length !== size) fail("Invalid meld tile count.");
  const from = integer(entry.from);
  const called = integer(entry.calledIndex);
  const added = integer(entry.addedIndex);
  if (
    type === "ANKAN"
      ? from !== -1 || called !== -1
      : from < 0 || from > 3 || called < 0 || called >= size
  )
    fail("Invalid called tile of a meld.");
  if (
    type === "KAKAN"
      ? added < 0 || added >= size || added === called
      : added !== -1
  )
    fail("Invalid added tile of a meld.");
  return value as Meld;
}

function roundOutcome(value: unknown): RoundOutcome {
  const outcome = object(value);
  if (outcome.type === "ryukyoku") return value as RoundOutcome;
  if (outcome.type !== "hora") fail("Unsupported round outcome.");
  seat(outcome.actor);
  seat(outcome.target);
  tile(outcome.winningTile);
  if (outcome.details !== undefined) {
    const details = object(outcome.details);
    if (details.han !== undefined) count(details.han);
    if (details.fu !== undefined) count(details.fu);
    if (details.points !== undefined) count(details.points);
    if (details.yakuman !== undefined) count(details.yakuman);
    if (details.yaku !== undefined)
      for (const entry of array(details.yaku)) {
        const yaku = object(entry);
        text(yaku.code);
        count(yaku.han);
        count(yaku.yakuman);
      }
    if (details.uraIndicators !== undefined) tileList(details.uraIndicators);
  }
  return value as RoundOutcome;
}

function modelInfo(value: unknown): ModelInfo {
  const model = object(value);
  text(model.modelId);
  text(model.revision);
  text(model.displayName);
  text(model.version);
  return value as ModelInfo;
}

function recordSummary(value: unknown): RecordSummary {
  const record = object(value);
  text(record.source);
  const names = array(record.names);
  if (names.length !== 4) fail("The record must name four players.");
  for (const name of names) text(name);
  if (record.finalScores !== undefined) {
    const scores = array(record.finalScores);
    if (scores.length !== 4) fail("The record must have four final scores.");
    for (const score of scores) integer(score);
  }
  count(record.roundCount);
  return value as RecordSummary;
}

function eventKind(value: unknown): EventKind {
  if (typeof value !== "string" || !EVENTS.has(value))
    fail("Unsupported event.");
  return value as EventKind;
}

function tileList(value: unknown): readonly unknown[] {
  const tiles = array(value);
  for (const entry of tiles) tile(entry);
  return tiles;
}

function tile(value: unknown): string {
  if (!isTileCode(value)) fail(`Invalid tile code: ${String(value)}.`);
  return value;
}

function seat(value: unknown): number {
  if (value !== 0 && value !== 1 && value !== 2 && value !== 3)
    fail("A seat must be 0 to 3.");
  return value;
}

/** 0 以上の整数（数・配列の位置・行動 ID）。 */
function count(value: unknown): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0)
    fail("A count or position must be a non-negative integer.");
  return value as number;
}

function integer(value: unknown): number {
  if (!Number.isSafeInteger(value)) fail("A value must be an integer.");
  return value as number;
}

function unit(value: unknown): void {
  if (typeof value !== "number" || !(value >= 0 && value <= 1))
    fail("A probability must be between 0 and 1.");
}

function loss(value: unknown): void {
  if (typeof value !== "number" || !(value >= 0 && value < Infinity))
    fail("A value loss must be a finite non-negative number.");
}

function positive(value: unknown): number {
  if (typeof value !== "number" || !(value > 0 && value < Infinity))
    fail("The temperature must be a positive number.");
  return value;
}

function boolean(value: unknown): void {
  if (typeof value !== "boolean") fail("A flag must be true or false.");
}

function text(value: unknown): string {
  if (typeof value !== "string") fail("A text field must be a string.");
  return value;
}

function object(value: unknown): Fields {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    fail("Expected an object.");
  return value as Fields;
}

function array(value: unknown): readonly unknown[] {
  if (!Array.isArray(value)) fail("Expected an array.");
  return value;
}

function fail(message: string): never {
  throw new ReviewFormatError(message);
}
