import type { ReviewResult } from "./types";
import { ApiError } from "./errors.ts";

function throwInvalidReview(message: string): never {
  throw new ApiError(message, "invalid_result");
}
function requireObject(value: unknown): Record<string, any> {
  if (value === null || typeof value !== "object" || Array.isArray(value))
    throwInvalidReview("The review contains an invalid object.");
  return value as Record<string, any>;
}
function requireArray(value: unknown): any[] {
  if (!Array.isArray(value))
    throwInvalidReview("The review contains an invalid array.");
  return value;
}
function requireInteger(value: unknown): number {
  if (!Number.isSafeInteger(value))
    throwInvalidReview("The review contains an invalid integer.");
  return value as number;
}
function requireSeat(value: unknown): number {
  const seat = requireInteger(value);
  if (seat < 0 || seat > 3)
    throwInvalidReview("The review contains an invalid player seat.");
  return seat;
}
function requireSupportedTileCode(value: unknown): void {
  if (typeof value !== "string" || !/^(?:[1-9][mps]r?|[ESWNPFC])$/.test(value))
    throwInvalidReview("The review contains an invalid tile.");
}
function requireBoolean(value: unknown): void {
  if (typeof value !== "boolean")
    throwInvalidReview("The review contains an invalid boolean.");
}
function requireRiverTileFields(value: unknown): void {
  const tile = requireObject(value);
  requireSupportedTileCode(tile.tile);
  for (const field of ["tsumogiri", "riichi", "called"])
    requireBoolean(tile[field]);
}
function requireMeldTileLayout(value: unknown): void {
  const meld = requireObject(value);
  if (!["CHI", "PON", "DAIMINKAN", "ANKAN", "KAKAN"].includes(meld.type))
    throwInvalidReview("The review contains an unsupported meld.");
  const tiles = requireArray(meld.tiles);
  const count = ["CHI", "PON"].includes(meld.type) ? 3 : 4;
  if (tiles.length !== count)
    throwInvalidReview("The meld tile count is invalid.");
  tiles.forEach(requireSupportedTileCode);
  requireInteger(meld.from);
  const called = requireInteger(meld.calledIndex),
    added = requireInteger(meld.addedIndex);
  if (meld.type === "ANKAN" ? called !== -1 : called < 0 || called >= count)
    throwInvalidReview("The called tile position is invalid.");
  if (
    meld.type === "KAKAN"
      ? added < 0 || added >= count || added === called
      : added !== -1
  )
    throwInvalidReview("The added tile position is invalid.");
}
function requirePatchLength(
  value: unknown,
  length: number,
  requireItem: (value: unknown) => void,
): number {
  if (value === null) return length;
  const patch = requireObject(value);
  const index = requireInteger(patch.index),
    remove = requireInteger(patch.removeCount);
  const values = requireArray(patch.values);
  if (index < 0 || remove < 0 || index > length || index + remove > length)
    throwInvalidReview("The review array patch is outside the previous array.");
  values.forEach(requireItem);
  return length - remove + values.length;
}
function requireMatchingEventOutcome(step: Record<string, any>): void {
  if (step.eventType !== "hora" && step.eventType !== "ryukyoku") {
    if (step.outcome !== null)
      throwInvalidReview("The non-settlement event has an outcome.");
    return;
  }
  const outcome = requireObject(step.outcome);
  if (outcome.type !== step.eventType)
    throwInvalidReview("The settlement type does not match its event.");
  if (outcome.deltas !== null) {
    const deltas = requireArray(outcome.deltas);
    if (deltas.length !== 4)
      throwInvalidReview("The settlement score count is invalid.");
    deltas.forEach(requireInteger);
  }
  if (outcome.type === "ryukyoku") {
    if (
      outcome.actor !== null ||
      outcome.target !== null ||
      outcome.winningTile !== null ||
      outcome.details !== null
    )
      throwInvalidReview("The drawn round contains winning-hand information.");
    return;
  }
  requireSeat(outcome.actor);
  requireSeat(outcome.target);
  requireSupportedTileCode(outcome.winningTile);
  if (outcome.details === null) return;
  const details = requireObject(outcome.details);
  for (const field of ["han", "fu", "points", "yakuman"])
    if (details[field] !== null) requireInteger(details[field]);
  if (details.uraIndicators !== null)
    requireArray(details.uraIndicators).forEach(requireSupportedTileCode);
  if (details.yaku !== null)
    for (const raw of requireArray(details.yaku)) {
      const yaku = requireObject(raw);
      if (typeof yaku.code !== "string")
        throwInvalidReview("The yaku identifier is invalid.");
      requireInteger(yaku.han);
      requireInteger(yaku.yakuman);
    }
}

// 参照と差分の範囲を入口で確認し、全局の手牌・河を復元しない。
export function readCurrentReviewResult(value: unknown): ReviewResult {
  const result = requireObject(value);
  if (result.formatVersion !== 4)
    throwInvalidReview("Unsupported review result format.");
  const names = requireArray(requireObject(result.metadata).names);
  if (names.length !== 4 || names.some((name) => typeof name !== "string"))
    throwInvalidReview("The review player names are invalid.");
  const rounds = requireArray(result.rounds);
  if (!rounds.length) throwInvalidReview("The review result has no rounds.");
  for (const rawRound of rounds) {
    const round = requireObject(rawRound);
    requireInteger(round.roundIndex);
    requireInteger(round.honba);
    requireSeat(round.dealer);
    const states = requireArray(round.states),
      steps = requireArray(round.steps);
    if (!states.length || !steps.length)
      throwInvalidReview("The review round has no states or steps.");
    const ids = new Set<number>();
    let previousId = -1,
      doraLength = 0;
    let lengths: { hand: number; river: number; melds: number }[] = [];
    for (let ordinal = 0; ordinal < states.length; ordinal++) {
      const entry = requireObject(states[ordinal]),
        id = requireInteger(entry.stateId);
      if (id <= previousId)
        throwInvalidReview("Review state IDs must increase within a round.");
      previousId = id;
      ids.add(id);
      if (ordinal % 32 === 0) {
        if (entry.delta !== null)
          throwInvalidReview("A checkpoint cannot contain a delta.");
        const state = requireObject(entry.checkpoint),
          players = requireArray(state.players);
        if (players.length !== 4)
          throwInvalidReview("The checkpoint must contain four players.");
        lengths = players.map((raw) => {
          const player = requireObject(raw);
          requireInteger(player.score);
          requireBoolean(player.riichi);
          if (player.draw !== null) requireSupportedTileCode(player.draw);
          const hand = requireArray(player.hand),
            river = requireArray(player.river),
            melds = requireArray(player.melds);
          hand.forEach(requireSupportedTileCode);
          river.forEach(requireRiverTileFields);
          melds.forEach(requireMeldTileLayout);
          return {
            hand: hand.length,
            river: river.length,
            melds: melds.length,
          };
        });
        const dora = requireArray(state.dora);
        dora.forEach(requireSupportedTileCode);
        doraLength = dora.length;
        for (const field of ["honba", "kyotaku", "remaining"])
          requireInteger(state[field]);
        requireSeat(state.activeSeat);
      } else {
        if (entry.checkpoint !== null)
          throwInvalidReview("A delta entry cannot contain a checkpoint.");
        const delta = requireObject(entry.delta),
          changed = new Set<number>();
        for (const raw of requireArray(delta.players)) {
          const player = requireObject(raw),
            seat = requireSeat(player.seat);
          if (changed.has(seat))
            throwInvalidReview("A player is changed twice in one state delta.");
          changed.add(seat);
          if (player.score !== null) requireInteger(player.score);
          if (player.riichi !== null) requireBoolean(player.riichi);
          if (player.draw !== null && requireObject(player.draw).tile !== null)
            requireSupportedTileCode(player.draw.tile);
          const length = lengths[seat];
          length.hand = requirePatchLength(
            player.hand,
            length.hand,
            requireSupportedTileCode,
          );
          length.river = requirePatchLength(
            player.river,
            length.river,
            requireRiverTileFields,
          );
          length.melds = requirePatchLength(
            player.melds,
            length.melds,
            requireMeldTileLayout,
          );
        }
        doraLength = requirePatchLength(
          delta.dora,
          doraLength,
          requireSupportedTileCode,
        );
        for (const field of ["honba", "kyotaku", "remaining"])
          if (delta[field] !== null) requireInteger(delta[field]);
        if (delta.activeSeat !== null) requireSeat(delta.activeSeat);
      }
    }
    const events = new Set(
      steps
        .filter((step) => step?.decision === false)
        .map((step) => step.eventIndex),
    );
    for (const raw of steps) {
      const step = requireObject(raw);
      requireInteger(step.index);
      requireInteger(step.eventIndex);
      if (!ids.has(step.stateId))
        throwInvalidReview("The replay step references an unavailable state.");
      requireBoolean(step.decision);
      if (
        step.decision
          ? !Number.isSafeInteger(step.causeEventIndex) ||
            !events.has(step.causeEventIndex)
          : step.causeEventIndex !== null
      )
        throwInvalidReview("The replay decision cause is invalid.");
      if (typeof step.eventType !== "string")
        throwInvalidReview("The replay event type is invalid.");
      if (step.actor !== -1) requireSeat(step.actor);
      for (const rawCandidate of requireArray(step.candidates)) {
        const candidate = requireObject(rawCandidate);
        requireInteger(candidate.actionId);
        if (
          ![
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
          ].includes(candidate.type)
        )
          throwInvalidReview("The review contains an unsupported action.");
        if (
          !Number.isFinite(candidate.probability) ||
          candidate.probability < 0 ||
          candidate.probability > 1
        )
          throwInvalidReview("The review action probability is invalid.");
        requireArray(candidate.tiles).forEach(requireSupportedTileCode);
        requireBoolean(candidate.chosen);
        requireBoolean(candidate.tsumogiri);
      }
      requireMatchingEventOutcome(step);
    }
  }
  return value as ReviewResult;
}
