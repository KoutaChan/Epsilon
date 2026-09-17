import type {
  ArrayPatch,
  PlayerSnapshot,
  ReviewResult,
  ReviewStep,
  Snapshot,
  StateDelta,
  TableSnapshot,
  TableState,
} from "../../api/types";

export const CHECKPOINT_INTERVAL = 32;
const MAX_CACHED_CHUNKS = 4;

function applyArrayPatch<T>(values: readonly T[], patch: ArrayPatch<T> | null) {
  if (patch === null) return values;
  return [
    ...values.slice(0, patch.index),
    ...patch.values,
    ...values.slice(patch.index + patch.removeCount),
  ];
}

function applyStateDelta(
  state: TableSnapshot,
  delta: StateDelta,
): TableSnapshot {
  const players = delta.players.length ? state.players.slice() : state.players;
  for (const change of delta.players) {
    const previous = players[change.seat];
    (players as PlayerSnapshot[])[change.seat] = {
      seat: previous.seat,
      name: previous.name,
      score: change.score ?? previous.score,
      riichi: change.riichi ?? previous.riichi,
      hand: applyArrayPatch(previous.hand, change.hand),
      draw: change.draw === null ? previous.draw : change.draw.tile,
      river: applyArrayPatch(previous.river, change.river),
      melds: applyArrayPatch(previous.melds, change.melds),
    };
  }
  return {
    players,
    dora: applyArrayPatch(state.dora, delta.dora),
    honba: delta.honba ?? state.honba,
    kyotaku: delta.kyotaku ?? state.kyotaku,
    remaining: delta.remaining ?? state.remaining,
    activeSeat: delta.activeSeat ?? state.activeSeat,
  };
}

// 復元した状態の寿命をチャンク単位で管理し、未表示の局の手牌・河は展開しない。
export function createReplayStateReader(result: ReviewResult) {
  const ordinals = result.rounds.map(
    (round) =>
      new Map(round.states.map((entry, index) => [entry.stateId, index])),
  );
  const chunks = new Map<
    string,
    {
      states: TableSnapshot[];
      snapshots: Map<ReviewStep, Snapshot>;
    }
  >();

  function readState(roundIndex: number, stateId: number): TableSnapshot {
    const ordinal = ordinals[roundIndex].get(stateId)!;
    const start = ordinal - (ordinal % CHECKPOINT_INTERVAL);
    const key = `${roundIndex}:${start}`;
    let chunk = chunks.get(key);
    if (chunk) {
      chunks.delete(key);
      chunks.set(key, chunk);
    } else {
      const initial = result.rounds[roundIndex].states[start].checkpoint!;
      chunk = {
        states: [
          {
            ...initial,
            players: initial.players.map((player, seat) => ({
              ...player,
              seat,
              name: result.metadata.names[seat],
            })),
          },
        ],
        snapshots: new Map(),
      };
      chunks.set(key, chunk);
      if (chunks.size > MAX_CACHED_CHUNKS)
        chunks.delete(chunks.keys().next().value!);
    }
    const states = result.rounds[roundIndex].states;
    for (let index = chunk.states.length; index <= ordinal - start; index++)
      chunk.states.push(
        applyStateDelta(chunk.states[index - 1], states[start + index].delta!),
      );
    return chunk.states[ordinal - start];
  }

  function readSnapshot(roundIndex: number, step: ReviewStep): Snapshot {
    const table = readState(roundIndex, step.stateId);
    const ordinal = ordinals[roundIndex].get(step.stateId)!;
    const start = ordinal - (ordinal % CHECKPOINT_INTERVAL);
    const snapshots = chunks.get(`${roundIndex}:${start}`)!.snapshots;
    const existing = snapshots.get(step);
    if (existing) return existing;
    const snapshot = { ...step, ...table };
    snapshots.set(step, snapshot);
    return snapshot;
  }

  return { readState, readSnapshot };
}

// 固定カメラに必要な副露だけを走査する。手牌・河の差分は復元しない。
export function maximumPlayerRowWidth(
  result: ReviewResult,
  measure: (melds: TableState["players"][number]["melds"]) => number,
): number {
  let width = measure([]);
  for (const round of result.rounds) {
    let melds: TableState["players"][number]["melds"][] = [];
    for (const entry of round.states) {
      if (entry.checkpoint) {
        melds = entry.checkpoint.players.map((player) => player.melds);
        for (const row of melds) width = Math.max(width, measure(row));
      } else {
        for (const change of entry.delta!.players) {
          if (change.melds !== null) {
            melds[change.seat] = applyArrayPatch(
              melds[change.seat],
              change.melds,
            );
            width = Math.max(width, measure(melds[change.seat]));
          }
        }
      }
    }
  }
  return width + 0.4;
}
