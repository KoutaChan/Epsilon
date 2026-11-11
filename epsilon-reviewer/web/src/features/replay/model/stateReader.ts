import type {
  Meld,
  Splice,
  StateDelta,
} from "../../../../../protocol/result.ts";
import type {
  PlayerSnapshot,
  ReplayModel,
  ReplayRound,
  ReplayStep,
  TableSnapshot,
} from "./types.ts";

/** 局面を作って覚えておく局の数。 */
export const CACHED_ROUNDS = 3;

export interface StateReader {
  /** 手順と、そのときの局面を返す。局がキャッシュから外れるまでは、同じ手順を何度読んでも、局面の席と配列は同じオブジェクトになる。 */
  readSnapshot<S extends ReplayStep>(round: number, step: S): S & TableSnapshot;
}

/**
 * 局面は、その局を初めて見たときに局の分を全て作り、最近見た {@link CACHED_ROUNDS} 局だけをキャッシュに持つ。変わらない席と
 * 配列は、前の局面と同じオブジェクトを使い回す。見ていない局の手牌・河は作らない。
 */
export function createStateReader(model: ReplayModel): StateReader {
  const names = model.record.names;
  const rounds = new Map<number, readonly TableSnapshot[]>();

  function tables(index: number): readonly TableSnapshot[] {
    let frames = rounds.get(index);
    if (frames) {
      rounds.delete(index);
    } else {
      frames = materialize(model.rounds[index]!, names);
      if (rounds.size === CACHED_ROUNDS)
        rounds.delete(rounds.keys().next().value!);
    }
    rounds.set(index, frames);
    return frames;
  }

  return {
    readSnapshot: (round, step) => ({
      ...step,
      ...tables(round)[step.frame]!,
    }),
  };
}

function materialize(
  round: ReplayRound,
  names: readonly string[],
): TableSnapshot[] {
  let table: TableSnapshot = {
    ...round.start,
    players: round.start.players.map((player, seat) => ({
      ...player,
      seat,
      name: names[seat]!,
    })),
  };
  const tables = [table];
  for (const delta of round.deltas) {
    table = applyDelta(table, delta);
    tables.push(table);
  }
  return tables;
}

function applyDelta(state: TableSnapshot, delta: StateDelta): TableSnapshot {
  let players = state.players;
  if (delta.players) {
    const changed = players.slice();
    for (const change of delta.players) {
      const player = changed[change.seat]!;
      changed[change.seat] = {
        seat: player.seat,
        name: player.name,
        score: change.score ?? player.score,
        riichi: change.riichi ?? player.riichi,
        hand: applySplice(player.hand, change.hand),
        draw: change.draw === undefined ? player.draw : change.draw,
        river: applySplice(player.river, change.river),
        melds: applySplice(player.melds, change.melds),
      } satisfies PlayerSnapshot;
    }
    players = changed;
  }
  return {
    players,
    dora: applySplice(state.dora, delta.dora),
    honba: delta.honba ?? state.honba,
    kyotaku: delta.kyotaku ?? state.kyotaku,
    remaining: delta.remaining ?? state.remaining,
    activeSeat: delta.activeSeat ?? state.activeSeat,
  };
}

/** 差し替えを適用した配列を返す。差し替えが無ければ、元の配列をそのまま返す。 */
function applySplice<T>(
  values: readonly T[],
  splice: Splice<T> | undefined,
): readonly T[] {
  if (splice === undefined) return values;
  const [index, removeCount] = splice;
  const result = values.slice(0, index);
  for (let position = 2; position < splice.length; position++)
    result.push(splice[position] as T);
  for (let position = index + removeCount; position < values.length; position++)
    result.push(values[position]!);
  return result;
}

/**
 * 対局全体で最も広い席の列の幅（副露だけで測る）に、余白 0.4 を足して返す。両方のカメラは、対局の間ずっとこの幅で固定する。
 */
export function maximumPlayerRowWidth(
  model: ReplayModel,
  measure: (melds: readonly Meld[]) => number,
): number {
  let width = measure([]);
  for (const round of model.rounds) {
    const melds = round.start.players.map((player) => player.melds);
    for (const row of melds) width = Math.max(width, measure(row));
    for (const delta of round.deltas)
      for (const change of delta.players ?? [])
        if (change.melds !== undefined) {
          const row = applySplice(melds[change.seat]!, change.melds);
          melds[change.seat] = row;
          width = Math.max(width, measure(row));
        }
  }
  return width + 0.4;
}
