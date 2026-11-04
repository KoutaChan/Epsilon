import type {
  DoraEdit,
  HandEdit,
  RiverEdit,
  ScenarioEdits,
} from "../../../../../protocol/model.ts";
import {
  tileOrder,
  type RiverTile,
  type TableState,
} from "../../../../../protocol/result.ts";
import type {
  Candidate,
  DecisionStep,
  PlayerSnapshot,
  Snapshot,
} from "../model/types.ts";

// もしもの変更は判断の局面の上の差分として持ち、卓の写しもサーバへの要求もここから作る。

/**
 * 変えられる所。席は絶対の席、index は river[]・dora[] の位置。手牌の index は判断の局面の hand[] の位置で、
 * hand.length はツモ牌。
 */
export type ScenarioTarget =
  | { kind: "river"; seat: number; index: number }
  | { kind: "dora"; index: number }
  | { kind: "hand"; index: number }
  | { kind: "score"; seat: number }
  | { kind: "honba" }
  | { kind: "kyotaku" };
/** 元の局面と違う所。並びは河・ドラ・手牌・点数（席の順）・本場・供託。 */
export type ScenarioChange =
  | {
      kind: "river";
      seat: number;
      index: number;
      from: RiverTile;
      to: RiverEdit;
    }
  | { kind: "dora"; index: number; from: string; to: string }
  | { kind: "hand"; index: number; from: string; to: string }
  | { kind: "score"; seat: number; from: number; to: number }
  | { kind: "honba"; from: number; to: number }
  | { kind: "kyotaku"; from: number; to: number };
/** 牌の一覧で選び直す所。 */
export type TileTarget = Extract<
  ScenarioTarget,
  { kind: "river" | "dora" | "hand" }
>;

/**
 * 変更を当てた局面。手牌は tileOrder の順に並べ直し、handOrigins が表示の位置ごとの判断の局面の hand[] の
 * 位置を持つ（ツモ牌はツモ牌のまま）。
 */
export interface EditedBoard {
  readonly snapshot: Snapshot;
  readonly handOrigins: readonly number[];
}

export const NO_EDITS: ScenarioEdits = {};
export const SCORE_STEP = 100;
const SCORE_LIMIT = 200_000;
export const STICK_LIMIT = 20;

/** 牌の一覧の行（34 種と赤 5）。 */
const TILE_ROWS: readonly (readonly string[])[] = [
  ["1m", "2m", "3m", "4m", "5m", "5mr", "6m", "7m", "8m", "9m"],
  ["1p", "2p", "3p", "4p", "5p", "5pr", "6p", "7p", "8p", "9p"],
  ["1s", "2s", "3s", "4s", "5s", "5sr", "6s", "7s", "8s", "9s"],
  ["E", "S", "W", "N", "P", "F", "C"],
];

const isRed = (tile: string) => tile.endsWith("r");
/** 赤を区別しない牌の種類。 */
const tileKind = (tile: string) => (isRed(tile) ? tile.slice(0, 2) : tile);

export const targetKey = (target: ScenarioTarget) =>
  target.kind === "river"
    ? `river:${target.seat}:${target.index}`
    : target.kind === "dora" || target.kind === "hand"
      ? `${target.kind}:${target.index}`
      : target.kind === "score"
        ? `score:${target.seat}`
        : target.kind;

/** 表示の手牌の位置の、判断の局面の hand[] の位置。ツモ牌の位置はそのまま。 */
export const handOrigin = (origins: readonly number[], position: number) =>
  origins[position] ?? position;

/** 判断の局面の手牌の位置の牌（hand.length はツモ牌）。 */
function handTile(player: PlayerSnapshot, index: number): string {
  return index === player.hand.length ? player.draw! : player.hand[index]!;
}

/** 変更の並びと省き方を決めた形にする。同じ局面は同じ JSON になる。手牌の変更は元と違うものだけを渡す。 */
function normalize(
  original: Snapshot,
  rivers: readonly RiverEdit[],
  dora: readonly DoraEdit[],
  hand: readonly HandEdit[],
  scores: readonly number[] | undefined,
  honba: number | undefined,
  kyotaku: number | undefined,
): ScenarioEdits {
  const edits: ScenarioEdits = {};
  const changedRivers = rivers
    .filter((edit) => {
      const tile = original.players[edit.seat]!.river[edit.index]!;
      return (
        tile.tile !== edit.tile || (tile.tsumogiri === true) !== edit.tsumogiri
      );
    })
    .sort((a, b) => a.seat - b.seat || a.index - b.index);
  if (changedRivers.length > 0) edits.rivers = changedRivers;
  const changedDora = dora
    .filter((edit) => original.dora[edit.index] !== edit.tile)
    .sort((a, b) => a.index - b.index);
  if (changedDora.length > 0) edits.dora = changedDora;
  if (hand.length > 0) edits.hand = [...hand].sort((a, b) => a.index - b.index);
  if (scores?.some((score, seat) => score !== original.players[seat]!.score))
    edits.scores = scores;
  if (honba !== undefined && honba !== original.honba) edits.honba = honba;
  if (kyotaku !== undefined && kyotaku !== original.kyotaku)
    edits.kyotaku = kyotaku;
  return Object.keys(edits).length > 0 ? edits : NO_EDITS;
}

export function setRiver(
  edits: ScenarioEdits,
  original: Snapshot,
  seat: number,
  index: number,
  change: Partial<Pick<RiverEdit, "tile" | "tsumogiri">>,
): ScenarioEdits {
  const rivers = edits.rivers ?? [];
  const tile = original.players[seat]!.river[index]!;
  const current = rivers.find(
    (edit) => edit.seat === seat && edit.index === index,
  ) ?? { seat, index, tile: tile.tile, tsumogiri: tile.tsumogiri === true };
  return normalize(
    original,
    [...rivers.filter((edit) => edit !== current), { ...current, ...change }],
    edits.dora ?? [],
    edits.hand ?? [],
    edits.scores,
    edits.honba,
    edits.kyotaku,
  );
}

export function setDora(
  edits: ScenarioEdits,
  original: Snapshot,
  index: number,
  tile: string,
): ScenarioEdits {
  return normalize(
    original,
    edits.rivers ?? [],
    [
      ...(edits.dora ?? []).filter((edit) => edit.index !== index),
      { index, tile },
    ],
    edits.hand ?? [],
    edits.scores,
    edits.honba,
    edits.kyotaku,
  );
}

/** 判断の席の手牌の 1 枚を入れ替える。index は判断の局面の hand[] の位置（hand.length はツモ牌）。 */
export function setHand(
  edits: ScenarioEdits,
  original: Snapshot,
  seat: number,
  index: number,
  tile: string,
): ScenarioEdits {
  const hand = (edits.hand ?? []).filter((edit) => edit.index !== index);
  if (tile !== handTile(original.players[seat]!, index))
    hand.push({ index, tile });
  return normalize(
    original,
    edits.rivers ?? [],
    edits.dora ?? [],
    hand,
    edits.scores,
    edits.honba,
    edits.kyotaku,
  );
}

export function setScore(
  edits: ScenarioEdits,
  original: Snapshot,
  seat: number,
  score: number,
): ScenarioEdits {
  const scores = (
    edits.scores ?? original.players.map((player) => player.score)
  ).slice();
  scores[seat] = Math.max(-SCORE_LIMIT, Math.min(SCORE_LIMIT, score));
  return normalize(
    original,
    edits.rivers ?? [],
    edits.dora ?? [],
    edits.hand ?? [],
    scores,
    edits.honba,
    edits.kyotaku,
  );
}

export function setSticks(
  edits: ScenarioEdits,
  original: Snapshot,
  kind: "honba" | "kyotaku",
  count: number,
): ScenarioEdits {
  const value = Math.max(0, Math.min(STICK_LIMIT, count));
  return normalize(
    original,
    edits.rivers ?? [],
    edits.dora ?? [],
    edits.hand ?? [],
    edits.scores,
    kind === "honba" ? value : edits.honba,
    kind === "kyotaku" ? value : edits.kyotaku,
  );
}

export function revert(
  edits: ScenarioEdits,
  original: Snapshot,
  target: ScenarioTarget,
): ScenarioEdits {
  switch (target.kind) {
    case "river": {
      const tile = original.players[target.seat]!.river[target.index]!;
      return setRiver(edits, original, target.seat, target.index, {
        tile: tile.tile,
        tsumogiri: tile.tsumogiri === true,
      });
    }
    case "dora":
      return setDora(
        edits,
        original,
        target.index,
        original.dora[target.index]!,
      );
    case "hand":
      return normalize(
        original,
        edits.rivers ?? [],
        edits.dora ?? [],
        (edits.hand ?? []).filter((edit) => edit.index !== target.index),
        edits.scores,
        edits.honba,
        edits.kyotaku,
      );
    case "score":
      return setScore(
        edits,
        original,
        target.seat,
        original.players[target.seat]!.score,
      );
    case "honba":
    case "kyotaku":
      return setSticks(edits, original, target.kind, original[target.kind]);
  }
}

function editedDiscard(base: RiverTile, edit: RiverEdit): RiverTile {
  const discard: RiverTile = { tile: edit.tile };
  if (edit.tsumogiri) discard.tsumogiri = true;
  if (base.riichi) discard.riichi = true;
  if (base.called) discard.called = true;
  return discard;
}

const sequence = (length: number) =>
  Array.from({ length }, (_, index) => index);

/** 変更を当てた局面の写しを作る。読み込みの局面は共有なので書き換えず、変わらない席と配列はそのまま使う。 */
export function applyEdits(
  snapshot: Snapshot,
  edits: ScenarioEdits,
  viewSeat: number,
): EditedBoard {
  const viewer = snapshot.players[viewSeat]!;
  let handOrigins: readonly number[] = sequence(viewer.hand.length);
  if (edits === NO_EDITS) return { snapshot, handOrigins };
  const players = snapshot.players.map((player) => {
    const rivers =
      edits.rivers?.filter((edit) => edit.seat === player.seat) ?? [];
    const score = edits.scores?.[player.seat] ?? player.score;
    const hand = player === viewer ? (edits.hand ?? []) : [];
    if (rivers.length === 0 && score === player.score && hand.length === 0)
      return player;
    let river = player.river;
    if (rivers.length > 0) {
      const discards = river.slice();
      for (const edit of rivers)
        discards[edit.index] = editedDiscard(discards[edit.index]!, edit);
      river = discards;
    }
    let tiles = player.hand;
    let draw = player.draw;
    if (hand.length > 0) {
      const replaced = player.hand.slice();
      for (const edit of hand)
        if (edit.index === replaced.length) draw = edit.tile;
        else replaced[edit.index] = edit.tile;
      // 並べ替えは安定なので、同じ牌は元の位置の順のまま。
      const origins = sequence(replaced.length).sort(
        (a, b) => tileOrder(replaced[a]!) - tileOrder(replaced[b]!),
      );
      tiles = origins.map((index) => replaced[index]!);
      handOrigins = origins;
    }
    return { ...player, score, river, hand: tiles, draw };
  });
  let dora = snapshot.dora;
  if (edits.dora) {
    const indicators = dora.slice();
    for (const edit of edits.dora) indicators[edit.index] = edit.tile;
    dora = indicators;
  }
  return {
    snapshot: {
      ...snapshot,
      players,
      dora,
      honba: edits.honba ?? snapshot.honba,
      kyotaku: edits.kyotaku ?? snapshot.kyotaku,
    },
    handOrigins,
  };
}

/** サーバが返した局面（鳴いた後など）を、判断の局面の名前と手番の情報を持つ卓の局面にする。 */
export function tableSnapshot(base: Snapshot, table: TableState): Snapshot {
  return {
    ...base,
    ...table,
    players: table.players.map((player, seat) => ({
      ...player,
      seat,
      name: base.players[seat]!.name,
    })),
  };
}

/** 鳴いた後の打牌を見られる候補（チー・ポン）。大明槓は嶺上のツモが分からないので含めない。 */
export const isCall = (candidate: Candidate) =>
  candidate.type === "CHI" || candidate.type === "PON";

export function listChanges(
  edits: ScenarioEdits,
  original: Snapshot,
  viewSeat: number,
): readonly ScenarioChange[] {
  const changes: ScenarioChange[] = [];
  for (const edit of edits.rivers ?? [])
    changes.push({
      kind: "river",
      seat: edit.seat,
      index: edit.index,
      from: original.players[edit.seat]!.river[edit.index]!,
      to: edit,
    });
  for (const edit of edits.dora ?? [])
    changes.push({
      kind: "dora",
      index: edit.index,
      from: original.dora[edit.index]!,
      to: edit.tile,
    });
  for (const edit of edits.hand ?? [])
    changes.push({
      kind: "hand",
      index: edit.index,
      from: handTile(original.players[viewSeat]!, edit.index),
      to: edit.tile,
    });
  edits.scores?.forEach((score, seat) => {
    const from = original.players[seat]!.score;
    if (score !== from) changes.push({ kind: "score", seat, from, to: score });
  });
  if (edits.honba !== undefined)
    changes.push({ kind: "honba", from: original.honba, to: edits.honba });
  if (edits.kyotaku !== undefined)
    changes.push({
      kind: "kyotaku",
      from: original.kyotaku,
      to: edits.kyotaku,
    });
  return changes;
}

/** 河の牌を変えられるか。鳴かれた牌は、鳴いた席の副露と食い違うので変えない。 */
export function canEditRiver(
  snapshot: Snapshot,
  seat: number,
  index: number,
): boolean {
  const tile = snapshot.players[seat]!.river[index];
  return tile !== undefined && tile.called !== true;
}

/** 見ている席の手牌を変えられるか。その席の判断があり、判断の局面でリーチしていない時だけ。 */
export function canEditHand(
  snapshot: Snapshot,
  viewSeat: number,
  decision: DecisionStep | null,
): boolean {
  return decision !== null && !snapshot.players[viewSeat]!.riichi;
}

/**
 * 見ている席から見えている牌を数える（自分の手牌とツモ、全員の河と副露、ドラ表示牌）。
 * サーバの検証と同じ範囲。変える所の牌は数えない。
 */
function visibleTiles(
  board: EditedBoard,
  viewSeat: number,
  skip: ScenarioTarget,
) {
  const counts = new Map<string, number>();
  const reds = new Set<string>();
  const add = (tile: string) => {
    const kind = tileKind(tile);
    counts.set(kind, (counts.get(kind) ?? 0) + 1);
    if (isRed(tile)) reds.add(kind);
  };
  const { snapshot, handOrigins } = board;
  const viewer = snapshot.players[viewSeat]!;
  const skipped = skip.kind === "hand" ? skip.index : -1;
  viewer.hand.forEach((tile, position) => {
    if (handOrigin(handOrigins, position) !== skipped) add(tile);
  });
  if (viewer.draw && skipped !== viewer.hand.length) add(viewer.draw);
  for (const player of snapshot.players) {
    player.river.forEach((tile, index) => {
      // 鳴かれた牌は副露の側で数える。
      const river =
        skip.kind === "river" &&
        skip.seat === player.seat &&
        skip.index === index;
      if (!tile.called && !river) add(tile.tile);
    });
    for (const meld of player.melds) meld.tiles.forEach(add);
  }
  snapshot.dora.forEach((tile, index) => {
    if (!(skip.kind === "dora" && skip.index === index)) add(tile);
  });
  return { counts, reds };
}

/** 変える所の今の牌。 */
function currentTile(board: EditedBoard, viewSeat: number, target: TileTarget) {
  const { snapshot, handOrigins } = board;
  if (target.kind === "river")
    return snapshot.players[target.seat]!.river[target.index]!.tile;
  if (target.kind === "dora") return snapshot.dora[target.index]!;
  const viewer = snapshot.players[viewSeat]!;
  return target.index === viewer.hand.length
    ? viewer.draw!
    : viewer.hand[handOrigins.indexOf(target.index)]!;
}

export interface TileChoice {
  readonly tile: string;
  /** 変える所の牌を除いて、見えている同じ種類の枚数。 */
  readonly visible: number;
  readonly current: boolean;
  readonly disabled: boolean;
}
/** 牌の一覧。1 種 4 枚・赤 5 は 1 枚を超える牌は選べない。 */
export function tileChoices(
  board: EditedBoard,
  viewSeat: number,
  target: TileTarget,
): readonly (readonly TileChoice[])[] {
  const current = currentTile(board, viewSeat, target);
  const { counts, reds } = visibleTiles(board, viewSeat, target);
  return TILE_ROWS.map((row) =>
    row.map((tile) => {
      const kind = tileKind(tile);
      const visible = counts.get(kind) ?? 0;
      const full = visible >= 4 || (isRed(tile) && reds.has(kind));
      return {
        tile,
        visible,
        current: tile === current,
        disabled: tile !== current && full,
      };
    }),
  );
}

const candidateKey = (candidate: Candidate) =>
  `${candidate.type}|${candidate.tiles.join(",")}|${candidate.tsumogiri}`;
/** もしもの候補ごとの元の候補と順位（1 から）。元にない候補は null。 */
export type CandidateComparison = ReadonlyMap<
  Candidate,
  { previous: Candidate; rank: number } | null
>;
/** もしもの候補を、同じ行動（種類・牌・ツモ切り）の元の候補と並べる。 */
export function compareCandidates(
  original: readonly Candidate[],
  edited: readonly Candidate[],
): CandidateComparison {
  const ranks = new Map(
    original.map((candidate, index) => [candidateKey(candidate), index]),
  );
  return new Map(
    edited.map((candidate) => {
      const index = ranks.get(candidateKey(candidate));
      return [
        candidate,
        index === undefined
          ? null
          : { previous: original[index]!, rank: index + 1 },
      ] as const;
    }),
  );
}
