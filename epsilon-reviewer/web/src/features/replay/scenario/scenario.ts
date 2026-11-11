import type {
  DoraEdit,
  HandEdit,
  RiverEdit,
  ScenarioEdits,
} from "../../../../../protocol/model.ts";
import {
  tileCopies,
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

// もしもの変更は、判断の局面からの差分として持つ。卓に描く写しも、サーバへの要求もこの差分から作る。

/**
 * 変えられる所。席は絶対の席。`index` は河・ドラ・手牌の中の位置で、0 から数える。手牌の位置は判断の局面でのもので、
 * 手牌の枚数（`hand.length`）ならツモ牌。
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
/** 牌の一覧から牌を選び直す所。 */
export type TileTarget = Extract<
  ScenarioTarget,
  { kind: "river" | "dora" | "hand" }
>;

/**
 * 変更を反映した局面。手牌は牌の並び順（`tileOrder`）に並べ直す。`handOrigins` は表示の並びの位置ごとに、
 * その牌が判断の局面で手牌のどこにあったかを持つ（ツモ牌はツモ牌のまま）。
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

export const targetKey = (target: ScenarioTarget) =>
  target.kind === "river"
    ? `river:${target.seat}:${target.index}`
    : target.kind === "dora" || target.kind === "hand"
      ? `${target.kind}:${target.index}`
      : target.kind === "score"
        ? `score:${target.seat}`
        : target.kind;

/** 表示の並びで指定した位置にある手牌が、判断の局面で手牌のどこにあったかを返す。ツモ牌の位置はそのまま返す。 */
export const handOrigin = (origins: readonly number[], position: number) =>
  origins[position] ?? position;

/** 判断の局面で、手牌の指定した位置にある牌を返す。位置が手牌の枚数ならツモ牌を返す。 */
function handTile(player: PlayerSnapshot, index: number): string {
  return index === player.hand.length ? player.draw! : player.hand[index]!;
}

/** 変更を決まった順に並べ、元と同じ所を省く。同じ局面なら同じ JSON になる。手牌の変更は、元と違うものだけを渡すこと。 */
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

/** 判断する席の手牌を 1 枚入れ替える。手牌の位置は判断の局面でのもので、手牌の枚数ならツモ牌。 */
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

/** 変更を反映した局面の写しを作る。読み込んだ局面はほかでも使っているので書き換えない。変わらない席と配列は、元のものをそのまま使う。 */
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

/** サーバが返した局面（鳴いた後など）を、卓に描く局面にする。返った局面に無い情報と対局者名は判断の局面から引き継ぐ。 */
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

/** 鳴いた後の打牌を見られる候補か（チー・ポン）。大明槓は嶺上のツモが分からないので含めない。 */
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

/**
 * 見ている席の手牌の、表示の並びで index の位置（hand.length ならツモ牌）を変えられるか。その席が判断する局面で、
 * まだリーチしておらず、その位置に牌があるときだけ変えられる。
 */
export function canEditHand(
  snapshot: Snapshot,
  viewSeat: number,
  decision: DecisionStep | null,
  index: number,
): boolean {
  const viewer = snapshot.players[viewSeat]!;
  return (
    decision !== null &&
    !viewer.riichi &&
    (index === viewer.hand.length
      ? viewer.draw !== null
      : index < viewer.hand.length)
  );
}

/**
 * 見ている席から見える牌を、牌の記号ごとに数える（自分の手牌とツモ牌、全員の河と副露、ドラ表示牌）。
 * 数える範囲はサーバの検証と同じ。変える所にある牌は数えない。
 */
function visibleTiles(
  board: EditedBoard,
  viewSeat: number,
  skip: ScenarioTarget,
): ReadonlyMap<string, number> {
  const counts = new Map<string, number>();
  const add = (tile: string) => counts.set(tile, (counts.get(tile) ?? 0) + 1);
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
  return counts;
}

/** 変える所に今ある牌を返す。 */
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

/** 変える所に判断の局面であった牌を返す。 */
function originalTile(
  original: Snapshot,
  viewSeat: number,
  target: TileTarget,
) {
  if (target.kind === "river")
    return original.players[target.seat]!.river[target.index]!.tile;
  if (target.kind === "dora") return original.dora[target.index]!;
  return handTile(original.players[viewSeat]!, target.index);
}

/**
 * 変える所にその牌を置けるか。今の牌は置ける。ほかの牌は、変える所を除いて見えている枚数が物理牌の数より
 * 少ないときだけ置ける（サーバが断る牌と同じ決まり）。
 */
function fits(
  tile: string,
  current: string,
  counts: ReadonlyMap<string, number>,
  seats: number,
) {
  return (
    tile === current || (counts.get(tile) ?? 0) < tileCopies(tile, seats)
  );
}

export interface TileChoice {
  readonly tile: string;
  /** 変える所の牌を除いて、見えている同じ記号の牌の枚数。 */
  readonly visible: number;
  readonly current: boolean;
  readonly disabled: boolean;
}
/**
 * 変える所に置く牌の一覧。見えている枚数が物理牌の数に達した牌は選べない（今の牌は選び直せる）。
 * 卓に無い牌（三人打ちの 2〜8 萬と赤の 5 萬）は出さない。
 */
export function tileChoices(
  board: EditedBoard,
  viewSeat: number,
  target: TileTarget,
): readonly (readonly TileChoice[])[] {
  const seats = board.snapshot.players.length;
  const current = currentTile(board, viewSeat, target);
  const counts = visibleTiles(board, viewSeat, target);
  return TILE_ROWS.map((row) =>
    row
      .filter((tile) => tileCopies(tile, seats) > 0)
      .map((tile) => ({
        tile,
        visible: counts.get(tile) ?? 0,
        current: tile === current,
        disabled: !fits(tile, current, counts, seats),
      })),
  );
}

/** 変えた所を元に戻せるか。牌は、元の牌を牌の一覧で選べるときだけ戻せる。 */
export function canRevert(
  board: EditedBoard,
  original: Snapshot,
  viewSeat: number,
  target: ScenarioTarget,
): boolean {
  switch (target.kind) {
    case "river":
    case "dora":
    case "hand":
      return fits(
        originalTile(original, viewSeat, target),
        currentTile(board, viewSeat, target),
        visibleTiles(board, viewSeat, target),
        board.snapshot.players.length,
      );
    case "score":
    case "honba":
    case "kyotaku":
      return true;
  }
}

const candidateKey = (candidate: Candidate) =>
  `${candidate.type}|${candidate.tiles.join(",")}|${candidate.tsumogiri}`;
/** もしもの候補ごとに、対応する元の候補とその順位（1 から）。元に無い候補は null。 */
export type CandidateComparison = ReadonlyMap<
  Candidate,
  { previous: Candidate; rank: number } | null
>;
/** もしもの候補を、同じ行動（種類・牌・ツモ切り）の元の候補と対応させる。 */
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
