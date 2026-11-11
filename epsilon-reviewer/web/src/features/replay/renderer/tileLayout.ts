import type { Meld } from "../../../../../protocol/result.ts";

export const PLAYER_ROW_WIDTH = 23;
export const PLAYER_ROW_CENTER = 0.24;
const HAND_PITCH = 1.012;
const DRAW_GAP = 0.48;
const HAND_MELD_GAP = 0.6;
const TILE_GAP = 0.025;
const MELD_GAP = 0.45;
const ADDED_KAN_OFFSET = 1 + TILE_GAP;

export interface MeldPlacement {
  readonly meld: Meld;
  readonly index: number;
  readonly x: number;
  /** 卓の中央の側へのずれ。ずれるのは加槓で足した牌だけで、ほかの牌は 0。 */
  readonly y: number;
  readonly added: boolean;
  readonly sideways: boolean;
}

export interface PlayerRow {
  readonly handLeft: number;
  readonly melds: readonly MeldPlacement[];
  readonly width: number;
}

export function concealedTileX(
  index: number,
  drawn: boolean,
  handLeft: number,
): number {
  return handLeft + index * HAND_PITCH + (drawn ? DRAW_GAP : 0);
}

function meldTileWidth(meld: Meld, index: number): number {
  if (meld.type === "KAKAN" && index === meld.addedIndex) return 0;
  return index === meld.calledIndex ? 4 / 3 : 1;
}

function meldWidth(meld: Meld): number {
  const baseCount = meld.tiles.length - (meld.type === "KAKAN" ? 1 : 0);
  let width = (baseCount - 1) * TILE_GAP;
  for (let index = 0; index < meld.tiles.length; index++)
    width += meldTileWidth(meld, index);
  return width;
}

/** 副露のうち、手牌を 3 枚減らすものの数（抜いた北は数えない）。 */
function sets(melds: readonly Meld[]): number {
  return melds.filter((meld) => meld.type !== "NUKI").length;
}

export function measurePlayerRowWidth(melds: readonly Meld[]): number {
  return (
    (13 - 3 * sets(melds)) * HAND_PITCH +
    1 +
    DRAW_GAP +
    (melds.length ? HAND_MELD_GAP : 0) +
    melds.reduce((sum, meld) => sum + meldWidth(meld), 0) +
    Math.max(0, melds.length - 1) * MELD_GAP
  );
}

// 最初の副露は対局を通して同じ右端に置き、後の副露はその左へ足していく。副露の中の牌の順は変えない。
export function layoutPlayerRow(
  melds: readonly Meld[],
  rowWidth = PLAYER_ROW_WIDTH,
): PlayerRow {
  const width = measurePlayerRowWidth(melds);
  const handWidth = (13 - 3 * sets(melds)) * HAND_PITCH + 1 + DRAW_GAP;
  const right = PLAYER_ROW_CENTER + rowWidth / 2;
  // 手牌はツモ枠を含めて中央に置く。副露に重なるときだけ、重なる分を左へずらす。
  const left = Math.min(PLAYER_ROW_CENTER - handWidth / 2, right - width);
  const placements: MeldPlacement[] = [];
  let end = right;
  for (const meld of melds) {
    const start = end - meldWidth(meld);
    let offset = 0,
      calledX = 0;
    for (let index = 0; index < meld.tiles.length; index++) {
      const added = meld.type === "KAKAN" && index === meld.addedIndex;
      const tileWidth = meldTileWidth(meld, index);
      const x = added ? calledX : start + offset + tileWidth / 2;
      if (index === meld.calledIndex) calledX = x;
      // 加槓で足した牌は鳴いた横向きの牌の奥に置き、両方の牌面が見えるようにする。
      placements.push({
        meld,
        index,
        x,
        y: added ? ADDED_KAN_OFFSET : 0,
        added,
        sideways: index === meld.calledIndex || added,
      });
      if (!added) offset += tileWidth + TILE_GAP;
    }
    end = start - MELD_GAP;
  }
  return { handLeft: left + 0.5, melds: placements, width };
}
