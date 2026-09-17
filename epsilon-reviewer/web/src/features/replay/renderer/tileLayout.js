export const PLAYER_ROW_WIDTH = 23;
export const PLAYER_ROW_CENTER = 0.24;
const HAND_PITCH = 1.012;
const DRAW_GAP = 0.48;
const HAND_MELD_GAP = 0.6;
const TILE_GAP = 0.025;
const MELD_GAP = 0.45;
const ADDED_KAN_OFFSET = 1 + TILE_GAP;

export function concealedTileX(index, drawn, handLeft) {
  return handLeft + index * HAND_PITCH + (drawn ? DRAW_GAP : 0);
}

function meldTileWidth(meld, index) {
  if (meld.type === "KAKAN" && index === meld.addedIndex) return 0;
  return index === meld.calledIndex ? 4 / 3 : 1;
}

function meldWidth(meld) {
  const baseCount = meld.tiles.length - (meld.type === "KAKAN" ? 1 : 0);
  let width = (baseCount - 1) * TILE_GAP;
  for (let index = 0; index < meld.tiles.length; index++)
    width += meldTileWidth(meld, index);
  return width;
}

export function measurePlayerRowWidth(melds) {
  return (
    (13 - 3 * melds.length) * HAND_PITCH +
    1 +
    DRAW_GAP +
    (melds.length ? HAND_MELD_GAP : 0) +
    melds.reduce((sum, meld) => sum + meldWidth(meld), 0) +
    Math.max(0, melds.length - 1) * MELD_GAP
  );
}

// 対局全体の右端へ最初の副露を固定し、新しい組を左へ足す。組の中の牌順は保つ。
export function layoutPlayerRow(melds, rowWidth = PLAYER_ROW_WIDTH) {
  const width = measurePlayerRowWidth(melds);
  const handWidth = (13 - 3 * melds.length) * HAND_PITCH + 1 + DRAW_GAP;
  const right = PLAYER_ROW_CENTER + rowWidth / 2;
  // 手牌はツモ枠込みで中央に置き、副露に重なる場合だけ必要な分を左へ寄せる。
  const left = Math.min(PLAYER_ROW_CENTER - handWidth / 2, right - width);
  const placements = [];
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
      // 加槓牌は元の横向き牌の奥へ並べ、両方の牌面を見せる。
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
