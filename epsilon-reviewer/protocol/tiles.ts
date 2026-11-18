export const HONOR_CODES = ["E", "S", "W", "N", "P", "F", "C"] as const;
export const SUITS = ["m", "p", "s"] as const;
export const RED_FIVES = SUITS.map((suit) => `5${suit}r`);
export const TILE_ROWS: readonly (readonly string[])[] = [
  ...SUITS.map((suit) =>
    Array.from({ length: 9 }, (_, i) =>
      i === 4 ? [`5${suit}`, `5${suit}r`] : [`${i + 1}${suit}`],
    ).flat(),
  ),
  HONOR_CODES,
];
export const TILE_CODES = TILE_ROWS.flat();
const codes = new Set(TILE_CODES);

/** 牌種の番号（0..33。萬・筒・索の 1..9、東南西北白發中の順）を牌の記号にする。 */
export function tileTypeCode(type: number): string {
  if (type >= 27) return HONOR_CODES[type - 27]!;
  return `${(type % 9) + 1}${SUITS[Math.floor(type / 9)]}`;
}

/** 牌の記号かを返す（萬筒索の 1..9、赤は 5 にだけ r を付ける、字牌は ESWNPFC）。 */
export function isTileCode(value: unknown): value is string {
  return typeof value === "string" && codes.has(value);
}

/** 手牌を並べるときの順の値。萬・筒・索の 1..9、東南西北白發中の順で、赤の 5 は同じ色の 5 の前に来る。 */
export function tileOrder(tile: string): number {
  const red = tile.length === 3;
  const honor = HONOR_CODES.indexOf(tile as (typeof HONOR_CODES)[number]);
  const type =
    honor >= 0
      ? 27 + honor
      : SUITS.indexOf(tile[1] as (typeof SUITS)[number]) * 9 +
        Number(tile[0]) -
        1;
  return type * 2 + (red ? 0 : 1);
}

/**
 * その記号の牌が山に何枚あるかを返す（Java の TileCopies.copies と同じ）。各色の 5 は 4 枚のうち 1 枚が赤なので、
 * 赤の 5 は 1 枚、赤でない 5 は 3 枚、ほかは 4 枚。三人打ち（seats が 3）には 2〜8 萬と赤の 5 萬が無い。
 */
export function tileCopies(tile: string, seats = 4): number {
  if (seats === 3 && tile[1] === "m" && tile[0] !== "1" && tile[0] !== "9")
    return 0;
  if (tile.length === 3) return 1;
  return tile[0] === "5" ? 3 : 4;
}
