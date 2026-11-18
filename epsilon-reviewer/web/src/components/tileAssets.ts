import type { Meld } from "@epsilon-reviewer/protocol/result";

// 牌の絵は public/tiles/Regular/<名前>.svg（FluffyStuff）を使う。DOM の牌も卓の牌も、絵の名前はここで決める。

export const BACK = "back";

const HONORS: Readonly<Record<string, string>> = {
  E: "Ton",
  S: "Nan",
  W: "Shaa",
  N: "Pei",
  P: "Haku",
  F: "Hatsu",
  C: "Chun",
};
const SUITS: Readonly<Record<string, string>> = {
  m: "Man",
  p: "Pin",
  s: "Sou",
};

/** 牌の記号から絵の名前を返す。赤 5 の名前は <種類>5-Dora。 */
export function tileAssetName(tile: string): string {
  return (
    HONORS[tile] ??
    `${SUITS[tile[1]!]}${tile.length === 3 ? "5-Dora" : tile[0]}`
  );
}

export function tileAsset(tile: string): string {
  return `./tiles/Regular/${tileAssetName(tile)}.svg`;
}

/** 暗槓は両端を伏せ、赤があれば内側に見せる。 */
export function meldFace(meld: Meld, index: number): string {
  const tile = meld.tiles[index]!;
  if (meld.type !== "ANKAN") return tile;
  if (index === 0 || index === meld.tiles.length - 1) return BACK;
  if (index === 1) return meld.tiles.find((face) => face.endsWith("r")) ?? tile;
  return tile.replace("r", "");
}
