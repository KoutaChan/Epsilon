// 牌の絵は public/tiles/Regular/<名前>.svg（FluffyStuff）。DOM の牌と卓の牌の絵の両方がここで名前を引く。

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

/** 絵のある 37 の牌の記号（字牌、萬・筒・索の 1..9 と赤の 5）。 */
export const ASSET_TILES: readonly string[] = [
  ...Object.keys(HONORS),
  ...Object.keys(SUITS).flatMap((suit) => [
    ...Array.from({ length: 9 }, (_, index) => `${index + 1}${suit}`),
    `5${suit}r`,
  ]),
];

/** 牌の記号の絵の名前。赤の 5 は <種類>5-Dora。 */
export function tileAssetName(tile: string): string {
  return (
    HONORS[tile] ??
    `${SUITS[tile[1]!]}${tile.length === 3 ? "5-Dora" : tile[0]}`
  );
}
