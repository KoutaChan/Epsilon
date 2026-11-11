// 牌の絵は public/tiles/Regular/<名前>.svg（FluffyStuff）を使う。DOM の牌も卓の牌も、絵の名前はここで決める。

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

/** 絵のある 37 種の牌の記号（字牌と、萬・筒・索の 1〜9 と赤 5）。 */
export const ASSET_TILES: readonly string[] = [
  ...Object.keys(HONORS),
  ...Object.keys(SUITS).flatMap((suit) => [
    ...Array.from({ length: 9 }, (_, index) => `${index + 1}${suit}`),
    `5${suit}r`,
  ]),
];

/** 牌の記号から絵の名前を返す。赤 5 の名前は <種類>5-Dora。 */
export function tileAssetName(tile: string): string {
  return (
    HONORS[tile] ??
    `${SUITS[tile[1]!]}${tile.length === 3 ? "5-Dora" : tile[0]}`
  );
}
