import { tileAssetName } from "../features/replay/tileAssets.ts";
import type { MessageKey, Translate } from "../language/catalog";
import { useLanguage } from "../language/LanguageProvider";

/** 伏せた牌（文書の牌の記号ではない）。 */
const BACK = "back";
const HONOR_NAMES: Readonly<Record<string, MessageKey>> = {
  E: "tile.E",
  S: "tile.S",
  W: "tile.W",
  N: "tile.N",
  P: "tile.P",
  F: "tile.F",
  C: "tile.C",
};
const SUIT_NAMES: Readonly<Record<string, MessageKey>> = {
  m: "tile.m",
  p: "tile.p",
  s: "tile.s",
};

export function tileName(tile: string, t: Translate): string {
  if (tile === BACK) return t("tile.back");
  const honor = HONOR_NAMES[tile];
  if (honor) return t(honor);
  const red = tile.length === 3;
  const name = t(SUIT_NAMES[tile[1]!]!, { number: red ? 5 : tile[0]! });
  return red ? t("tile.red", { tile: name }) : name;
}

export function tileAsset(tile: string): string {
  return `./tiles/Regular/${tileAssetName(tile)}.svg`;
}

export function Tile({
  tile,
  className = "",
}: {
  tile: string;
  className?: string;
}) {
  const { t } = useLanguage();
  const back = tile === BACK;
  return (
    <span
      className={`em-face ${back ? "is-back" : ""} ${className}`}
      role="img"
      aria-label={tileName(tile, t)}
    >
      {!back && (
        <img
          src={tileAsset(tile)}
          alt=""
          width="300"
          height="400"
          draggable={false}
        />
      )}
    </span>
  );
}
