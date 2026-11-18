import { BACK, tileAsset } from "./tileAssets.ts";
import { HONOR_CODES, SUITS } from "@epsilon-reviewer/protocol/tiles";
import type { Translate } from "../language/catalog";
import { useLanguage } from "../language/LanguageProvider";

export function tileName(tile: string, t: Translate): string {
  if (tile === BACK) return t("tile.back");
  const honor = HONOR_CODES.find((code) => code === tile);
  if (honor) return t(`tile.${honor}`);
  const red = tile.length === 3;
  const name = t(`tile.${tile[1] as (typeof SUITS)[number]}`, {
    number: red ? 5 : tile[0]!,
  });
  return red ? t("tile.red", { tile: name }) : name;
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
