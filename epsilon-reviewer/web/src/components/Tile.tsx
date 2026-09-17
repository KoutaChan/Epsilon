import type { Translate } from "../language/catalog";
import { useLanguage } from "../language/LanguageProvider";

const honors: Record<string, string> = {
  E: "Ton",
  S: "Nan",
  W: "Shaa",
  N: "Pei",
  P: "Haku",
  F: "Hatsu",
  C: "Chun",
};
const suits: Record<string, string> = { m: "Man", p: "Pin", s: "Sou" };
export function tileName(tile: string, t: Translate): string {
  if (tile === "back" || tile === "?") return t("tile.back");
  if (honors[tile]) return t(`tile.${tile}` as "tile.E");
  const red = tile.includes("r") || tile[0] === "0";
  const name = t(`tile.${tile[1]}` as "tile.m", { number: red ? 5 : tile[0] });
  return red ? t("tile.red", { tile: name }) : name;
}
export function tileAsset(tile: string): string {
  const name =
    honors[tile] ||
    `${suits[tile[1]]}${tile.includes("r") || tile[0] === "0" ? "5-Dora" : tile[0]}`;
  return `./tiles/Regular/${name}.svg`;
}
export function Tile({
  tile,
  className = "",
}: {
  tile: string;
  className?: string;
}) {
  const { t } = useLanguage();
  const back = tile === "back" || tile === "?";
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
