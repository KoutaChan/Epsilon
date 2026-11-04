import type { CSSProperties } from "react";
import type { Meld } from "../../../../protocol/result.ts";
import { Tile } from "../../components/Tile";
import { actionMessageKeys } from "../../language/actionMessages";
import { useLanguage } from "../../language/LanguageProvider";
import type { WinningHand } from "./model/outcomeSummary.ts";

/** 牌の間（牌の幅を 1 とする）。 */
const RESULT_TILE_GAP = 0.025;
/** 手牌・和了牌・副露の組の間。 */
const RESULT_GROUP_GAP = 0.35;

/** 和了の手。伏せた手牌、和了牌、副露（新しい順）の一列。 */
export function WinningHandRow({ winner }: { winner: WinningHand }) {
  const { t } = useLanguage();
  const {
    hand,
    winningTile,
    player: { melds },
  } = winner;
  const groupCount = Number(hand.length > 0) + 1 + melds.length;
  let widthUnits =
    hand.length +
    1 +
    Math.max(0, hand.length - 1) * RESULT_TILE_GAP +
    Math.max(0, groupCount - 1) * RESULT_GROUP_GAP;
  for (const meld of melds) {
    const tileCount = meld.tiles.length - Number(meld.type === "KAKAN");
    widthUnits +=
      tileCount +
      Number(meld.calledIndex >= 0) / 3 +
      (tileCount - 1) * RESULT_TILE_GAP;
  }
  const style = {
    "--em-result-width-units": widthUnits,
    "--em-result-tile-gap-units": RESULT_TILE_GAP,
    "--em-result-group-gap-units": RESULT_GROUP_GAP,
  } as CSSProperties;
  return (
    <div
      className="em-result-hand"
      aria-label={t("outcome.winningHand")}
      data-stacked={melds.some((meld) => meld.type === "KAKAN")}
      style={style}
    >
      {hand.length > 0 && (
        <div className="em-result-concealed">
          {hand.map((tile, index) => (
            <Tile key={index} tile={tile} />
          ))}
        </div>
      )}
      <div className="em-result-winning-tile">
        <Tile tile={winningTile} />
        <span>{t("outcome.winningTile")}</span>
      </div>
      {melds.length > 0 && (
        <div className="em-result-melds" aria-label={t("outcome.melds")}>
          {melds.map((_, index) => {
            const meldIndex = melds.length - 1 - index;
            return <WinningMeld key={meldIndex} meld={melds[meldIndex]!} />;
          })}
        </div>
      )}
    </div>
  );
}

function WinningMeld({ meld }: { meld: Meld }) {
  const { t } = useLanguage();
  const concealedKan = meld.type === "ANKAN";
  const redTile = concealedKan
    ? meld.tiles.find((tile) => tile.includes("r"))
    : undefined;
  return (
    <div
      className="em-result-meld"
      role="group"
      aria-label={t(actionMessageKeys[meld.type])}
    >
      {meld.tiles.map((tile, index) => {
        if (meld.type === "KAKAN" && index === meld.addedIndex) return null;
        const sideways = index === meld.calledIndex;
        // 暗槓は両端を伏せ、赤の 5 は 2 枚目に見せる。
        const value = concealedKan
          ? index === 0 || index === meld.tiles.length - 1
            ? "back"
            : index === 1 && redTile
              ? redTile
              : tile.replace("r", "")
          : tile;
        return (
          <span
            className="em-result-meld-tile"
            data-sideways={sideways}
            key={index}
          >
            <Tile tile={value} />
            {sideways && meld.type === "KAKAN" && (
              <span className="em-result-added-tile">
                <Tile tile={meld.tiles[meld.addedIndex]!} />
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}
