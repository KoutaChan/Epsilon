import { useLayoutEffect, useRef, type CSSProperties } from "react";
import { ArrowLeft, ArrowRight, RotateCcw, Share2 } from "lucide-react";
import type { Meld, ReviewResult } from "../../api/types";
import { Tile } from "../../components/Tile";
import { playerName, roundName, seatName } from "../../language/format";
import { actionMessageKeys } from "../../language/actionMessages";
import { useLanguage } from "../../language/LanguageProvider";
import {
  collectWinningHands,
  hasRecordedMatchEnd,
  rankFinalPlayers,
  roundScoreChanges,
  type WinningHand,
} from "./outcomeSummary";
import { WinningDetails } from "./WinningDetails";
import type { Playback } from "./usePlayback";
import "./outcome.css";

const RESULT_TILE_GAP = 0.025;
const RESULT_GROUP_GAP = 0.35;

function WinningHandRow({ winner }: { winner: WinningHand }) {
  const { t } = useLanguage();
  const {
    hand,
    winningTile,
    player: { melds },
  } = winner;
  const groupCount =
    Number(hand.length > 0) + Number(winningTile !== null) + melds.length;
  let widthUnits =
    hand.length +
    Number(winningTile !== null) +
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
      {winningTile && (
        <div className="em-result-winning-tile">
          <Tile tile={winningTile} />
          <span>{t("outcome.winningTile")}</span>
        </div>
      )}
      {melds.length > 0 && (
        <div className="em-result-melds" aria-label={t("outcome.melds")}>
          {melds.map((_, index) => {
            const meldIndex = melds.length - 1 - index;
            return <WinningMeld key={meldIndex} meld={melds[meldIndex]} />;
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
                <Tile tile={meld.tiles[meld.addedIndex]} />
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}

export function OutcomeScreen({
  result,
  playback,
  onBack,
  backLabel,
  onShare,
}: {
  result: ReviewResult;
  playback: Playback;
  onBack: () => void;
  backLabel: string;
  onShare?: () => void;
}) {
  const { t, locale } = useLanguage();
  const { stage, round, snapshot, nextRound, outcomeEvents, move, dispatch } =
    playback;
  const heading = useRef<HTMLHeadingElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const isMatch = stage === "match-result";
  const matchEnded = hasRecordedMatchEnd(result);
  const scoreTitle = t(
    matchEnded ? "outcome.finalScores" : "outcome.lastScores",
  );
  const winners = collectWinningHands(outcomeEvents);
  const changes = roundScoreChanges(playback.roundStart, snapshot);
  const number = (value: number) => value.toLocaleString(locale);
  const delta = (value: number) =>
    `${value > 0 ? "+" : value < 0 ? "−" : "±"}${number(Math.abs(value))}`;
  const name = (seat: number) =>
    playerName(snapshot.players[seat].name, seat, t);
  const roundLabel = `${roundName(round, t)}${round.honba ? ` · ${t("replay.honbaCount", { count: round.honba })}` : ""}`;
  const nextLabel =
    nextRound &&
    `${roundName(nextRound, t)}${nextRound.honba ? ` · ${t("replay.honbaCount", { count: nextRound.honba })}` : ""}`;
  const title = isMatch
    ? t(matchEnded ? "outcome.matchTitle" : "outcome.recordEnd")
    : winners.length === 0
      ? t("event.ryukyoku")
      : winners.length === 1
        ? t(`outcome.${winners[0].method}`)
        : t("outcome.multipleRon", { count: winners.length });
  useLayoutEffect(() => {
    heading.current!.focus({ preventScroll: true });
    content.current!.scrollTop = 0;
  }, [stage, round.id]);
  return (
    <section
      className="em-outcome"
      data-outcome={isMatch ? "match" : "round"}
      aria-labelledby="outcome-title"
    >
      <div className="em-outcome-panel">
        <header className="em-outcome-header">
          <div>
            <p>{isMatch ? scoreTitle : roundLabel}</p>
            <h1 id="outcome-title" ref={heading} tabIndex={-1}>
              {title}
            </h1>
          </div>
          {!isMatch && winners.length === 1 && (
            <span className="em-outcome-winner">
              {name(winners[0].player.seat)}
            </span>
          )}
        </header>
        <div className="em-outcome-scroll" ref={content}>
          {isMatch ? (
            <ol className="em-final-ranking" aria-label={scoreTitle}>
              {rankFinalPlayers(
                snapshot.players,
                result.metadata.finalScores,
              ).map(({ player, score, rank }) => (
                <li
                  key={player.seat}
                  data-rank={rank}
                  data-perspective={player.seat === playback.seat}
                >
                  <span
                    className="em-final-rank"
                    aria-label={t("outcome.rank", { rank })}
                  >
                    {rank}
                  </span>
                  <span className="em-final-name">
                    {playerName(player.name, player.seat, t)}
                    {player.seat === playback.seat && (
                      <small>{t("outcome.perspective")}</small>
                    )}
                  </span>
                  <strong>
                    {number(score)}
                    <small>{t("outcome.points")}</small>
                  </strong>
                </li>
              ))}
            </ol>
          ) : (
            <div
              className={`em-round-outcome-grid ${winners.length === 0 ? "is-draw" : ""}`}
            >
              {winners.length > 0 && (
                <div className="em-winning-hands">
                  {winners.map((winner) => (
                    <article
                      className="em-winning-hand"
                      key={winner.player.seat}
                    >
                      <div className="em-winning-heading">
                        <h2>
                          <span>
                            {seatName(winner.player.seat, round.dealer, t)}
                          </span>
                          {name(winner.player.seat)}
                        </h2>
                        <p>
                          {winner.method === "ron"
                            ? t("outcome.ronFrom", {
                                name: name(winner.target!),
                              })
                            : t(`outcome.${winner.method}`)}
                        </p>
                      </div>
                      <WinningHandRow winner={winner} />
                      <WinningDetails
                        details={winner.details}
                        dora={snapshot.dora}
                      />
                    </article>
                  ))}
                </div>
              )}
              <section
                className="em-outcome-scores"
                aria-labelledby="outcome-scores-title"
              >
                <h2 id="outcome-scores-title">{t("outcome.settlement")}</h2>
                <table>
                  <thead>
                    <tr>
                      <th scope="col">{t("outcome.player")}</th>
                      <th scope="col">{t("outcome.roundChange")}</th>
                      <th scope="col">{t("outcome.score")}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {snapshot.players.map((player) => (
                      <tr
                        key={player.seat}
                        data-winner={winners.some(
                          (winner) => winner.player.seat === player.seat,
                        )}
                      >
                        <th scope="row">
                          <span className="em-score-wind">
                            {seatName(player.seat, round.dealer, t)}
                          </span>
                          <span>{name(player.seat)}</span>
                        </th>
                        <td
                          className={
                            changes[player.seat] > 0
                              ? "is-positive"
                              : changes[player.seat] < 0
                                ? "is-negative"
                                : ""
                          }
                        >
                          {delta(changes[player.seat])}
                        </td>
                        <td>{number(player.score)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            </div>
          )}
        </div>
        <footer className="em-outcome-footer">
          <button className="em-outcome-back" onClick={() => move(-1)}>
            <ArrowLeft />
            <span>
              {t(isMatch ? "outcome.backToSettlement" : "outcome.backToTable")}
            </span>
          </button>
          {isMatch ? (
            <div className="em-outcome-final-actions">
              {onShare && (
                <button onClick={onShare}>
                  <Share2 />
                  <span>{t("share.title")}</span>
                </button>
              )}
              <button onClick={() => dispatch({ type: "round", index: 0 })}>
                <RotateCcw />
                <span>{t("outcome.replayFromStart")}</span>
              </button>
              <button className="em-outcome-primary" onClick={onBack}>
                {backLabel}
                <ArrowRight />
              </button>
            </div>
          ) : (
            <button className="em-outcome-primary" onClick={() => move(1)}>
              <span>
                {t(
                  nextRound
                    ? "outcome.nextRound"
                    : matchEnded
                      ? "outcome.toMatchResult"
                      : "outcome.toRecordResult",
                )}
                {nextLabel && <small>{nextLabel}</small>}
              </span>
              <ArrowRight />
            </button>
          )}
        </footer>
      </div>
    </section>
  );
}
