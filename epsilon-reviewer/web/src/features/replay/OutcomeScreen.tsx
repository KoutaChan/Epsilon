import { useLayoutEffect, useRef } from "react";
import { ArrowLeft, ArrowRight, RotateCcw, Share2 } from "lucide-react";
import { playerName, roundName, seatName } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import {
  collectWinningHands,
  hasRecordedMatchEnd,
  rankFinalPlayers,
  roundScoreChanges,
} from "./model/outcomeSummary.ts";
import type { ReplayRound } from "./model/types.ts";
import type { Playback } from "./usePlayback.ts";
import { WinningDetails } from "./WinningDetails";
import { WinningHandRow } from "./WinningHand";
import "./outcome.css";

/** 局の精算（和了の手・点数の増減）と、対局の結果（順位）。 */
export function OutcomeScreen({
  playback,
  onBack,
  backLabel,
  onShare,
}: {
  playback: Playback;
  onBack: () => void;
  backLabel: string;
  onShare: (() => void) | null;
}) {
  const { t, locale } = useLanguage();
  const {
    model,
    stage,
    round,
    snapshot,
    nextRound,
    outcomeEvents,
    move,
    dispatch,
  } = playback;
  const heading = useRef<HTMLHeadingElement>(null);
  const content = useRef<HTMLDivElement>(null);
  const isMatch = stage === "match-result";
  const matchEnded = hasRecordedMatchEnd(model);
  const scoreTitle = t(
    matchEnded ? "outcome.finalScores" : "outcome.lastScores",
  );
  const winners = collectWinningHands(outcomeEvents);
  const changes = roundScoreChanges(playback.roundStart, snapshot);
  const number = (value: number) => value.toLocaleString(locale);
  const delta = (value: number) =>
    t(
      value > 0 ? "outcome.gain" : value < 0 ? "outcome.loss" : "outcome.even",
      {
        points: number(Math.abs(value)),
      },
    );
  const name = (seat: number) =>
    playerName(snapshot.players[seat]!.name, seat, t);
  const roundLabel = (entry: ReplayRound) =>
    entry.honba
      ? t("outcome.roundHonba", {
          round: roundName(entry, t),
          honba: t("replay.honbaCount", { count: entry.honba }),
        })
      : roundName(entry, t);
  const title = isMatch
    ? t(matchEnded ? "outcome.matchTitle" : "outcome.recordEnd")
    : winners.length === 0
      ? t("event.ryukyoku")
      : winners.length === 1
        ? t(`outcome.${winners[0]!.method}`)
        : t("outcome.multipleRon", { count: winners.length });
  useLayoutEffect(() => {
    heading.current!.focus({ preventScroll: true });
    content.current!.scrollTop = 0;
  }, [stage, round.index]);
  return (
    <section
      className="em-outcome"
      data-outcome={isMatch ? "match" : "round"}
      aria-labelledby="outcome-title"
    >
      <div className="em-outcome-panel">
        <header className="em-outcome-header">
          <div>
            <p>{isMatch ? scoreTitle : roundLabel(round)}</p>
            <h1 id="outcome-title" ref={heading} tabIndex={-1}>
              {title}
            </h1>
          </div>
          {!isMatch && winners.length === 1 && (
            <span className="em-outcome-winner">
              {name(winners[0]!.player.seat)}
            </span>
          )}
        </header>
        <div className="em-outcome-scroll" ref={content}>
          {isMatch ? (
            <ol className="em-final-ranking" aria-label={scoreTitle}>
              {rankFinalPlayers(snapshot.players, model.record.finalScores).map(
                ({ player, score, rank }) => (
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
                ),
              )}
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
                                name: name(winner.target),
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
                    {snapshot.players.map((player) => {
                      const change = changes[player.seat]!;
                      return (
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
                              change > 0
                                ? "is-positive"
                                : change < 0
                                  ? "is-negative"
                                  : ""
                            }
                          >
                            {delta(change)}
                          </td>
                          <td>{number(player.score)}</td>
                        </tr>
                      );
                    })}
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
                {nextRound && <small>{roundLabel(nextRound)}</small>}
              </span>
              <ArrowRight />
            </button>
          )}
        </footer>
      </div>
    </section>
  );
}
