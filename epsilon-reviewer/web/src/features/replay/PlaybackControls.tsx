import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from "react";
import {
  ArrowLeft,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  MoreHorizontal,
  Pause,
  Play,
  Share2,
} from "lucide-react";
import type { ReviewResult } from "../../api/types";
import { playerName, roundName, seatName } from "../../language/format";
import { LanguageSelect } from "../../language/LanguageSelect";
import { useLanguage } from "../../language/LanguageProvider";
import { summarizePlayerDecisions } from "./decisionStats";
import type { Playback } from "./usePlayback";
import { usePlaybackInput } from "./usePlaybackInput";

type Menu = "round" | "seat" | "options";
export function PlaybackControls({
  surface,
  result,
  playback,
  onBack,
  backLabel,
  onShare,
}: {
  surface: RefObject<HTMLDivElement | null>;
  result: ReviewResult;
  playback: Playback;
  onBack: () => void;
  backLabel: string;
  onShare?: () => void;
}) {
  const { t, locale } = useLanguage();
  const {
    round,
    snapshot,
    seat,
    stepIndex,
    playing,
    reveal,
    last,
    canPrevious,
    canNext,
    canPlay,
    dispatch,
    move,
    togglePlay,
  } = playback;
  const [menu, setMenu] = useState<Menu | null>(null);
  const navigation = useRef<HTMLElement>(null);
  const triggers = useRef<Partial<Record<Menu, HTMLElement | null>>>({});
  const close = useCallback(() => {
    if (menu) triggers.current[menu]?.focus();
    setMenu(null);
  }, [menu]);
  usePlaybackInput({
    surface,
    playback,
    menuOpen: menu !== null,
    closeMenu: close,
  });
  const toggle = (target: Menu) => {
    dispatch({ type: "pause" });
    setMenu(menu === target ? null : target);
  };
  useEffect(() => {
    const pointer = (event: PointerEvent) => {
      if (!navigation.current!.contains(event.target as Node)) setMenu(null);
    };
    document.addEventListener("pointerdown", pointer);
    return () => document.removeEventListener("pointerdown", pointer);
  }, []);
  useEffect(() => {
    if (playback.stage !== "table") setMenu(null);
  }, [playback.stage]);
  const statistics = useMemo(
    () => summarizePlayerDecisions(result.rounds),
    [result],
  );
  const playerStats = statistics[seat];
  const currentPlayer = snapshot.players[seat];
  const name = playerName(currentPlayer.name, seat, t);
  const percent = new Intl.NumberFormat(locale, {
    style: "percent",
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
  const formatDecisionRate = (count: number) =>
    playerStats.reviewed === 0
      ? "—"
      : percent.format(count / playerStats.reviewed);
  return (
    <nav
      className="em-playback"
      ref={navigation}
      aria-label={t("replay.controls")}
      inert={playback.stage !== "table"}
    >
      <details className="em-round-picker" open={menu === "round"}>
        <summary
          ref={(element) => {
            triggers.current.round = element;
          }}
          className="em-round-trigger"
          onClick={(event) => {
            event.preventDefault();
            toggle("round");
          }}
          aria-label={t("replay.chooseRound", { round: roundName(round, t) })}
        >
          <span className="em-round-label">{roundName(round, t, true)}</span>
          <ChevronDown />
        </summary>
        <div className="em-round-menu">
          {result.rounds.map((entry, index) => (
            <button
              key={entry.id}
              aria-pressed={index === playback.roundIndex}
              onClick={() => {
                dispatch({ type: "round", index });
                close();
              }}
            >
              <span>
                {roundName(entry, t)}
                {entry.honba > 0
                  ? ` ${t("replay.honbaCount", { count: entry.honba })}`
                  : ""}
              </span>
              {index === playback.roundIndex && <Check />}
            </button>
          ))}
        </div>
      </details>
      <details className="em-seat-picker" open={menu === "seat"}>
        <summary
          ref={(element) => {
            triggers.current.seat = element;
          }}
          className="em-seat-trigger"
          onClick={(event) => {
            event.preventDefault();
            toggle("seat");
          }}
          aria-label={t("replay.chooseSeat", {
            seat: seatName(seat, round.dealer, t),
            name: playerName(currentPlayer.name, seat, t),
          })}
        >
          <span className="em-seat-label">
            {seatName(seat, round.dealer, t)}
          </span>
          <ChevronDown />
        </summary>
        <div className="em-seat-menu">
          {[0, 1, 2, 3].map((offset) => {
            const player = snapshot.players[(round.dealer + offset) % 4];
            return (
              <button
                key={player.seat}
                aria-pressed={player.seat === seat}
                onClick={() => {
                  dispatch({ type: "seat", seat: player.seat });
                  close();
                }}
              >
                <span className="em-seat-option-wind">
                  {seatName(player.seat, round.dealer, t)}
                </span>
                <span className="em-seat-option-name">
                  {playerName(player.name, player.seat, t)}
                </span>
                {player.seat === seat && <Check />}
              </button>
            );
          })}
        </div>
      </details>
      <div className="em-transport">
        <button
          className="em-icon"
          data-step="-1"
          aria-label={t("replay.previous")}
          disabled={!canPrevious}
          onClick={() => move(-1)}
        >
          <ChevronLeft />
          <span className="em-action-label">{t("common.back")}</span>
        </button>
        <button
          className="em-icon em-play"
          aria-label={t(playing ? "replay.pause" : "replay.play")}
          disabled={!canPlay && !playing}
          onClick={() => {
            setMenu(null);
            togglePlay();
          }}
        >
          {playing ? <Pause /> : <Play />}
        </button>
        <button
          className="em-icon"
          data-step="1"
          aria-label={t("replay.next")}
          disabled={!canNext}
          onClick={() => move(1)}
        >
          <span className="em-action-label">{t("replay.forward")}</span>
          <ChevronRight />
        </button>
      </div>
      <div className="em-playback-meta">
        <details className="em-info" open={menu === "options"}>
          <summary
            ref={(element) => {
              triggers.current.options = element;
            }}
            onClick={(event) => {
              event.preventDefault();
              toggle("options");
            }}
            aria-label={t("replay.options")}
          >
            <MoreHorizontal />
          </summary>
          <div className="em-info-content">
            <section
              className="em-decision-stats"
              aria-label={t("replay.statistics")}
            >
              <strong className="em-stats-player" title={name}>
                {name}
              </strong>
              <span className="em-stats-scope">
                {t("replay.statisticsScope", {
                  count: playerStats.reviewed.toLocaleString(locale),
                })}
              </span>
              <dl>
                <div title={t("replay.agreementDescription")}>
                  <dt>{t("replay.agreementRate")}</dt>
                  <dd
                    aria-label={
                      playerStats.reviewed === 0
                        ? t("replay.noDecisions")
                        : undefined
                    }
                  >
                    {formatDecisionRate(playerStats.matched)}
                  </dd>
                </div>
                <div title={t("replay.badMoveDescription")}>
                  <dt>{t("replay.badMoveRate")}</dt>
                  <dd
                    aria-label={
                      playerStats.reviewed === 0
                        ? t("replay.noDecisions")
                        : undefined
                    }
                  >
                    {formatDecisionRate(playerStats.badMoves)}
                  </dd>
                </div>
              </dl>
            </section>
            <label className="em-reveal-option">
              <span>{t("replay.reveal")}</span>
              <input
                className="em-reveal-hands"
                type="checkbox"
                checked={reveal}
                onChange={(event) =>
                  dispatch({ type: "reveal", reveal: event.target.checked })
                }
              />
            </label>
            <span className="em-position">
              {stepIndex + 1} / {last + 1}
            </span>
            <label className="em-seek-label" htmlFor="review-seek">
              {t("replay.position")}
            </label>
            <input
              id="review-seek"
              className="em-timeline"
              type="range"
              min="0"
              max={last}
              step="1"
              value={stepIndex}
              onChange={(event) =>
                dispatch({ type: "seek", index: Number(event.target.value) })
              }
            />
            <LanguageSelect />
            <p>
              {result.model.displayName} {result.model.version}
            </p>
            <div className="em-review-actions">
              {onShare && (
                <button
                  className="review-home-button"
                  onClick={() => {
                    close();
                    onShare();
                  }}
                >
                  <Share2 />
                  {t("share.title")}
                </button>
              )}
              <button className="review-home-button" onClick={onBack}>
                <ArrowLeft />
                {backLabel}
              </button>
            </div>
            <p className="tile-credit">{t("replay.credit")}</p>
          </div>
        </details>
      </div>
    </nav>
  );
}
