import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type MouseEvent,
} from "react";
import {
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  Split,
} from "lucide-react";
import { playerName, roundName, seatName } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import { OptionsMenu } from "./OptionsMenu";
import type { Scenario } from "./scenario/useScenario.ts";
import type { TableElements } from "./tableElements.ts";
import type { Playback } from "./usePlayback.ts";
import { usePlaybackInput } from "./usePlaybackInput.ts";

type Menu = "round" | "seat" | "options";

/** 再生の操作の帯。局・視点・設定の 3 つのメニューと、戻る・再生・進む・もしもの押し釦。 */
export function PlaybackControls({
  elements,
  playback,
  scenario,
  onBack,
  backLabel,
  onShare,
}: {
  elements: TableElements;
  playback: Playback;
  scenario: Scenario;
  onBack: () => void;
  backLabel: string;
  onShare: (() => void) | null;
}) {
  const { t } = useLanguage();
  const {
    model,
    round,
    snapshot,
    seat,
    playing,
    canPrevious,
    canNext,
    canPlay,
    stage,
    dispatch,
    move,
    togglePlay,
  } = playback;
  const [menu, setMenu] = useState<Menu | null>(null);
  // メニューを閉じたら、開いた札へ戻る。
  const opener = useRef<HTMLElement | null>(null);
  const close = useCallback(() => {
    if (menu) opener.current?.focus();
    setMenu(null);
  }, [menu]);
  usePlaybackInput({
    surface: elements.root,
    playback,
    menuOpen: menu !== null,
    closeMenu: close,
  });
  const toggle = (target: Menu) => (event: MouseEvent<HTMLElement>) => {
    event.preventDefault();
    dispatch({ type: "pause" });
    opener.current = event.currentTarget;
    setMenu(menu === target ? null : target);
  };
  useEffect(() => {
    const pointer = (event: PointerEvent) => {
      const node = event.target;
      if (!(node instanceof Node) || !elements.playback.current?.contains(node))
        setMenu(null);
    };
    document.addEventListener("pointerdown", pointer);
    return () => document.removeEventListener("pointerdown", pointer);
  }, [elements]);
  useEffect(() => {
    if (stage !== "table") setMenu(null);
  }, [stage]);
  // 押せないときは理由を出す。
  const scenarioReason =
    scenario.active || scenario.canStart
      ? undefined
      : scenario.available
        ? t("scenario.noTurn", { seat: seatName(seat, round.dealer, t) })
        : t("scenario.unavailableModel");
  const currentPlayer = snapshot.players[seat]!;
  return (
    <nav
      className="em-playback"
      ref={elements.playback}
      aria-label={t("replay.controls")}
      inert={stage !== "table"}
    >
      <details className="em-round-picker" open={menu === "round"}>
        <summary
          ref={elements.control}
          className="em-round-trigger"
          onClick={toggle("round")}
          aria-label={t("replay.chooseRound", { round: roundName(round, t) })}
        >
          <span className="em-round-label">{roundName(round, t, true)}</span>
          <ChevronDown />
        </summary>
        <div className="em-round-menu">
          {model.rounds.map((entry) => (
            <button
              key={entry.index}
              aria-pressed={entry.index === playback.roundIndex}
              onClick={() => {
                dispatch({ type: "round", index: entry.index });
                close();
              }}
            >
              <span>
                {entry.honba > 0
                  ? t("replay.roundHonba", {
                      round: roundName(entry, t),
                      honba: t("replay.honbaCount", { count: entry.honba }),
                    })
                  : roundName(entry, t)}
              </span>
              {entry.index === playback.roundIndex && <Check />}
            </button>
          ))}
        </div>
      </details>
      <details className="em-seat-picker" open={menu === "seat"}>
        <summary
          ref={elements.control}
          className="em-seat-trigger"
          onClick={toggle("seat")}
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
            const player = snapshot.players[(round.dealer + offset) % 4]!;
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
          ref={elements.control}
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
          ref={elements.control}
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
          ref={elements.control}
          className="em-icon"
          data-step="1"
          aria-label={t("replay.next")}
          disabled={!canNext}
          onClick={() => move(1)}
        >
          <span className="em-action-label">{t("replay.forward")}</span>
          <ChevronRight />
        </button>
        <button
          ref={elements.control}
          className="em-icon em-scenario-toggle"
          aria-label={t("scenario.button")}
          aria-pressed={scenario.active}
          disabled={!scenario.active && !scenario.canStart}
          title={scenarioReason}
          onClick={() => {
            setMenu(null);
            scenario.toggle();
          }}
        >
          <Split />
          <span className="em-action-label">{t("scenario.button")}</span>
        </button>
      </div>
      <div className="em-playback-meta">
        <OptionsMenu
          elements={elements}
          playback={playback}
          open={menu === "options"}
          onToggle={toggle("options")}
          onClose={close}
          onBack={onBack}
          backLabel={backLabel}
          onShare={onShare}
        />
      </div>
    </nav>
  );
}
