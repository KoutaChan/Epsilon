import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { Eye, LoaderCircle, RotateCcw } from "lucide-react";
import { errorKey, type MessageKey } from "../../language/catalog";
import { seatName } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import {
  DoraEditor,
  HandEditor,
  RiverEditor,
  ScoreEditor,
  SticksEditor,
} from "./ScenarioEditors";
import { targetKey } from "./scenario/scenario.ts";
import type { Scenario, ScenarioStatus } from "./scenario/useScenario.ts";
import type { TableElements } from "./tableElements.ts";
import type { Playback } from "./usePlayback.ts";

const ANALYSE_KEYS: Record<ScenarioStatus, MessageKey> = {
  idle: "scenario.analyse",
  ready: "scenario.analyse",
  error: "scenario.analyse",
  loading: "scenario.analysing",
  done: "scenario.analysed",
  stale: "scenario.reanalyse",
};
/** 吹き出しと押した所の間。 */
const EDITOR_GAP = 12;
/** 吹き出しと卓の縁の間。 */
const EDITOR_MARGIN = 8;

/** 右上の操作。ドラ表示と同じ幅・同じ作りで、中身が変わっても大きさを変えない。 */
export function ScenarioControls({
  elements,
  scenario,
  playback,
}: {
  elements: TableElements;
  scenario: Scenario;
  playback: Playback;
}) {
  const { t } = useLanguage();
  const { decision, changes, status } = scenario;
  const viewer = seatName(playback.seat, playback.round.dealer, t);
  const error = decision !== null && scenario.error !== null;
  const state =
    decision === null
      ? t("scenario.noTurn", { seat: viewer })
      : error
        ? t(errorKey(scenario.error))
        : changes.length === 0
          ? t("scenario.noChanges")
          : t("scenario.changeCount", { count: changes.length });
  const original = scenario.board === "original";
  return (
    <section
      className="em-scenario-panel"
      ref={elements.panel}
      aria-label={t("scenario.title")}
    >
      <div className="em-scenario-head">
        <strong>{t("scenario.title")}</strong>
        <span
          className="em-scenario-state"
          data-error={error}
          title={state}
          role="status"
        >
          {state}
        </span>
      </div>
      <div className="em-scenario-actions">
        <button
          className="em-scenario-analyse"
          disabled={
            decision === null ||
            status === "idle" ||
            status === "loading" ||
            status === "done"
          }
          onClick={scenario.analyse}
        >
          {status === "loading" && (
            <LoaderCircle className="spin" aria-hidden="true" />
          )}
          {t(ANALYSE_KEYS[status])}
        </button>
        <button
          className="em-scenario-icon"
          aria-label={t("scenario.showOriginal")}
          title={t("scenario.showOriginal")}
          aria-pressed={original}
          disabled={changes.length === 0}
          onClick={() => scenario.showBoard(original ? "edited" : "original")}
        >
          <Eye aria-hidden="true" />
        </button>
        <button
          className="em-scenario-icon"
          aria-label={t("scenario.resetAll")}
          title={t("scenario.resetAll")}
          disabled={changes.length === 0}
          onClick={scenario.revertAll}
        >
          <RotateCcw aria-hidden="true" />
        </button>
      </div>
    </section>
  );
}

/**
 * 選んだ所の編集。横長の画面では押した所のそばに浮かべ、ほかの欄を動かさない。
 * 縦の画面では卓の下の一覧の場所に置く。外を押すか Esc で閉じる。
 */
export function ScenarioEditor({
  scenario,
  playback,
  floating,
}: {
  scenario: Scenario;
  playback: Playback;
  floating: boolean;
}) {
  const target = scenario.selection!;
  const key = targetKey(target);
  const element = useRef<HTMLElement>(null);
  const [place, setPlace] = useState<{ left: number; top: number } | null>(
    null,
  );
  const { select } = scenario;
  // 手牌は横一列なので上に、ほかは横（右、入らなければ左）に置き、画面からはみ出さないようにする。
  useLayoutEffect(() => {
    if (!floating) return;
    const editor = element.current!;
    const frame = editor.offsetParent as HTMLElement;
    const anchor = frame.querySelector(`[data-scenario-key="${key}"]`);
    if (!anchor) return;
    const host = frame.getBoundingClientRect();
    const a = anchor.getBoundingClientRect();
    const width = editor.offsetWidth,
      height = editor.offsetHeight;
    const clamp = (value: number, size: number, limit: number) =>
      Math.max(EDITOR_MARGIN, Math.min(limit - size - EDITOR_MARGIN, value));
    let left: number, top: number;
    if (target.kind === "hand") {
      left = clamp(
        a.left - host.left + a.width / 2 - width / 2,
        width,
        host.width,
      );
      top = clamp(a.top - host.top - EDITOR_GAP - height, height, host.height);
    } else {
      const right = a.right - host.left + EDITOR_GAP;
      left =
        right + width <= host.width - EDITOR_MARGIN
          ? right
          : Math.max(EDITOR_MARGIN, a.left - host.left - EDITOR_GAP - width);
      top = clamp(
        a.top - host.top + a.height / 2 - height / 2,
        height,
        host.height,
      );
    }
    if (place?.left !== left || place?.top !== top) setPlace({ left, top });
  });
  useEffect(() => {
    const pointer = (event: PointerEvent) => {
      const node = event.target;
      if (
        node instanceof Element &&
        (element.current?.contains(node) || node.closest("[data-scenario-key]"))
      )
        return;
      select(null);
    };
    const keyboard = (event: KeyboardEvent) => {
      if (event.key === "Escape") select(null);
    };
    document.addEventListener("pointerdown", pointer);
    document.addEventListener("keydown", keyboard);
    return () => {
      document.removeEventListener("pointerdown", pointer);
      document.removeEventListener("keydown", keyboard);
    };
  }, [select]);
  const props = { scenario, playback };
  return (
    <section
      ref={element}
      className="em-scenario-editor"
      data-floating={floating}
      data-placed={!floating || place !== null}
      style={
        floating && place ? { left: place.left, top: place.top } : undefined
      }
    >
      {target.kind === "river" ? (
        <RiverEditor target={target} {...props} />
      ) : target.kind === "dora" ? (
        <DoraEditor target={target} {...props} />
      ) : target.kind === "hand" ? (
        <HandEditor target={target} {...props} />
      ) : target.kind === "score" ? (
        <ScoreEditor target={target} {...props} />
      ) : (
        <SticksEditor target={target} {...props} />
      )}
    </section>
  );
}
