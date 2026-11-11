import { useEffect, type RefObject } from "react";
import type { Playback } from "./usePlayback.ts";

/** これらの要素の中でキーを押したりホイールを回したりしても、再生は動かさない。 */
const INTERACTIVE_TARGETS =
  "input,textarea,select,button,summary,a,dialog,[contenteditable]:not([contenteditable=false]),[role=dialog],[role=menu],.em-review-detail,.em-scenario-panel,.em-scenario-editor,.em-scenario-targets";

/** 小さな wheel のイベントが続くとき（タッチパッドや慣性スクロール）は、縦の量がこれだけたまるごとに 1 つ動かす（px）。 */
const WHEEL_STEP_PX = 50;
/** wheel のイベントがこれより長く空いたら新しい操作として数え、次のイベントですぐ 1 つ動かす（ms）。 */
const WHEEL_PAUSE_MS = 150;
/** deltaMode の値のうち、量が px のもの（WheelEvent.DOM_DELTA_PIXEL）。 */
const DELTA_PIXEL = 0;

/** 続いている wheel の操作。rest は、まだ位置の移動に替えていない縦の量（px、0 以上）。 */
interface WheelTravel {
  readonly time: number;
  readonly direction: number;
  readonly rest: number;
}

const insideInteractive = (target: EventTarget | null, selector: string) =>
  target instanceof Element && target.closest(selector) !== null;

/**
 * wheel のイベント 1 つで位置を動かすか（1 つのイベントで動かすのは 1 つまで）と、次のイベントへ持ち越す状態を返す。
 * 操作の始めのイベントはすぐ動かす。続くイベントと、向きを変えた後のイベントは、量が WHEEL_STEP_PX たまるごとに動かす。
 *
 * @param amount 縦の量（px）。正が進む向き
 * @param time イベントの時刻（ms）
 */
function wheelStep(
  previous: WheelTravel | null,
  amount: number,
  time: number,
): { step: boolean; travel: WheelTravel } {
  const direction = Math.sign(amount);
  const carried =
    previous === null || time - previous.time > WHEEL_PAUSE_MS
      ? WHEEL_STEP_PX
      : previous.direction === direction
        ? previous.rest
        : 0;
  const total = carried + Math.abs(amount);
  return {
    step: total >= WHEEL_STEP_PX,
    travel: { time, direction, rest: total % WHEEL_STEP_PX },
  };
}

/**
 * ← → で 1 つ戻る・進む。Space で再生と一時停止を切り替え、Esc でメニューを閉じる。縦のホイールは 1 刻みで 1 つ進む・戻る。
 * タッチパッドのように小さな wheel のイベントが続くときは、イベントの数ではなく縦の量に応じて動かす。
 */
export function usePlaybackInput({
  surface,
  playback,
  menuOpen,
  closeMenu,
}: {
  surface: RefObject<HTMLElement | null>;
  playback: Playback;
  menuOpen: boolean;
  closeMenu: () => void;
}) {
  const { move, togglePlay } = playback;
  useEffect(() => {
    const element = surface.current!;
    const keyboard = (event: KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (event.key === "Escape" && menuOpen) {
        event.preventDefault();
        closeMenu();
        return;
      }
      if (
        menuOpen ||
        event.ctrlKey ||
        event.metaKey ||
        event.altKey ||
        event.shiftKey ||
        insideInteractive(event.target, INTERACTIVE_TARGETS)
      )
        return;
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        move(event.key === "ArrowRight" ? 1 : -1);
      } else if (event.key === " ") {
        event.preventDefault();
        if (!event.repeat) togglePlay();
      }
    };
    let travel: WheelTravel | null = null;
    const wheel = (event: WheelEvent) => {
      if (
        menuOpen ||
        event.defaultPrevented ||
        event.ctrlKey ||
        event.metaKey ||
        event.altKey ||
        event.shiftKey ||
        Math.abs(event.deltaX) >= Math.abs(event.deltaY) ||
        insideInteractive(
          event.target,
          `${INTERACTIVE_TARGETS},.em-outcome-scroll`,
        )
      )
        return;
      event.preventDefault();
      // 行やページで数える wheel は刻みのあるホイールなので、イベント 1 つを 1 刻みとして扱う。
      const amount =
        event.deltaMode === DELTA_PIXEL
          ? event.deltaY
          : Math.sign(event.deltaY) * WHEEL_STEP_PX;
      const next = wheelStep(travel, amount, event.timeStamp);
      travel = next.travel;
      if (next.step) move(next.travel.direction);
    };
    document.addEventListener("keydown", keyboard);
    element.addEventListener("wheel", wheel, { passive: false });
    return () => {
      document.removeEventListener("keydown", keyboard);
      element.removeEventListener("wheel", wheel);
    };
  }, [surface, menuOpen, closeMenu, move, togglePlay]);
}
