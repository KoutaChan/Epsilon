import { useEffect, type RefObject } from "react";
import type { Playback } from "./usePlayback";

const INTERACTIVE_TARGETS =
  "input,textarea,select,button,summary,a,dialog,[contenteditable]:not([contenteditable=false]),[role=dialog],[role=menu],.em-review-detail";

export function usePlaybackInput({
  surface,
  playback,
  menuOpen,
  closeMenu,
}: {
  surface: RefObject<HTMLDivElement | null>;
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
        (event.target as Element).closest(INTERACTIVE_TARGETS)
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
    const wheel = (event: WheelEvent) => {
      if (
        menuOpen ||
        event.defaultPrevented ||
        event.ctrlKey ||
        event.metaKey ||
        event.altKey ||
        event.shiftKey ||
        Math.abs(event.deltaX) >= Math.abs(event.deltaY) ||
        (event.target as Element).closest(
          `${INTERACTIVE_TARGETS},.em-outcome-scroll`,
        )
      )
        return;
      event.preventDefault();
      move(event.deltaY > 0 ? 1 : -1);
    };
    document.addEventListener("keydown", keyboard);
    element.addEventListener("wheel", wheel, { passive: false });
    return () => {
      document.removeEventListener("keydown", keyboard);
      element.removeEventListener("wheel", wheel);
    };
  }, [surface, menuOpen, closeMenu, move, togglePlay]);
}
