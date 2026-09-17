import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type RefObject,
} from "react";
import { useLanguage } from "../../language/LanguageProvider";
import { roundName, windName } from "../../language/format";
import type { Playback } from "./usePlayback";
import { createTableRenderer } from "./table-renderer";

export type Layout = "desktop" | "portrait" | "landscape";
export function useTable(
  root: RefObject<HTMLDivElement | null>,
  playback: Playback,
) {
  const { t, language } = useLanguage();
  const [viewport, setViewport] = useState({
    width: window.innerWidth,
    height: window.innerHeight,
  });
  const [error, setError] = useState(false);
  const [generation, setGeneration] = useState(0);
  const renderer = useRef<ReturnType<typeof createTableRenderer> | null>(null);
  const layout: Layout =
    viewport.width <= 680 && viewport.height > viewport.width
      ? "portrait"
      : viewport.height <= 540 && viewport.width > viewport.height
        ? "landscape"
        : "desktop";
  useLayoutEffect(() => {
    const element = root.current!;
    const observer = new ResizeObserver(() => {
      const box = element.getBoundingClientRect();
      const width = Math.round(box.width),
        height = Math.round(box.height);
      setViewport((previous) =>
        previous.width === width && previous.height === height
          ? previous
          : { width, height },
      );
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [root]);
  useEffect(() => {
    const element = root.current!;
    const failed = () => {
      setError(true);
      playback.dispatch({ type: "pause" });
    };
    element.addEventListener("review-render-error", failed);
    try {
      renderer.current = createTableRenderer(element, playback.playerRowWidth);
    } catch (cause) {
      console.error("Failed to initialize the table renderer.", cause);
      failed();
    }
    return () => {
      element.removeEventListener("review-render-error", failed);
      renderer.current?.dispose();
      renderer.current = null;
    };
  }, [root, generation, playback.dispatch, playback.playerRowWidth]);
  const labels = useMemo(
    () => ({
      remaining: t("replay.remaining"),
      loadingTiles: t("replay.loadingTiles"),
      draw: t("replay.draw"),
      round: roundName(playback.round, t, true),
      winds: [0, 1, 2, 3].map((wind) => windName(wind, t, true)),
    }),
    [language, playback.round],
  );
  // Canvas は画面と同じ DTO を借用し、React の DOM から対局状態を復元しない。
  useEffect(() => {
    renderer.current?.update({
      snapshot: playback.snapshot,
      event: playback.event,
      round: playback.round,
      viewSeat: playback.seat,
      reveal: playback.reveal,
      animateDraw: playback.animateDraw,
      layout,
      labels,
    });
  }, [
    playback.snapshot,
    playback.event,
    playback.round,
    playback.seat,
    playback.reveal,
    playback.animateDraw,
    layout,
    labels,
    generation,
  ]);
  return {
    viewport,
    layout,
    error,
    retry: () => {
      setError(false);
      setGeneration((value) => value + 1);
    },
  };
}
