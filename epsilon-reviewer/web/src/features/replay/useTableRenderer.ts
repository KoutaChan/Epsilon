import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { roundName, windName } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import type { Snapshot } from "./model/types.ts";
import { createTableRenderer } from "./renderer/tableRenderer.ts";
import type { Layout, TableRenderer, TableTargets } from "./renderer/types.ts";
import { tableHost, type TableElements } from "./tableElements.ts";
import type { Playback } from "./usePlayback.ts";

export interface Viewport {
  readonly width: number;
  readonly height: number;
}

/** 幅 680 以下の縦長は縦、高さ 540 以下の横長は横、他は PC の配置。 */
function layoutOf({ width, height }: Viewport): Layout {
  return width <= 680 && height > width
    ? "portrait"
    : height <= 540 && width > height
      ? "landscape"
      : "desktop";
}

/**
 * 画面の大きさを測って配置を決め、卓の描画を作って局面を渡す。snapshot は卓に描く局面（もしもの写しの
 * ことがある）、targets はもしもの当たり枠を受け取るか。
 */
export function useTableRenderer(
  elements: TableElements,
  playback: Playback,
  { snapshot, targets }: { snapshot: Snapshot; targets: boolean },
) {
  const { t } = useLanguage();
  const { dispatch, rowWidth, round, event, seat, reveal, animateDraw } =
    playback;
  const [viewport, setViewport] = useState<Viewport>(() => ({
    width: window.innerWidth,
    height: window.innerHeight,
  }));
  const [error, setError] = useState(false);
  const [generation, setGeneration] = useState(0);
  const [tableTargets, setTableTargets] = useState<TableTargets | null>(null);
  const renderer = useRef<TableRenderer | null>(null);
  const layout = layoutOf(viewport);
  const uiScale = viewport.width / 1440;
  useLayoutEffect(() => {
    const element = elements.root.current!;
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
  }, [elements]);
  useEffect(() => {
    const failed = (cause: unknown) => {
      console.error("The table renderer failed.", cause);
      setError(true);
      dispatch({ type: "pause" });
    };
    try {
      renderer.current = createTableRenderer(tableHost(elements), rowWidth, {
        onError: failed,
        onTargets: setTableTargets,
      });
    } catch (cause) {
      failed(cause);
    }
    return () => {
      renderer.current?.dispose();
      renderer.current = null;
      setTableTargets(null);
    };
  }, [elements, generation, dispatch, rowWidth]);
  const labels = useMemo(
    () => ({
      remaining: t("replay.remaining"),
      loadingTiles: t("replay.loadingTiles"),
      draw: t("replay.draw"),
      round: roundName(round, t, true),
      winds: [0, 1, 2, 3].map((wind) => windName(wind, t, true)),
    }),
    [t, round],
  );
  // 卓は画面と同じ局面を借用し、React の DOM から対局状態を戻さない。
  useEffect(() => {
    renderer.current?.update({
      snapshot,
      event,
      dealer: round.dealer,
      viewSeat: seat,
      reveal,
      animateDraw,
      layout,
      uiScale,
      labels,
      targets,
    });
  }, [
    snapshot,
    event,
    round,
    seat,
    reveal,
    animateDraw,
    layout,
    uiScale,
    labels,
    targets,
    generation,
  ]);
  return {
    viewport,
    layout,
    /** 当たり枠の形。当たり枠を求めていない間は null。 */
    targets: targets ? tableTargets : null,
    error,
    retry: () => {
      setError(false);
      setGeneration((value) => value + 1);
    },
  };
}
