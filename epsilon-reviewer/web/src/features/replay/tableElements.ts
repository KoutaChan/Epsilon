import { createRef, type RefCallback, type RefObject } from "react";
import type { SeatSlot, TableHost } from "./renderer/types.ts";

/** 描画が読む要素の ref。画面の部品が付け、useTableRenderer が描画に渡す。 */
export interface TableElements {
  readonly root: RefObject<HTMLDivElement | null>;
  readonly device: RefObject<HTMLDivElement | null>;
  readonly board: RefObject<HTMLElement | null>;
  readonly labels: Readonly<Record<SeatSlot, RefObject<HTMLDivElement | null>>>;
  readonly playback: RefObject<HTMLElement | null>;
  readonly dora: RefObject<HTMLDivElement | null>;
  readonly panel: RefObject<HTMLElement | null>;
  readonly candidates: RefObject<HTMLElement | null>;
  /** 名札が避ける再生の操作。 */
  readonly controls: ReadonlySet<Element>;
  /** 再生の操作に付ける ref。外れた要素は controls から外す。 */
  readonly control: RefCallback<Element>;
}

export function createTableElements(): TableElements {
  const controls = new Set<Element>();
  return {
    root: createRef(),
    device: createRef(),
    board: createRef(),
    labels: {
      south: createRef(),
      east: createRef(),
      north: createRef(),
      west: createRef(),
    },
    playback: createRef(),
    dora: createRef(),
    panel: createRef(),
    candidates: createRef(),
    controls,
    control: (element) => {
      if (!element) return;
      controls.add(element);
      return () => {
        controls.delete(element);
      };
    },
  };
}

function mounted<T>(ref: RefObject<T | null>): T {
  if (ref.current === null)
    throw new Error("The table elements are not mounted.");
  return ref.current;
}

/** 付いた要素から描画の host を作る。描画を作る時（要素が付いた後）に呼ぶ。 */
export function tableHost(elements: TableElements): TableHost {
  const { labels } = elements;
  return {
    root: mounted(elements.root),
    board: mounted(elements.board),
    device: mounted(elements.device),
    playerLabels: {
      south: mounted(labels.south),
      east: mounted(labels.east),
      north: mounted(labels.north),
      west: mounted(labels.west),
    },
    playback: mounted(elements.playback),
    dora: mounted(elements.dora),
    controls: elements.controls,
    panel: () => elements.panel.current,
    candidates: () => elements.candidates.current,
  };
}
