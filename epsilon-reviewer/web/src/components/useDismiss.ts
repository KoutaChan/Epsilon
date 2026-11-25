import { useEffect } from "react";

/** 開いているメニューの外を押すか、Escape を押したときに閉じる。 */
export function useDismiss(
  open: boolean,
  contains: (target: Element) => boolean,
  outside: () => void,
  escape?: () => void,
) {
  useEffect(() => {
    if (!open) return;
    const pointer = (event: PointerEvent) => {
      if (!(event.target instanceof Element) || !contains(event.target))
        outside();
    };
    const keyboard = (event: KeyboardEvent) => {
      if (event.key === "Escape") escape?.();
    };
    document.addEventListener("pointerdown", pointer);
    if (escape) document.addEventListener("keydown", keyboard);
    return () => {
      document.removeEventListener("pointerdown", pointer);
      if (escape) document.removeEventListener("keydown", keyboard);
    };
  }, [open, contains, outside, escape]);
}
