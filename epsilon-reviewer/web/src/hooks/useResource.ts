import { useCallback, useEffect, useState } from "react";

export type ResourceState<T> =
  | { status: "loading" }
  | { status: "error"; error: unknown }
  | { status: "ready"; value: T };

export interface Resource<T> {
  state: ResourceState<T>;
  /** もう一度読む。読みかけは中止する。 */
  reload(): void;
}

/**
 * deps が変わるたび・reload のたびに load を呼ぶ。前の読みは中止し、中止した読みの結果は当てない。
 * load は deps の値だけを使う。
 */
export function useResource<T>(
  load: (signal: AbortSignal) => Promise<T>,
  deps: readonly unknown[],
): Resource<T> {
  const [state, setState] = useState<ResourceState<T>>({ status: "loading" });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setState((current) =>
      current.status === "loading" ? current : { status: "loading" },
    );
    load(controller.signal).then(
      (value) => {
        if (!controller.signal.aborted) setState({ status: "ready", value });
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setState({ status: "error", error });
      },
    );
    return () => controller.abort();
  }, [attempt, ...deps]);
  const reload = useCallback(() => setAttempt((value) => value + 1), []);
  return { state, reload };
}
