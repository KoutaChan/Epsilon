import { useCallback, useEffect, useState } from "react";

export type ResourceState<T> =
  | { status: "loading" }
  | { status: "error"; error: unknown }
  | { status: "ready"; value: T };

export interface Resource<T> {
  state: ResourceState<T>;
  /** もう一度読み込む。読み込み中のものは中止する。 */
  reload(): void;
}

/**
 * 依存の値（`deps`）が変わったときと、読み込み直しを求められたときに読み込む。前の読み込みは中止し、中止した読み込みの
 * 結果は状態に反映しない。読み込む関数は `deps` の値だけを使うこと。
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
