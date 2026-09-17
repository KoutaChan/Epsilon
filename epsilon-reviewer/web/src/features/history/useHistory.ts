import { useCallback, useEffect, useState } from "react";
import { api } from "../../api/client";
import type { HistoryEntry } from "../../api/types";

type HistoryState =
  | { status: "loading" }
  | { status: "error"; error: unknown }
  | { status: "ready"; entries: HistoryEntry[] };
export function useHistory(connection: number) {
  const [state, setState] = useState<HistoryState>({ status: "loading" });
  const [revision, setRevision] = useState(0);
  const refresh = useCallback(() => setRevision((value) => value + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    setState({ status: "loading" });
    api
      .history(controller.signal)
      .then(({ results }) => {
        if (!controller.signal.aborted)
          setState({ status: "ready", entries: results });
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) setState({ status: "error", error });
      });
    return () => controller.abort();
  }, [connection, revision]);
  return { state, refresh };
}
export type History = ReturnType<typeof useHistory>;
