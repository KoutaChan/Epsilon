import { useCallback, useEffect, useRef, useState } from "react";

export interface Task {
  busy: boolean;
  error: unknown;
  clearError(): void;
  /** 前の実行を置き換えて実行する。結果・失敗・実行中かどうかを反映するのは最後に始めた実行だけで、画面が消えた後は何も反映しない。 */
  run<T>(
    operation: () => Promise<T>,
    complete: (value: T) => void,
  ): Promise<void>;
}

export function useTask(): Task {
  const generation = useRef(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  useEffect(
    () => () => {
      generation.current++;
    },
    [],
  );
  const run = useCallback(
    async <T>(operation: () => Promise<T>, complete: (value: T) => void) => {
      const id = ++generation.current;
      setBusy(true);
      setError(null);
      try {
        const value = await operation();
        if (generation.current === id) complete(value);
      } catch (cause) {
        if (generation.current === id) setError(cause);
      } finally {
        if (generation.current === id) setBusy(false);
      }
    },
    [],
  );
  const clearError = useCallback(() => setError(null), []);
  return { busy, error, clearError, run };
}
