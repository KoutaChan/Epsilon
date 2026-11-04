import { useCallback, useEffect, useRef, useState } from "react";

export interface Task {
  busy: boolean;
  error: unknown;
  clearError(): void;
  /** 前の run を置き換える。結果・失敗・busy は最後に始めた run だけが、画面の寿命の間だけ当てる。 */
  run<T>(operation: () => Promise<T>, complete: (value: T) => void): Promise<void>;
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
