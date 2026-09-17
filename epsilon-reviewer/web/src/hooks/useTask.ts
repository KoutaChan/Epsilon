import { useCallback, useEffect, useRef, useState } from "react";

// 非同期処理の完了は、開始した画面・接続先の寿命内でのみ反映する。
export function useTask() {
  const pending = useRef(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const invalidate = useCallback(() => {
    pending.current++;
    setBusy(false);
    setError(null);
  }, []);
  useEffect(
    () => () => {
      pending.current++;
    },
    [],
  );
  const run = useCallback(
    async <T>(operation: () => Promise<T>, complete: (value: T) => void) => {
      const id = ++pending.current;
      setBusy(true);
      setError(null);
      try {
        const value = await operation();
        if (pending.current === id) complete(value);
      } catch (cause) {
        if (pending.current === id) setError(cause);
      } finally {
        if (pending.current === id) setBusy(false);
      }
    },
    [],
  );
  return { busy, error, clearError: () => setError(null), run, invalidate };
}
