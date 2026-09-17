import { useEffect, useState } from "react";
import { api, ApiError } from "../../api/client";
import type { Job, ReviewResult } from "../../api/types";

export function useAnalysisJob(
  jobId: string,
  onComplete: (result: ReviewResult) => void,
) {
  const [job, setJob] = useState<Job>({
    jobId,
    status: "queued",
    progress: 0,
    resultId: null,
    error: null,
  });
  const [error, setError] = useState<unknown>(null);
  const [attempt, setAttempt] = useState(0);
  const [cancelling, setCancelling] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    setError(null);
    async function poll() {
      try {
        const next = await api.job(jobId, controller.signal);
        if (controller.signal.aborted) return;
        setJob(next);
        if (next.status === "completed") {
          if (!next.resultId)
            throw new ApiError(
              "A completed analysis has no result ID.",
              "invalid_response",
            );
          const result = await api.result(next.resultId, controller.signal);
          if (!controller.signal.aborted) onComplete(result);
        } else if (next.status !== "failed" && next.status !== "cancelled")
          timer = setTimeout(poll, 750);
      } catch (cause) {
        if (!controller.signal.aborted) setError(cause);
      }
    }
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [jobId, attempt, onComplete]);
  async function cancel() {
    setCancelling(true);
    setError(null);
    try {
      await api.cancel(jobId);
      setAttempt((value) => value + 1);
    } catch (cause) {
      setError(cause);
      setCancelling(false);
    }
  }
  return {
    job,
    error,
    cancelling,
    cancel,
    retry: () => setAttempt((value) => value + 1),
  };
}
