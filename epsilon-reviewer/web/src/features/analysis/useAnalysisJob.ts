import { useEffect, useState } from "react";
import { JOB_POLL_INTERVAL_MS } from "../../../../protocol/commands.ts";
import type { JobSnapshot } from "../../../../protocol/model.ts";
import type { ResultDocument } from "../../../../protocol/result.ts";
import { ApiError } from "../../api/errors";
import { api } from "../../api/platform";

export interface AnalysisJob {
  job: JobSnapshot;
  /** 進みか結果を読めなかった失敗。refresh で読み直す。 */
  error: unknown;
  /** 取り消しを頼んだ後、仕事が終わるまで真。 */
  cancelling: boolean;
  cancel(): Promise<void>;
  refresh(): void;
}

/** 解析の仕事を応答のたびに JOB_POLL_INTERVAL_MS 待って聞き直し、終われば文書を読んで onComplete を呼ぶ。 */
export function useAnalysisJob(
  jobId: string,
  onComplete: (document: ResultDocument) => void,
): AnalysisJob {
  const [job, setJob] = useState<JobSnapshot>({
    jobId,
    status: "queued",
    progress: 0,
  });
  const [error, setError] = useState<unknown>(null);
  const [attempt, setAttempt] = useState(0);
  const [cancelling, setCancelling] = useState(false);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    setError(null);
    const poll = async () => {
      try {
        const next = await api.job(jobId, controller.signal);
        if (controller.signal.aborted) return;
        setJob(next);
        if (next.status === "completed") {
          if (next.resultId === undefined)
            throw new ApiError(
              "invalid_response",
              "A completed analysis has no result ID.",
            );
          const document = await api.result(next.resultId, controller.signal);
          if (!controller.signal.aborted) onComplete(document);
        } else if (next.status !== "failed" && next.status !== "cancelled")
          timer = setTimeout(poll, JOB_POLL_INTERVAL_MS);
      } catch (cause) {
        if (!controller.signal.aborted) setError(cause);
      }
    };
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [jobId, attempt, onComplete]);
  const cancel = async () => {
    setCancelling(true);
    setError(null);
    try {
      setJob(await api.cancel(jobId));
      setAttempt((value) => value + 1);
    } catch (cause) {
      setError(cause);
      setCancelling(false);
    }
  };
  return {
    job,
    error,
    cancelling,
    cancel,
    refresh: () => setAttempt((value) => value + 1),
  };
}
