import { useEffect, useState } from "react";
import { JOB_POLL_INTERVAL_MS } from "../../../../protocol/commands.ts";
import type { JobSnapshot } from "../../../../protocol/model.ts";
import type { ResultDocument } from "../../../../protocol/result.ts";
import { ApiError } from "../../api/errors";
import { api } from "../../api/platform";

export interface AnalysisJob {
  job: JobSnapshot;
  /** 進み具合か結果を読めなかったときの失敗。refresh を呼ぶと読み直す。 */
  error: unknown;
  /** 取り消しを要求してから、ジョブが終わるまで true。 */
  cancelling: boolean;
  cancel(): Promise<void>;
  refresh(): void;
}

/** 解析のジョブの状態を繰り返し取り直す。前の応答が返ってから JOB_POLL_INTERVAL_MS（ms）待って、次を取る。完了したら文書を読んで onComplete を呼ぶ。失敗か取り消しで終わったら、そこで取り直しをやめる。 */
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
