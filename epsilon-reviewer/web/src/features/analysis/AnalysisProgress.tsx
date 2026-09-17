import { RefreshCw } from "lucide-react";
import type { ImportedRecord, Model, ReviewResult } from "../../api/types";
import { errorKey, type MessageKey } from "../../language/catalog";
import { modelLabel, sourceLabel } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import { useAnalysisJob } from "./useAnalysisJob";

const statusKeys = {
  queued: "progress.queued",
  running: "progress.running",
  saving: "progress.saving",
  completed: "progress.saved",
  failed: "progress.failure",
  cancelled: "progress.cancellation",
} satisfies Record<string, MessageKey>;
export function AnalysisProgress({
  jobId,
  record,
  model,
  onComplete,
  onBack,
}: {
  jobId: string;
  record: ImportedRecord;
  model: Model;
  onComplete: (result: ReviewResult) => void;
  onBack: () => void;
}) {
  const { t } = useLanguage();
  const { job, error, cancelling, cancel, retry } = useAnalysisJob(
    jobId,
    onComplete,
  );
  const terminal = job.status === "failed" || job.status === "cancelled";
  const title =
    error !== null
      ? "progress.disconnected"
      : job.status === "failed"
        ? "progress.failed"
        : job.status === "cancelled"
          ? "progress.cancelled"
          : job.status === "saving"
            ? "progress.saving"
            : "progress.analysing";
  return (
    <section className="content-card progress-card">
      <span className="eyebrow">{modelLabel(model)}</span>
      <h2>{t(title)}</h2>
      <p>{sourceLabel(record.metadata.source, t)}</p>
      <progress
        value={job.progress}
        max="100"
        aria-label={t("progress.label")}
      />
      <div className="progress-caption">
        <span>{t(statusKeys[job.status])}</span>
        <strong>{Math.round(job.progress)}%</strong>
      </div>
      {job.error && (
        <p className="error" role="alert">
          {t(errorKey(job.error))}
        </p>
      )}
      {error !== null && (
        <div className="job-connection-error" role="alert">
          <p>{t(errorKey(error))}</p>
          <p>{t("progress.reconnectHint")}</p>
          <button className="button secondary" onClick={retry}>
            <RefreshCw />
            {t("progress.refresh")}
          </button>
        </div>
      )}
      <div className="progress-actions">
        {terminal ? (
          <button className="button" onClick={onBack}>
            {t("common.back")}
          </button>
        ) : (
          <button
            className="button"
            disabled={cancelling}
            onClick={() => void cancel()}
          >
            {t(cancelling ? "progress.cancelPending" : "progress.cancel")}
          </button>
        )}
      </div>
    </section>
  );
}
