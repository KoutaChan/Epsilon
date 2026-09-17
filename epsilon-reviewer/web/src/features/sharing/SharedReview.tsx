import { useEffect, useState } from "react";
import { ArrowLeft, CircleAlert, LoaderCircle } from "lucide-react";
import { api, ApiError } from "../../api/client";
import type { ReviewResult } from "../../api/types";
import { errorKey } from "../../language/catalog";
import { LanguageSelect } from "../../language/LanguageSelect";
import { useLanguage } from "../../language/LanguageProvider";
import { ReviewView } from "../replay/ReviewView";
import "./sharing.css";

type SharedReviewState =
  | { status: "loading" }
  | { status: "ready"; result: ReviewResult }
  | { status: "error"; error: unknown };

export function SharedReview({ id }: { id: string }) {
  const { t } = useLanguage();
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<SharedReviewState>({ status: "loading" });
  useEffect(() => {
    const controller = new AbortController();
    setState({ status: "loading" });
    api.sharedResult(id, controller.signal).then(
      (result) => {
        if (!controller.signal.aborted) setState({ status: "ready", result });
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setState({ status: "error", error });
      },
    );
    return () => controller.abort();
  }, [id, attempt]);

  if (state.status === "ready")
    return (
      <ReviewView
        key={state.result.resultId}
        result={state.result}
        initialSeat={0}
        shareable={false}
        backLabel={t("common.home")}
        onBack={() => window.location.assign("./")}
      />
    );

  const unavailable =
    state.status === "error" &&
    state.error instanceof ApiError &&
    state.error.code === "share_not_found";
  return (
    <main className="shared-review-state">
      <div className="shared-review-language">
        <LanguageSelect />
      </div>
      <section
        className="shared-review-message"
        aria-label={t("share.pageTitle")}
      >
        {state.status === "loading" ? (
          <div role="status">
            <LoaderCircle className="spin" aria-hidden="true" />
            <h1>{t("share.loadingReview")}</h1>
          </div>
        ) : (
          <>
            <div role="alert">
              <CircleAlert aria-hidden="true" />
              <h1>{t("share.unavailable")}</h1>
              <p>{t(errorKey(state.error))}</p>
            </div>
            <div className="shared-review-actions">
              {!unavailable && (
                <button
                  className="button"
                  onClick={() => setAttempt((value) => value + 1)}
                >
                  {t("common.retry")}
                </button>
              )}
              <a className="button primary" href="./">
                <ArrowLeft aria-hidden="true" />
                {t("common.home")}
              </a>
            </div>
          </>
        )}
      </section>
    </main>
  );
}
