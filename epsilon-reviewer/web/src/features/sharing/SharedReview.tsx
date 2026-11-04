import { ArrowLeft, CircleAlert, LoaderCircle } from "lucide-react";
import { ApiError } from "../../api/errors";
import { api } from "../../api/platform";
import { useResource } from "../../hooks/useResource";
import { errorKey } from "../../language/catalog";
import { LanguageSelect } from "../../language/LanguageSelect";
import { useLanguage } from "../../language/LanguageProvider";
import { isAvailable } from "../models/availability";
import { ReviewView } from "../replay/ReviewView";
import "./sharing.css";

const goHome = () => window.location.assign("./");

/** 共有リンクの閲覧。Cookie を送らず、持ち主の画面（取り込み・履歴）は出さない。 */
export function SharedReview({ resultId }: { resultId: string }) {
  const { t } = useLanguage();
  const { state, reload } = useResource(
    async (signal) => {
      const [document, catalog] = await Promise.all([
        api.sharedResult(resultId, signal),
        // カタログを読めなければ、もしもを使えないだけで解析結果は開く。
        api.models(signal).catch(() => null),
      ]);
      const scenario =
        catalog !== null && isAvailable(catalog, document.model)
          ? api.scenarioClient(resultId, "share")
          : null;
      return { document, scenario };
    },
    [resultId],
  );

  if (state.status === "ready")
    return (
      <ReviewView
        key={resultId}
        document={state.value.document}
        initialSeat={0}
        scenario={state.value.scenario}
        onShare={null}
        back="home"
        onBack={goHome}
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
                <button className="button" onClick={reload}>
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
