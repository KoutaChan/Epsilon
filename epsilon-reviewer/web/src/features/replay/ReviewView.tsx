import { lazy, Suspense, useMemo } from "react";
import { ArrowLeft, CircleAlert, LoaderCircle } from "lucide-react";
import type { ResultDocument } from "../../../../protocol/result.ts";
import { useLanguage } from "../../language/LanguageProvider";
import type { ReviewEntryProps } from "../../review-entry.ts";
import { decodeResult } from "./model/decodeResult.ts";
import { ReviewFormatError } from "./model/formatError.ts";
import { buildTimelines, type Timelines } from "./model/timeline.ts";
import type { ReplayModel } from "./model/types.ts";

const Replay = lazy(() =>
  import("./Replay").then((module) => ({ default: module.Replay })),
);

interface ReadReplay {
  readonly model: ReplayModel;
  readonly timelines: Timelines;
}

/** 文書を 1 回だけ読んで再生の位置まで作る。読めなければ ReviewFormatError を返す。 */
function readReplay(document: ResultDocument): ReadReplay | ReviewFormatError {
  try {
    const model = decodeResult(document);
    return { model, timelines: buildTimelines(model) };
  } catch (error) {
    if (!(error instanceof ReviewFormatError)) throw error;
    console.error("The analysis result cannot be read.", error.message);
    return error;
  }
}

export function ReviewView({
  document,
  initialSeat,
  scenario,
  onShare,
  back,
  onBack,
}: ReviewEntryProps) {
  const { t } = useLanguage();
  const replay = useMemo(() => readReplay(document), [document]);
  const backLabel = t(back === "history" ? "replay.home" : "common.home");
  if (replay instanceof ReviewFormatError)
    return (
      <div className="replay-loading" role="alert">
        <CircleAlert aria-hidden="true" />
        <span>{t("errors.invalid_result")}</span>
        <button className="button" onClick={onBack}>
          <ArrowLeft aria-hidden="true" />
          {backLabel}
        </button>
      </div>
    );
  return (
    <Suspense
      fallback={
        <div className="replay-loading" role="status">
          <LoaderCircle className="spin" aria-hidden="true" />
          {t("replay.loading")}
        </div>
      }
    >
      <Replay
        model={replay.model}
        timelines={replay.timelines}
        initialSeat={initialSeat}
        client={scenario}
        onShare={onShare}
        backLabel={backLabel}
        onBack={onBack}
      />
    </Suspense>
  );
}
