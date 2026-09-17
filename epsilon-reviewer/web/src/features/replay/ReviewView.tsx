import { lazy, Suspense, useState } from "react";
import { LoaderCircle } from "lucide-react";
import type { ReviewResult } from "../../api/types";
import { useLanguage } from "../../language/LanguageProvider";
import { ShareDialog } from "../sharing/ShareDialog";

const Replay = lazy(() =>
  import("./Replay").then((module) => ({ default: module.Replay })),
);

export function ReviewView({
  result,
  initialSeat,
  onBack,
  backLabel,
  shareable,
}: {
  result: ReviewResult;
  initialSeat: number;
  onBack: () => void;
  backLabel?: string;
  shareable: boolean;
}) {
  const { t } = useLanguage();
  const [sharing, setSharing] = useState(false);
  return (
    <>
      <Suspense
        fallback={
          <div className="replay-loading" role="status">
            <LoaderCircle className="spin" aria-hidden="true" />
            {t("replay.loading")}
          </div>
        }
      >
        <Replay
          key={result.resultId}
          result={result}
          initialSeat={initialSeat}
          onBack={onBack}
          backLabel={backLabel}
          onShare={shareable ? () => setSharing(true) : undefined}
        />
      </Suspense>
      {sharing && shareable && (
        <ShareDialog
          key={result.resultId}
          entry={result}
          onClose={() => setSharing(false)}
        />
      )}
    </>
  );
}
