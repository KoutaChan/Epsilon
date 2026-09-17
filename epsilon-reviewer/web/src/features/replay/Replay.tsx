import { useRef, type CSSProperties } from "react";
import { RefreshCw } from "lucide-react";
import type { ReviewResult } from "../../api/types";
import { playerName, roundName } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import { CandidateList } from "./CandidateList";
import { RecentEventList, replayEventText } from "./RecentEventList";
import { describeReplayEvent } from "./recentEvents";
import { OutcomeScreen } from "./OutcomeScreen";
import { PlaybackControls } from "./PlaybackControls";
import { TableOverlay } from "./TableOverlay";
import { usePlayback } from "./usePlayback";
import { useTable } from "./useTable";
import "./table.css";

export function Replay({
  result,
  initialSeat,
  onBack,
  backLabel,
  onShare,
}: {
  result: ReviewResult;
  initialSeat: number;
  onBack: () => void;
  backLabel?: string;
  onShare?: () => void;
}) {
  const { t } = useLanguage();
  const root = useRef<HTMLDivElement>(null);
  const playback = usePlayback(result, initialSeat);
  const homeLabel = backLabel ?? t("replay.home");
  const share =
    onShare &&
    (() => {
      playback.dispatch({ type: "pause" });
      onShare();
    });
  const {
    round,
    snapshot,
    event,
    eventPrevious,
    recentEvents,
    candidates,
    seat,
    reveal,
  } = playback;
  const { viewport, layout, error, retry } = useTable(root, playback);
  const doraTile =
    layout === "desktop"
      ? Math.round(viewport.width / 30)
      : layout === "landscape"
        ? 24
        : 23;
  const style = {
    "--em-viewport-width": `${viewport.width}px`,
    "--em-viewport-height": `${viewport.height}px`,
    "--em-pc-ui-scale": String(viewport.width / 1440),
    "--em-dora-tile": `${doraTile}px`,
    "--em-dora-width": `${doraTile * 5 + 22}px`,
    "--em-hud-inset": layout === "portrait" ? "10px" : "16px",
  } as CSSProperties;
  return (
    <div
      id="epsilon-review-material"
      ref={root}
      className="review-root"
      data-stage={playback.stage}
      aria-label={t("replay.title")}
      style={style}
    >
      <div className="em-device" data-layout={layout} style={style}>
        <div className="em-app">
          <span className="em-current em-sr-only" aria-live="polite">
            {roundName(round, t)},{" "}
            {replayEventText(
              describeReplayEvent(event, eventPrevious, reveal, seat),
              t,
            )}
          </span>
          <PlaybackControls
            result={result}
            playback={playback}
            surface={root}
            onBack={onBack}
            backLabel={homeLabel}
            onShare={share}
          />
          <aside
            className="em-review-detail"
            aria-label={t("replay.details")}
            inert={playback.stage !== "table"}
          >
            {candidates.length > 0 ? (
              <CandidateList candidates={candidates} portrait />
            ) : (
              <RecentEventList
                events={recentEvents}
                reveal={reveal}
                seat={seat}
                perspective={playerName(snapshot.players[seat].name, seat, t)}
              />
            )}
          </aside>
          <TableOverlay playback={playback} />
          {playback.stage !== "table" && (
            <OutcomeScreen
              result={result}
              playback={playback}
              onBack={onBack}
              backLabel={homeLabel}
              onShare={share}
            />
          )}
          {error && playback.stage === "table" && (
            <div className="render-error" role="alert">
              <p>{t("errors.renderer_failed")}</p>
              <button onClick={retry}>
                <RefreshCw />
                {t("replay.retryRenderer")}
              </button>
              <button onClick={onBack}>{homeLabel}</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
