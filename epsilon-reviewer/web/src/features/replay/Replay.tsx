import { useState, type CSSProperties } from "react";
import { RefreshCw } from "lucide-react";
import { playerName, roundName } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import type { ScenarioClient } from "../../review-entry.ts";
import { CandidateList } from "./CandidateList";
import { describeReplayEvent } from "./model/recentEvents.ts";
import type { Timelines } from "./model/timeline.ts";
import type { ReplayModel } from "./model/types.ts";
import { OutcomeScreen } from "./OutcomeScreen";
import { PlaybackControls } from "./PlaybackControls";
import { RecentEventList, replayEventText } from "./RecentEventList";
import { ScenarioControls, ScenarioEditor } from "./ScenarioPanel";
import { targetKey } from "./scenario/scenario.ts";
import { useScenario } from "./scenario/useScenario.ts";
import { createTableElements } from "./tableElements.ts";
import { TableOverlay } from "./TableOverlay";
import { usePlayback } from "./usePlayback.ts";
import { useTableRenderer } from "./useTableRenderer.ts";
import "./table.css";

export function Replay({
  model,
  timelines,
  initialSeat,
  client,
  onShare,
  backLabel,
  onBack,
}: {
  model: ReplayModel;
  timelines: Timelines;
  initialSeat: number;
  client: ScenarioClient | null;
  onShare: (() => void) | null;
  backLabel: string;
  onBack: () => void;
}) {
  const { t } = useLanguage();
  const [elements] = useState(createTableElements);
  const playback = usePlayback(model, timelines, initialSeat);
  const scenario = useScenario(playback, client);
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
    seat,
    reveal,
    stage,
  } = playback;
  const { candidates, comparison } = scenario;
  const { viewport, layout, targets, error, retry } = useTableRenderer(
    elements,
    playback,
    { snapshot: scenario.display, targets: scenario.editing },
  );
  const doraTile =
    layout === "desktop"
      ? Math.round(viewport.width / 30)
      : layout === "landscape"
        ? 24
        : 23;
  const style = {
    "--em-pc-ui-scale": String(viewport.width / 1440),
    // 候補欄は左家の名札の下から画面の下までに収めるので、横長で縦の短い画面では高さに合わせて小さくする。
    "--em-pc-candidate-scale": String(
      Math.min(viewport.width / 1440, viewport.height / 760),
    ),
    "--em-dora-tile": `${doraTile}px`,
    "--em-dora-width": `${doraTile * 5 + 22}px`,
    "--em-hud-inset": layout === "portrait" ? "10px" : "16px",
  } as CSSProperties;
  return (
    <div
      id="epsilon-review-material"
      ref={elements.root}
      className="review-root"
      data-stage={stage}
      aria-label={t("replay.title")}
    >
      <div
        className="em-device"
        ref={elements.device}
        data-layout={layout}
        style={style}
      >
        <div className="em-app">
          <span className="em-current em-sr-only" aria-live="polite">
            {t("replay.current", {
              round: roundName(round, t),
              event: replayEventText(
                describeReplayEvent(event, eventPrevious, reveal, seat),
                t,
              ),
            })}
          </span>
          <PlaybackControls
            elements={elements}
            playback={playback}
            scenario={scenario}
            onBack={onBack}
            backLabel={backLabel}
            onShare={share}
          />
          {/* 縦の画面では卓の下の一覧に、もしもの操作を先頭に置き、編集の間は候補の代わりに編集を出す。 */}
          {layout === "portrait" && (
            <aside
              className="em-review-detail"
              aria-label={t("replay.details")}
              inert={stage !== "table"}
              data-scenario={scenario.active ? scenario.listStatus : undefined}
            >
              {scenario.active && (
                <ScenarioControls
                  elements={elements}
                  scenario={scenario}
                  playback={playback}
                />
              )}
              {scenario.editing && scenario.selection ? (
                <ScenarioEditor
                  key={targetKey(scenario.selection)}
                  scenario={scenario}
                  playback={playback}
                  floating={false}
                />
              ) : candidates.length > 0 ? (
                <CandidateList
                  candidates={candidates}
                  comparison={comparison}
                  portrait
                  onCall={scenario.branchTo}
                  afterCall={
                    scenario.branch?.status === "done"
                      ? scenario.branch.call
                      : null
                  }
                  onLeave={scenario.leaveBranch}
                />
              ) : (
                !scenario.active && (
                  <RecentEventList
                    events={recentEvents}
                    reveal={reveal}
                    seat={seat}
                    perspective={playerName(
                      snapshot.players[seat]!.name,
                      seat,
                      t,
                    )}
                  />
                )
              )}
            </aside>
          )}
          <TableOverlay
            elements={elements}
            playback={playback}
            scenario={scenario}
            targets={targets}
            layout={layout}
          />
          {/* 横の画面の編集は卓の外の層に浮かべ、手牌の上の再生の操作より前に出す。 */}
          {stage === "table" &&
            layout !== "portrait" &&
            scenario.editing &&
            scenario.selection && (
              <ScenarioEditor
                key={targetKey(scenario.selection)}
                scenario={scenario}
                playback={playback}
                floating
              />
            )}
          {stage !== "table" && (
            <OutcomeScreen
              playback={playback}
              onBack={onBack}
              backLabel={backLabel}
              onShare={share}
            />
          )}
          {error && stage === "table" && (
            <div className="render-error" role="alert">
              <p>{t("errors.renderer_failed")}</p>
              <button onClick={retry}>
                <RefreshCw />
                {t("replay.retryRenderer")}
              </button>
              <button onClick={onBack}>{backLabel}</button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
