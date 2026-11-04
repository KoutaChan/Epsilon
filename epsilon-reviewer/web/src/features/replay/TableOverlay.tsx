import { Tile, tileName } from "../../components/Tile";
import {
  playerName,
  roundName,
  seatName,
  windName,
} from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import { CandidateList } from "./CandidateList";
import type { Layout, TableTargets } from "./renderer/types.ts";
import { ScenarioControls } from "./ScenarioPanel";
import { ScenarioPick, ScenarioTargets } from "./ScenarioTargets";
import type { Scenario } from "./scenario/useScenario.ts";
import type { TableElements } from "./tableElements.ts";
import type { Playback } from "./usePlayback.ts";

const SLOTS = ["south", "east", "north", "west"] as const;

/** 卓の上の DOM（ドラ表示・名札・候補の欄・もしもの操作と当たり枠）と、読み上げの文。 */
export function TableOverlay({
  elements,
  playback,
  scenario,
  targets,
  layout,
}: {
  elements: TableElements;
  playback: Playback;
  scenario: Scenario;
  targets: TableTargets | null;
  layout: Layout;
}) {
  const { t, locale } = useLanguage();
  const { round, seat, reveal, stage } = playback;
  // 卓の文字も、もしもの局面を出している間はその局面で書く。
  const snapshot = scenario.display;
  const { candidates, comparison, editing } = scenario;
  const list = (tiles: readonly string[]) =>
    tiles.map((tile) => tileName(tile, t)).join(t("replay.listSeparator"));
  const sticks = [
    {
      kind: "kyotaku" as const,
      label: t("replay.kyotakuCount", { count: snapshot.kyotaku }),
      content: (
        <>
          <span className="em-point-stick is-riichi" aria-hidden="true" />
          <span aria-hidden="true">{t("replay.kyotaku")}</span>
          <strong aria-hidden="true">{snapshot.kyotaku}</strong>
        </>
      ),
    },
    {
      kind: "honba" as const,
      label: t("replay.honbaCount", { count: snapshot.honba }),
      content: (
        <>
          <span className="em-point-stick" aria-hidden="true" />
          <span aria-hidden="true">{t("replay.honba")}</span>
          <strong aria-hidden="true">{snapshot.honba}</strong>
        </>
      ),
    },
  ];
  return (
    <div className="em-table-surface" inert={stage !== "table"}>
      <section
        className="em-board"
        ref={elements.board}
        aria-label={t("replay.board", { round: roundName(round, t) })}
      >
        <div
          className="em-dora-hud"
          ref={elements.dora}
          aria-label={t("replay.dora")}
        >
          <div className="em-dora">
            {Array.from({ length: 5 }, (_, index) => {
              const tile = snapshot.dora[index] ?? "back";
              return editing && index < snapshot.dora.length ? (
                <ScenarioPick
                  key={index}
                  scenario={scenario}
                  target={{ kind: "dora", index }}
                  className="em-scenario-dora"
                  label={t("scenario.targetValue", {
                    target: t("scenario.dora"),
                    value: tileName(tile, t),
                  })}
                >
                  <Tile tile={tile} />
                </ScenarioPick>
              ) : (
                <Tile key={index} tile={tile} />
              );
            })}
          </div>
          <span className="em-dora-count">
            {sticks.map(({ kind, label, content }) =>
              editing ? (
                <ScenarioPick
                  key={kind}
                  scenario={scenario}
                  target={{ kind }}
                  className="em-dora-stat"
                  label={label}
                >
                  {content}
                </ScenarioPick>
              ) : (
                <span key={kind} className="em-dora-stat" aria-label={label}>
                  {content}
                </span>
              ),
            )}
          </span>
        </div>
        {editing && targets && (
          <ScenarioTargets
            scenario={scenario}
            targets={targets}
            playback={playback}
          />
        )}
        {SLOTS.map((slot, offset) => {
          const player = snapshot.players[(seat + offset) % 4]!,
            name = playerName(player.name, player.seat, t),
            wind = windName((player.seat - round.dealer + 4) % 4, t, true);
          return (
            <div
              key={slot}
              ref={elements.labels[slot]}
              className="em-player-label"
              data-player-label={slot}
              data-active={player.seat === snapshot.activeSeat}
              title={t("replay.playerTitle", {
                seat: seatName(player.seat, round.dealer, t),
                name,
                score: player.score.toLocaleString(locale),
              })}
            >
              <span className="em-player-wind">
                {player.riichi
                  ? t("replay.windRiichi", {
                      wind,
                      riichi: t("mahjong.riichi"),
                    })
                  : wind}
              </span>
              <span className="em-player-name">{name}</span>
            </div>
          );
        })}
        {layout !== "portrait" && (
          <section
            className="em-action-candidates"
            ref={elements.candidates}
            aria-label={t("replay.topCandidates")}
            hidden={candidates.length === 0}
            data-scenario={scenario.active ? scenario.listStatus : undefined}
          >
            {candidates.length > 0 && (
              <CandidateList
                candidates={candidates}
                comparison={comparison}
                onCall={scenario.branchTo}
                afterCall={
                  scenario.branch?.status === "done"
                    ? scenario.branch.call
                    : null
                }
                onLeave={scenario.leaveBranch}
              />
            )}
          </section>
        )}
        {scenario.active && layout !== "portrait" && (
          <ScenarioControls
            elements={elements}
            scenario={scenario}
            playback={playback}
          />
        )}
        <div className="em-sr-only">
          {snapshot.players.map((player) => {
            const name = playerName(player.name, player.seat, t),
              visible = reveal || player.seat === seat;
            return (
              <section
                key={player.seat}
                aria-label={t("replay.hand", { name })}
              >
                <p>
                  {t("replay.labelValue", {
                    label: t("replay.hand", { name }),
                    value: visible ? list(player.hand) : t("tile.back"),
                  })}
                </p>
                {player.draw && (
                  <p>
                    {t("replay.labelValue", {
                      label: t("replay.draw"),
                      value: visible
                        ? tileName(player.draw, t)
                        : t("tile.back"),
                    })}
                  </p>
                )}
                <p>
                  {t("replay.labelValue", {
                    label: t("replay.river", { name }),
                    value: list(
                      player.river
                        .filter((tile) => !tile.called)
                        .map((tile) => tile.tile),
                    ),
                  })}
                </p>
                <p>
                  {t("replay.labelValue", {
                    label: t("replay.melds", { name }),
                    value: player.melds
                      .map((meld) => list(meld.tiles))
                      .join(t("replay.meldSeparator")),
                  })}
                </p>
              </section>
            );
          })}
        </div>
      </section>
    </div>
  );
}
