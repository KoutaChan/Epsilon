import { Tile, tileName } from "../../components/Tile";
import { playerName } from "../../language/format";
import type { Translate } from "../../language/catalog";
import { useLanguage } from "../../language/LanguageProvider";
import {
  describeReplayEvent,
  type RecentReplayEvent,
  type ReplayEventDetails,
} from "./recentEvents";

export function replayEventText(
  details: ReplayEventDetails,
  t: Translate,
): string {
  const action = t(details.action);
  const label = details.player
    ? t("event.description", {
        name: playerName(details.player.name, details.player.seat, t),
        action,
      })
    : action;
  return [label, ...details.tiles.map((tile) => tileName(tile, t))].join(" · ");
}

export function RecentEventList({
  events,
  reveal,
  seat,
  perspective,
}: {
  events: readonly RecentReplayEvent[];
  reveal: boolean;
  seat: number;
  perspective: string;
}) {
  const { t } = useLanguage();
  return (
    <>
      <div className="em-review-heading">
        <h2>{t("replay.recent")}</h2>
        <span>{t("replay.perspective", { name: perspective })}</span>
      </div>
      <ol className="em-review-list em-recent-events">
        {events.map(({ event, previous, kind }) => {
          const details = describeReplayEvent(event, previous, reveal, seat);
          const name = details.player
            ? playerName(details.player.name, details.player.seat, t)
            : null;
          return (
            <li
              key={`${kind}-${event.index}`}
              className="em-review-row em-review-action em-event-row"
            >
              <div className="em-review-copy">
                {name && (
                  <span className="em-event-player" title={name}>
                    {name}
                  </span>
                )}
                <span className="em-choice-name">{t(details.action)}</span>
              </div>
              {details.tiles.length > 0 && (
                <div className="em-choice-tiles">
                  {details.tiles.map((tile, index) => (
                    <Tile tile={tile} key={index} />
                  ))}
                </div>
              )}
            </li>
          );
        })}
      </ol>
    </>
  );
}
