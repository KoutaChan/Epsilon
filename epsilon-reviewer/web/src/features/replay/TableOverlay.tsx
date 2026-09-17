import { Tile, tileName } from "../../components/Tile";
import {
  playerName,
  roundName,
  seatName,
  windName,
} from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import { CandidateList } from "./CandidateList";
import type { Playback } from "./usePlayback";

const slots = ["south", "east", "north", "west"] as const;
export function TableOverlay({ playback }: { playback: Playback }) {
  const { t, locale } = useLanguage();
  const { round, snapshot, candidates, seat, reveal } = playback;
  return (
    <div className="em-table-surface" inert={playback.stage !== "table"}>
      <section
        className="em-board"
        aria-label={t("replay.board", { round: roundName(round, t) })}
      >
        <div className="em-dora-hud" aria-label={t("replay.dora")}>
          <div className="em-dora">
            {Array.from({ length: 5 }, (_, index) => (
              <Tile key={index} tile={snapshot.dora[index] ?? "back"} />
            ))}
          </div>
          <span className="em-dora-count">
            <span
              className="em-dora-stat"
              aria-label={t("replay.kyotakuCount", { count: snapshot.kyotaku })}
            >
              <span className="em-point-stick is-riichi" aria-hidden="true" />
              <span aria-hidden="true">{t("replay.kyotaku")}</span>
              <strong aria-hidden="true">{snapshot.kyotaku}</strong>
            </span>
            <span
              className="em-dora-stat"
              aria-label={t("replay.honbaCount", { count: snapshot.honba })}
            >
              <span className="em-point-stick" aria-hidden="true" />
              <span aria-hidden="true">{t("replay.honba")}</span>
              <strong aria-hidden="true">{snapshot.honba}</strong>
            </span>
          </span>
        </div>
        {slots.map((slot, offset) => {
          const player = snapshot.players[(seat + offset) % 4],
            name = playerName(player.name, player.seat, t);
          return (
            <div
              key={slot}
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
                {windName((player.seat - round.dealer + 4) % 4, t, true)}
                {player.riichi ? ` · ${t("mahjong.riichi")}` : ""}
              </span>
              <span className="em-player-name">{name}</span>
            </div>
          );
        })}
        <section
          className="em-action-candidates"
          aria-label={t("replay.topCandidates")}
          hidden={candidates.length === 0}
        >
          {candidates.length > 0 && <CandidateList candidates={candidates} />}
        </section>
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
                  {t("replay.hand", { name })}:{" "}
                  {visible
                    ? player.hand.map((tile) => tileName(tile, t)).join(", ")
                    : t("tile.back")}
                </p>
                {player.draw && (
                  <p>
                    {t("replay.draw")}:{" "}
                    {visible ? tileName(player.draw, t) : t("tile.back")}
                  </p>
                )}
                <p>
                  {t("replay.river", { name })}:{" "}
                  {player.river
                    .filter((tile) => !tile.called)
                    .map((tile) => tileName(tile.tile, t))
                    .join(", ")}
                </p>
                <p>
                  {t("replay.melds", { name })}:{" "}
                  {player.melds
                    .map((meld) =>
                      meld.tiles.map((tile) => tileName(tile, t)).join(", "),
                    )
                    .join("; ")}
                </p>
              </section>
            );
          })}
        </div>
      </section>
    </div>
  );
}
