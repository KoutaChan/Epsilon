import { FolderOpen } from "lucide-react";
import type { HistoryEntry } from "../../../../protocol/model.ts";
import {
  dateLabel,
  modelLabel,
  playerName,
  sourceLabel,
} from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import { HistoryMenu, type MenuAction } from "./HistoryMenu";

/** 履歴の 1 項目のカード。席は最終の点数が高い順に並べ、点数の無い席は最後に置く。 */
export function HistoryCard({
  entry,
  busy,
  menuOpen,
  shareable,
  onToggleMenu,
  onChoose,
  onOpen,
}: {
  entry: HistoryEntry;
  busy: boolean;
  menuOpen: boolean;
  shareable: boolean;
  onToggleMenu: (trigger: HTMLElement) => void;
  onChoose: (action: MenuAction) => void;
  onOpen: () => void;
}) {
  const { t, locale } = useLanguage();
  const { record } = entry;
  const source = sourceLabel(record.source, t);
  const scores = record.names
    .map((name, seat) => ({
      seat,
      name: playerName(name, seat, t),
      points: record.finalScores?.[seat],
    }))
    .sort((a, b) => (b.points ?? -Infinity) - (a.points ?? -Infinity));
  return (
    <article className="history-card">
      <div className="history-card-top">
        <div>
          <strong className="source-label">{source}</strong>
          <span className="history-date">
            {t("history.analysedAt", {
              date: dateLabel(entry.createdAt, locale, t),
            })}
          </span>
          <p className="history-model">{modelLabel(entry.model)}</p>
        </div>
        <HistoryMenu
          source={source}
          open={menuOpen}
          busy={busy}
          shareable={shareable}
          onToggle={onToggleMenu}
          onChoose={onChoose}
        />
      </div>
      <div className="history-card-bottom">
        <div className="history-scores">
          <span className="score-caption">{t("history.finalScores")}</span>
          <dl>
            {scores.map((player) => (
              <div key={player.seat}>
                <dt>{player.name}</dt>
                <dd>
                  {player.points === undefined ? (
                    t("common.unknown")
                  ) : (
                    <>
                      {player.points.toLocaleString(locale)}
                      <small>{t("common.points")}</small>
                    </>
                  )}
                </dd>
              </div>
            ))}
          </dl>
        </div>
        <button
          className="button open-button"
          disabled={busy}
          onClick={onOpen}
          aria-label={t("history.open", { source })}
        >
          <FolderOpen />
          <span>{t("common.open")}</span>
        </button>
      </div>
    </article>
  );
}
