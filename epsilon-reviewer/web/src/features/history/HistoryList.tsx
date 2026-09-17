import { useEffect, useRef, useState } from "react";
import {
  CircleAlert,
  Download,
  FolderOpen,
  LoaderCircle,
  MoreHorizontal,
  RefreshCw,
  Search,
  Share2,
  Trash2,
} from "lucide-react";
import { isDesktop } from "../../api/client";
import type { HistoryEntry } from "../../api/types";
import { errorKey } from "../../language/catalog";
import {
  dateLabel,
  modelLabel,
  playerName,
  sourceLabel,
} from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import type { History } from "./useHistory";

export interface HistoryActions {
  share?: (entry: HistoryEntry) => void;
  open: (entry: HistoryEntry) => void;
  export: (entry: HistoryEntry) => void;
  reanalyse: (entry: HistoryEntry) => void;
  delete: (entry: HistoryEntry) => void;
}
export function HistoryList({
  history,
  busy,
  actions,
}: {
  history: History;
  busy: boolean;
  actions: HistoryActions;
}) {
  const { t, locale } = useLanguage();
  const [search, setSearch] = useState("");
  const [menu, setMenu] = useState<string | null>(null);
  const trigger = useRef<HTMLElement | null>(null);
  const { state, refresh } = history;
  const share = actions.share;
  useEffect(() => {
    const closeOutside = (event: PointerEvent) => {
      if (!(event.target as Element).closest(".history-menu")) setMenu(null);
    };
    const closeKeyboard = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenu(null);
        trigger.current?.focus();
      }
    };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", closeKeyboard);
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      document.removeEventListener("keydown", closeKeyboard);
    };
  }, []);
  const entries =
    state.status === "ready"
      ? state.entries.filter((entry) =>
          `${entry.metadata.source} ${sourceLabel(entry.metadata.source, t)} ${entry.metadata.names.join(" ")} ${modelLabel(entry.model)}`
            .toLocaleLowerCase(locale)
            .includes(search.toLocaleLowerCase(locale)),
        )
      : [];
  const act = (action: (entry: HistoryEntry) => void, entry: HistoryEntry) => {
    setMenu(null);
    trigger.current?.focus();
    action(entry);
  };
  return (
    <section className="history-section">
      <div className="history-heading">
        <div>
          <h2>
            {t("history.title")}{" "}
            <span>{state.status === "ready" ? state.entries.length : ""}</span>
          </h2>
          <p>{t(isDesktop ? "history.local" : "history.browser")}</p>
        </div>
        <button
          className="icon-button"
          aria-label={t("history.refresh")}
          disabled={state.status === "loading"}
          onClick={refresh}
        >
          <RefreshCw className={state.status === "loading" ? "spin" : ""} />
        </button>
      </div>
      <label className="search-field">
        <Search />
        <input
          type="search"
          placeholder={t("history.searchPlaceholder")}
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          aria-label={t("history.search")}
        />
      </label>
      {state.status === "loading" ? (
        <div className="empty-state" role="status">
          <LoaderCircle className="spin" />
          <p>{t("history.loading")}</p>
        </div>
      ) : state.status === "error" ? (
        <div className="empty-state error" role="alert">
          <CircleAlert />
          <p>{t(errorKey(state.error))}</p>
          <button className="button" onClick={refresh}>
            {t("common.retry")}
          </button>
        </div>
      ) : entries.length === 0 ? (
        <div className="empty-state">
          <FolderOpen />
          <h3>{t(search ? "history.noMatches" : "history.empty")}</h3>
          <p>{t(search ? "history.searchHint" : "history.emptyHint")}</p>
        </div>
      ) : (
        <div className="history-list">
          {entries.map((entry) => {
            const source = sourceLabel(entry.metadata.source, t);
            const scores = entry.metadata.names
              .map((name, seat) => ({
                seat,
                name: playerName(name, seat, t),
                points: entry.metadata.finalScores?.[seat],
              }))
              .sort(
                (a, b) => (b.points ?? -Infinity) - (a.points ?? -Infinity),
              );
            return (
              <article className="history-card" key={entry.resultId}>
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
                  <details
                    className="history-menu"
                    open={menu === entry.resultId}
                  >
                    <summary
                      className="icon-button"
                      aria-label={t("history.actions", { source })}
                      onClick={(event) => {
                        event.preventDefault();
                        trigger.current = event.currentTarget;
                        setMenu(
                          menu === entry.resultId ? null : entry.resultId,
                        );
                      }}
                    >
                      <MoreHorizontal />
                    </summary>
                    <div className="floating-menu">
                      {share && (
                        <button
                          disabled={busy}
                          onClick={() => act(share, entry)}
                        >
                          <Share2 />
                          {t("share.title")}
                        </button>
                      )}
                      <button
                        disabled={busy}
                        onClick={() => act(actions.export, entry)}
                      >
                        <Download />
                        {t("history.export")}
                      </button>
                      <button
                        disabled={busy}
                        onClick={() => act(actions.reanalyse, entry)}
                      >
                        <RefreshCw />
                        {t("history.reanalyse")}
                      </button>
                      <button
                        disabled={busy}
                        className="danger-text"
                        onClick={() => act(actions.delete, entry)}
                      >
                        <Trash2 />
                        {t("history.delete")}
                      </button>
                    </div>
                  </details>
                </div>
                <div className="history-card-bottom">
                  <div className="history-scores">
                    <span className="score-caption">
                      {t("history.finalScores")}
                    </span>
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
                    onClick={() => actions.open(entry)}
                    aria-label={t("history.open", { source })}
                  >
                    <FolderOpen />
                    <span>{t("common.open")}</span>
                  </button>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </section>
  );
}
