import { useEffect, useMemo, useRef, useState } from "react";
import { CircleAlert, FolderOpen, LoaderCircle, RefreshCw, Search } from "lucide-react";
import type {
  HistoryEntry,
  HistoryList as HistoryData,
} from "../../../../protocol/model.ts";
import { api } from "../../api/platform";
import type { Resource } from "../../hooks/useResource";
import { errorKey } from "../../language/catalog";
import { modelLabel, sourceLabel } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import { HistoryCard } from "./HistoryCard";
import type { MenuAction } from "./HistoryMenu";

export interface HistoryActions {
  /** 共有の窓を開く（Web）。デスクトップは null。 */
  share: ((entry: HistoryEntry) => void) | null;
  open(entry: HistoryEntry): void;
  export(entry: HistoryEntry): void;
  reanalyse(entry: HistoryEntry): void;
  delete(entry: HistoryEntry): void;
}

/** 履歴。検索は牌譜の種類（元の値と表示）・対局者名・モデルの表示を UI の言語で小文字にして探す。 */
export function HistoryList({
  history,
  busy,
  actions,
}: {
  history: Resource<HistoryData>;
  busy: boolean;
  actions: HistoryActions;
}) {
  const { t, locale } = useLanguage();
  const [search, setSearch] = useState("");
  const [menu, setMenu] = useState<string | null>(null);
  const trigger = useRef<HTMLElement | null>(null);
  const { state, reload } = history;
  const entries = state.status === "ready" ? state.value.results : null;
  const searchable = useMemo(
    () =>
      (entries ?? []).map((entry) => ({
        entry,
        text: `${entry.record.source} ${sourceLabel(entry.record.source, t)} ${entry.record.names.join(" ")} ${modelLabel(entry.model)}`.toLocaleLowerCase(
          locale,
        ),
      })),
    [entries, t, locale],
  );
  useEffect(() => {
    if (menu === null) return;
    const closeOutside = (event: PointerEvent) => {
      if (
        !(event.target instanceof Element) ||
        !event.target.closest(".history-menu")
      )
        setMenu(null);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setMenu(null);
      trigger.current?.focus();
    };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOutside);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [menu]);
  const query = search.toLocaleLowerCase(locale);
  const shown = searchable
    .filter((item) => item.text.includes(query))
    .map((item) => item.entry);
  const choose = (entry: HistoryEntry, action: MenuAction) => {
    setMenu(null);
    trigger.current?.focus();
    actions[action]?.(entry);
  };
  return (
    <section className="history-section">
      <div className="history-heading">
        <div>
          <h2>
            {t("history.title")}{" "}
            <span>{entries ? entries.length : ""}</span>
          </h2>
          <p>
            {t(api.platform === "desktop" ? "history.local" : "history.browser")}
          </p>
        </div>
        <button
          className="icon-button"
          aria-label={t("history.refresh")}
          disabled={state.status === "loading"}
          onClick={reload}
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
          <button className="button" onClick={reload}>
            {t("common.retry")}
          </button>
        </div>
      ) : shown.length === 0 ? (
        <div className="empty-state">
          <FolderOpen />
          <h3>{t(search ? "history.noMatches" : "history.empty")}</h3>
          <p>{t(search ? "history.searchHint" : "history.emptyHint")}</p>
        </div>
      ) : (
        <div className="history-list">
          {shown.map((entry) => (
            <HistoryCard
              key={entry.resultId}
              entry={entry}
              busy={busy}
              menuOpen={menu === entry.resultId}
              shareable={actions.share !== null}
              onToggleMenu={(element) => {
                trigger.current = element;
                setMenu(menu === entry.resultId ? null : entry.resultId);
              }}
              onChoose={(action) => choose(entry, action)}
              onOpen={() => actions.open(entry)}
            />
          ))}
        </div>
      )}
    </section>
  );
}
