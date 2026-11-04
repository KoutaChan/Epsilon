import { Download, MoreHorizontal, RefreshCw, Share2, Trash2 } from "lucide-react";
import { useLanguage } from "../../language/LanguageProvider";

export type MenuAction = "share" | "export" | "reanalyse" | "delete";

/** 項目の「…」の menu。開閉は HistoryList が持つ（開くのは 1 つだけ）。 */
export function HistoryMenu({
  source,
  open,
  busy,
  shareable,
  onToggle,
  onChoose,
}: {
  /** 牌譜の種類の表示（ボタンの名前に使う）。 */
  source: string;
  open: boolean;
  busy: boolean;
  shareable: boolean;
  onToggle: (trigger: HTMLElement) => void;
  onChoose: (action: MenuAction) => void;
}) {
  const { t } = useLanguage();
  return (
    <details className="history-menu" open={open}>
      <summary
        className="icon-button"
        aria-label={t("history.actions", { source })}
        onClick={(event) => {
          event.preventDefault();
          onToggle(event.currentTarget);
        }}
      >
        <MoreHorizontal />
      </summary>
      <div className="floating-menu">
        {shareable && (
          <button disabled={busy} onClick={() => onChoose("share")}>
            <Share2 />
            {t("share.title")}
          </button>
        )}
        <button disabled={busy} onClick={() => onChoose("export")}>
          <Download />
          {t("history.export")}
        </button>
        <button disabled={busy} onClick={() => onChoose("reanalyse")}>
          <RefreshCw />
          {t("history.reanalyse")}
        </button>
        <button
          disabled={busy}
          className="danger-text"
          onClick={() => onChoose("delete")}
        >
          <Trash2 />
          {t("history.delete")}
        </button>
      </div>
    </details>
  );
}
