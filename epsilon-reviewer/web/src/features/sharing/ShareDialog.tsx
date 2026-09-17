import { useEffect, useRef, useState } from "react";
import { Check, Copy, LoaderCircle } from "lucide-react";
import { api } from "../../api/client";
import type { HistoryEntry, ShareLink } from "../../api/types";
import { Dialog } from "../../components/Dialog";
import { useTask } from "../../hooks/useTask";
import { errorKey } from "../../language/catalog";
import { playerName } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import "./sharing.css";

export function ShareDialog({
  entry,
  onClose,
}: {
  entry: HistoryEntry;
  onClose: () => void;
}) {
  const { t } = useLanguage();
  const { busy, error, run } = useTask();
  const [attempt, setAttempt] = useState(0);
  const [link, setLink] = useState<ShareLink | null>(null);
  const [copied, setCopied] = useState<"copied" | "manual" | null>(null);
  const input = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const controller = new AbortController();
    setLink(null);
    setCopied(null);
    void run(() => api.shareLink(entry.resultId, controller.signal), setLink);
    return () => controller.abort();
  }, [entry.resultId, attempt, run]);

  async function copyLink() {
    const field = input.current!;
    const url = field.value;
    let status: "copied" | "manual" = "manual";
    setCopied(null);
    if (navigator.clipboard) {
      try {
        await navigator.clipboard.writeText(url);
        status = "copied";
      } catch {
        // HTTP接続やブラウザーの権限制限がある場合は手動コピーへ進む。
      }
    }
    if (input.current !== field || field.value !== url) return;
    if (status === "manual") {
      field.focus();
      field.select();
    }
    setCopied(status);
  }

  return (
    <Dialog title={t("share.title")} onClose={onClose}>
      <p className="share-players">
        {entry.metadata.names
          .map((name, seat) => playerName(name, seat, t))
          .join(" · ")}
      </p>
      <p>{t("share.description")}</p>
      {busy && (
        <div className="share-loading" role="status">
          <LoaderCircle className="spin" aria-hidden="true" />
          {t("share.loading")}
        </div>
      )}
      {error !== null && (
        <div className="share-error" role="alert">
          <p>{t(errorKey(error))}</p>
          <button
            className="button"
            disabled={busy}
            onClick={() => setAttempt((value) => value + 1)}
          >
            {t("common.retry")}
          </button>
        </div>
      )}
      {link && (
        <>
          <label className="share-link-field">
            <span>{t("share.url")}</span>
            <input
              ref={input}
              type="url"
              value={link.url}
              readOnly
              spellCheck={false}
              onFocus={(event) => event.currentTarget.select()}
            />
          </label>
          <div className="share-copy-status" role="status">
            {copied &&
              t(copied === "copied" ? "share.copied" : "share.copyManually")}
          </div>
          <div className="dialog-actions">
            <button className="button primary" onClick={() => void copyLink()}>
              {copied === "copied" ? (
                <Check aria-hidden="true" />
              ) : (
                <Copy aria-hidden="true" />
              )}
              {t("share.copy")}
            </button>
          </div>
        </>
      )}
    </Dialog>
  );
}
