import { useRef, useState } from "react";
import { Check, Copy, LoaderCircle } from "lucide-react";
import { api } from "../../api/platform";
import { Dialog } from "../../components/Dialog";
import { useResource } from "../../hooks/useResource";
import { errorKey } from "../../language/catalog";
import { playerName } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import "./sharing.css";

/** 共有リンクの窓。写せなければ欄を選んで手で写してもらう。 */
export function ShareDialog({
  resultId,
  names,
  onClose,
}: {
  resultId: string;
  /** 4 席の対局者名。 */
  names: readonly string[];
  onClose: () => void;
}) {
  const { t } = useLanguage();
  const { state, reload } = useResource(
    (signal) => api.shareLink(resultId, signal),
    [resultId],
  );
  const [copied, setCopied] = useState<"copied" | "manual" | null>(null);
  const input = useRef<HTMLInputElement>(null);

  async function copyLink() {
    const field = input.current;
    if (field === null) return;
    const url = field.value;
    let status: "copied" | "manual" = "manual";
    setCopied(null);
    if (navigator.clipboard) {
      try {
        await navigator.clipboard.writeText(url);
        status = "copied";
      } catch {
        // HTTP の接続やブラウザーの権限で写せなければ手で写してもらう。
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
        {names
          .map((name, seat) => playerName(name, seat, t))
          .join(t("common.listSeparator"))}
      </p>
      <p>{t("share.description")}</p>
      {state.status === "loading" && (
        <div className="share-loading" role="status">
          <LoaderCircle className="spin" aria-hidden="true" />
          {t("share.loading")}
        </div>
      )}
      {state.status === "error" && (
        <div className="share-error" role="alert">
          <p>{t(errorKey(state.error))}</p>
          <button className="button" onClick={reload}>
            {t("common.retry")}
          </button>
        </div>
      )}
      {state.status === "ready" && (
        <>
          <label className="share-link-field">
            <span>{t("share.url")}</span>
            <input
              ref={input}
              type="url"
              value={state.value.url}
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
