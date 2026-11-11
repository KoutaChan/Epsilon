import { Dialog } from "../components/Dialog";
import { errorKey } from "../language/catalog";
import { useLanguage } from "../language/LanguageProvider";

/** 解析結果を消す前の確認のダイアログ。開いている間は、操作の失敗をトーストではなくこのダイアログに出す。 */
export function DeleteDialog({
  busy,
  error,
  shareable,
  onCancel,
  onDelete,
}: {
  busy: boolean;
  error: unknown;
  /** 共有リンクも使えなくなることを書き添えるか。Web で使う。 */
  shareable: boolean;
  onCancel: () => void;
  onDelete: () => void;
}) {
  const { t } = useLanguage();
  return (
    <Dialog
      title={t("history.deleteTitle")}
      onClose={() => !busy && onCancel()}
    >
      <p>{t("history.deleteDescription")}</p>
      {shareable && <p>{t("share.deleteHint")}</p>}
      {error !== null && (
        <p className="error" role="alert">
          {t(errorKey(error))}
        </p>
      )}
      <div className="dialog-actions">
        <button className="button" disabled={busy} onClick={onCancel}>
          {t("common.cancel")}
        </button>
        <button className="button danger" disabled={busy} onClick={onDelete}>
          {t("history.deleteAction")}
        </button>
      </div>
    </Dialog>
  );
}
