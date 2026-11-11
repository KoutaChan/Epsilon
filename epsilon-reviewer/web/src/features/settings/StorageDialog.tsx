import { api } from "../../api/platform";
import { Dialog } from "../../components/Dialog";
import { useLanguage } from "../../language/LanguageProvider";

/** 保存する情報の説明。保存先と送信するものの説明文は、Web 版とデスクトップ版で違う。 */
export function StorageDialog({ onClose }: { onClose: () => void }) {
  const { t } = useLanguage();
  const desktop = api.platform === "desktop";
  return (
    <Dialog title={t("storage.title")} onClose={onClose}>
      <dl className="stored-fields">
        <div>
          <dt>{t("storage.game")}</dt>
          <dd>{t("storage.gameDescription")}</dd>
        </div>
        <div>
          <dt>{t("storage.analysis")}</dt>
          <dd>{t("storage.analysisDescription")}</dd>
        </div>
      </dl>
      <h3>{t("storage.location")}</h3>
      <p>{t(desktop ? "storage.desktop" : "storage.web")}</p>
      <p>{t(desktop ? "storage.desktopTransfer" : "storage.cookie")}</p>
      <p>{t("storage.delete")}</p>
      {!desktop && <p>{t("storage.sharing")}</p>}
      <div className="dialog-actions">
        <button className="button" onClick={onClose}>
          {t("common.close")}
        </button>
      </div>
    </Dialog>
  );
}
