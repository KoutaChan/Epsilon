import { isDesktop } from "../../api/client";
import { Dialog } from "../../components/Dialog";
import { useLanguage } from "../../language/LanguageProvider";

export function StorageDialog({ onClose }: { onClose: () => void }) {
  const { t } = useLanguage();
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
      <p>{t(isDesktop ? "storage.desktop" : "storage.web")}</p>
      <p>{t(isDesktop ? "storage.desktopTransfer" : "storage.cookie")}</p>
      <p>{t("storage.delete")}</p>
      {!isDesktop && <p>{t("storage.sharing")}</p>}
      <div className="dialog-actions">
        <button className="button" onClick={onClose}>
          {t("common.close")}
        </button>
      </div>
    </Dialog>
  );
}
