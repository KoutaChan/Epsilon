import { Info, Settings } from "lucide-react";
import { tileAsset } from "../components/Tile";
import { LanguageSelect } from "../language/LanguageSelect";
import { useLanguage } from "../language/LanguageProvider";

/** 頭の帯。onSettings はデスクトップだけ（Web は null で設定のボタンを出さない）。 */
export function AppHeader({
  busy,
  onSettings,
  onStorage,
}: {
  busy: boolean;
  onSettings: (() => void) | null;
  onStorage: () => void;
}) {
  const { t } = useLanguage();
  return (
    <header className="app-header">
      <div className="brand">
        <img src={tileAsset("F")} alt="" width="34" height="46" />
        <h1 aria-label={t("app.title")}>
          <span>{t("app.brand")}</span>
          <span>{t("app.reviewer")}</span>
        </h1>
      </div>
      <div className="header-actions">
        <LanguageSelect />
        {onSettings && (
          <button
            className="icon-button"
            aria-label={t("settings.title")}
            disabled={busy}
            onClick={onSettings}
          >
            <Settings />
          </button>
        )}
        <button className="storage-button" onClick={onStorage}>
          <Info />
          <span>{t("storage.title")}</span>
        </button>
      </div>
    </header>
  );
}
