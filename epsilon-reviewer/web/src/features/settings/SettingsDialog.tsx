import { useEffect, useState } from "react";
import { FolderOpen } from "lucide-react";
import type { DesktopSettings } from "../../../../protocol/bridge.ts";
import type { DesktopSettingsAccess } from "../../api/client";
import { Dialog } from "../../components/Dialog";
import { useTask } from "../../hooks/useTask";
import { errorKey } from "../../language/catalog";
import { useLanguage } from "../../language/LanguageProvider";

/** デスクトップの保存先。変えるとすぐ保存され、解析の側が新しいフォルダーで開き直す。 */
export function SettingsDialog({
  desktop,
  onClose,
  onFolderChanged,
}: {
  desktop: DesktopSettingsAccess;
  onClose: () => void;
  onFolderChanged: () => void;
}) {
  const { t } = useLanguage();
  const [settings, setSettings] = useState<DesktopSettings | null>(null);
  const { run, busy, error } = useTask();
  useEffect(() => {
    void run(() => desktop.settings(), setSettings);
  }, [run, desktop]);
  return (
    <Dialog title={t("settings.title")} onClose={() => !busy && onClose()}>
      {error !== null && (
        <p className="error" role="alert">
          {t(errorKey(error))}
        </p>
      )}
      {settings ? (
        <>
          <h3>{t("settings.directory")}</h3>
          <p className="path-copy">{settings.resultsDirectory}</p>
          <button
            className="button"
            disabled={busy}
            onClick={() =>
              void run(
                () => desktop.chooseResultsDirectory(),
                (value) => {
                  setSettings(value);
                  if (value.resultsDirectory !== settings.resultsDirectory)
                    onFolderChanged();
                },
              )
            }
          >
            <FolderOpen />
            {t("settings.changeDirectory")}
          </button>
        </>
      ) : (
        <p role="status">{t("settings.loading")}</p>
      )}
    </Dialog>
  );
}
