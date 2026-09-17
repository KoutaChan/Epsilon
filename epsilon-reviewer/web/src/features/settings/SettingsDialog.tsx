import { useEffect, useState } from "react";
import { FolderOpen, Link } from "lucide-react";
import type { DesktopSettings } from "../../api/types";
import { Dialog } from "../../components/Dialog";
import { useTask } from "../../hooks/useTask";
import { errorKey } from "../../language/catalog";
import { useLanguage } from "../../language/LanguageProvider";

export function SettingsDialog({
  onClose,
  onChanged,
}: {
  onClose: () => void;
  onChanged: (serverChanged: boolean) => void;
}) {
  const { t } = useLanguage();
  const [settings, setSettings] = useState<DesktopSettings | null>(null);
  const [serverUrl, setServerUrl] = useState("");
  const { run, busy, error } = useTask();
  useEffect(() => {
    void run(
      () => window.epsilonReviewer!.getSettings(),
      (value) => {
        setSettings(value);
        setServerUrl(value.serverUrl);
      },
    );
  }, [run]);
  return (
    <Dialog title={t("settings.title")} onClose={() => !busy && onClose()}>
      {error !== null && (
        <p className="error" role="alert">
          {t(errorKey(error))}
        </p>
      )}
      {settings ? (
        <>
          <label className="filled-field">
            <Link />
            <span>
              <span className="field-label">{t("settings.server")}</span>
              <input
                type="url"
                value={serverUrl}
                onChange={(event) => setServerUrl(event.target.value)}
                placeholder="http://127.0.0.1:8080"
              />
            </span>
          </label>
          <h3>{t("settings.directory")}</h3>
          <p className="path-copy">{settings.resultsDirectory}</p>
          <button
            className="button"
            disabled={busy}
            onClick={() =>
              void run(
                () => window.epsilonReviewer!.chooseResultsDirectory(),
                (value) => {
                  setSettings(value);
                  onChanged(false);
                },
              )
            }
          >
            <FolderOpen />
            {t("settings.changeDirectory")}
          </button>
          <div className="dialog-actions">
            <button
              className="button primary"
              disabled={busy || !serverUrl.trim()}
              onClick={() =>
                void run(
                  () => window.epsilonReviewer!.setServerUrl(serverUrl),
                  (value) => {
                    onChanged(value.serverUrl !== settings.serverUrl);
                    onClose();
                  },
                )
              }
            >
              {t("common.save")}
            </button>
          </div>
        </>
      ) : (
        <p role="status">{t("settings.loading")}</p>
      )}
    </Dialog>
  );
}
