import { useCallback, useEffect, useState } from "react";
import { CircleAlert, Info, Settings } from "lucide-react";
import { api, isDesktop } from "./api/client";
import type {
  HistoryEntry,
  ImportedRecord,
  Model,
  ReviewResult,
} from "./api/types";
import { Dialog } from "./components/Dialog";
import { tileAsset } from "./components/Tile";
import { AnalysisProgress } from "./features/analysis/AnalysisProgress";
import { ConfirmAnalysis } from "./features/analysis/ConfirmAnalysis";
import { HistoryList } from "./features/history/HistoryList";
import { useHistory } from "./features/history/useHistory";
import { ImportCard } from "./features/home/ImportCard";
import { useModels } from "./features/models/useModels";
import { ReviewView } from "./features/replay/ReviewView";
import { SettingsDialog } from "./features/settings/SettingsDialog";
import { StorageDialog } from "./features/settings/StorageDialog";
import { ShareDialog } from "./features/sharing/ShareDialog";
import { SharedReview } from "./features/sharing/SharedReview";
import { useTask } from "./hooks/useTask";
import { errorKey, type MessageKey } from "./language/catalog";
import { modelKey } from "./language/format";
import { LanguageSelect } from "./language/LanguageSelect";
import { useLanguage } from "./language/LanguageProvider";

type Screen =
  | { name: "home" }
  | { name: "confirm"; record: ImportedRecord; seat: number }
  | {
      name: "progress";
      record: ImportedRecord;
      seat: number;
      jobId: string;
      model: Model;
      connection: number;
    }
  | { name: "review"; result: ReviewResult; seat: number };
type Modal =
  | { name: "storage" | "settings" }
  | { name: "delete" | "share"; entry: HistoryEntry }
  | null;

export function App() {
  const sharedId = isDesktop
    ? null
    : new URLSearchParams(window.location.search).get("share");
  return sharedId === null ? <OwnerApp /> : <SharedReview id={sharedId} />;
}

function OwnerApp() {
  const { t } = useLanguage();
  const [connection, setConnection] = useState(0);
  const models = useModels(connection);
  const history = useHistory(connection);
  const task = useTask();
  const [screen, setScreen] = useState<Screen>({ name: "home" });
  const [modal, setModal] = useState<Modal>(null);
  const [toast, setToast] = useState<MessageKey | null>(null);
  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 4200);
    return () => clearTimeout(timer);
  }, [toast]);
  const completed = useCallback(
    (result: ReviewResult) => {
      setScreen((current) =>
        current.name === "progress"
          ? { name: "review", result, seat: current.seat }
          : current,
      );
      history.refresh();
    },
    [history.refresh],
  );
  const home = () => {
    setScreen({ name: "home" });
    task.clearError();
  };
  const imported = (record: ImportedRecord) =>
    setScreen({ name: "confirm", record, seat: 0 });
  const connectionChanged = (changed: boolean) => {
    if (changed) {
      task.invalidate();
      models.select("");
      setConnection((value) => value + 1);
      setScreen((current) =>
        current.name === "confirm" ? { name: "home" } : current,
      );
    } else history.refresh();
  };
  if (screen.name === "review")
    return (
      <ReviewView
        key={screen.result.resultId}
        result={screen.result}
        initialSeat={screen.seat}
        shareable={!isDesktop}
        onBack={home}
      />
    );
  return (
    <div className="app-shell">
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
          {isDesktop && (
            <button
              className="icon-button"
              aria-label={t("settings.title")}
              disabled={task.busy}
              onClick={() => setModal({ name: "settings" })}
            >
              <Settings />
            </button>
          )}
          <button
            className="storage-button"
            onClick={() => setModal({ name: "storage" })}
          >
            <Info />
            <span>{t("storage.title")}</span>
          </button>
        </div>
      </header>
      <main className="app-main">
        {task.error !== null && modal?.name !== "delete" && (
          <div className="error-banner" role="alert">
            <CircleAlert />
            <span>{t(errorKey(task.error))}</span>
            <button className="text-button" onClick={task.clearError}>
              {t("common.close")}
            </button>
          </div>
        )}
        {screen.name === "home" && (
          <div className="home-layout">
            <ImportCard
              models={models}
              busy={task.busy}
              onFile={(file) =>
                void task.run(() => api.importFile(file), imported)
              }
              onUrl={(url) => void task.run(() => api.importUrl(url), imported)}
            />
            <HistoryList
              history={history}
              busy={task.busy}
              actions={{
                share: isDesktop
                  ? undefined
                  : (entry) => setModal({ name: "share", entry }),
                open: (entry) =>
                  void task.run(
                    () => api.result(entry.resultId),
                    (result) => setScreen({ name: "review", result, seat: 0 }),
                  ),
                export: (entry) =>
                  void task.run(
                    () => api.export(entry.resultId),
                    (saved) => {
                      if (saved)
                        setToast(
                          isDesktop
                            ? "history.exported"
                            : "history.downloadStarted",
                        );
                    },
                  ),
                reanalyse: (entry) =>
                  void task.run(
                    () => api.reanalyse(entry.resultId),
                    (record) => {
                      imported(record);
                      const key = modelKey(entry.model);
                      models.select(
                        models.state.status === "ready" &&
                          models.state.catalog.models.some(
                            (model) =>
                              modelKey(model) === key &&
                              model.availability === "available",
                          )
                          ? key
                          : "",
                      );
                    },
                  ),
                delete: (entry) => {
                  task.clearError();
                  setModal({ name: "delete", entry });
                },
              }}
            />
          </div>
        )}
        {screen.name === "confirm" && (
          <ConfirmAnalysis
            record={screen.record}
            seat={screen.seat}
            models={models}
            busy={task.busy}
            onSeat={(seat) => setScreen({ ...screen, seat })}
            onBack={home}
            onStart={() => {
              const model = models.selected!;
              void task.run(
                () =>
                  api.start(
                    screen.record.recordId,
                    model.modelId,
                    model.revision,
                  ),
                ({ jobId, model: usedModel }) =>
                  setScreen({
                    name: "progress",
                    record: screen.record,
                    seat: screen.seat,
                    jobId,
                    model: usedModel,
                    connection,
                  }),
              );
            }}
          />
        )}
        {screen.name === "progress" && (
          <AnalysisProgress
            key={screen.jobId}
            jobId={screen.jobId}
            record={screen.record}
            model={screen.model}
            onComplete={completed}
            onBack={() => {
              setScreen(
                screen.connection === connection
                  ? {
                      name: "confirm",
                      record: screen.record,
                      seat: screen.seat,
                    }
                  : { name: "home" },
              );
            }}
          />
        )}
      </main>
      {toast && (
        <div className="toast" role="status">
          {t(toast)}
        </div>
      )}
      {modal?.name === "storage" && (
        <StorageDialog onClose={() => setModal(null)} />
      )}
      {modal?.name === "settings" && (
        <SettingsDialog
          onClose={() => setModal(null)}
          onChanged={connectionChanged}
        />
      )}
      {modal?.name === "share" && (
        <ShareDialog entry={modal.entry} onClose={() => setModal(null)} />
      )}
      {modal?.name === "delete" && (
        <Dialog
          title={t("history.deleteTitle")}
          onClose={() => !task.busy && setModal(null)}
        >
          <p>{t("history.deleteDescription")}</p>
          {!isDesktop && <p>{t("share.deleteHint")}</p>}
          {task.error !== null && (
            <p className="error" role="alert">
              {t(errorKey(task.error))}
            </p>
          )}
          <div className="dialog-actions">
            <button
              className="button"
              disabled={task.busy}
              onClick={() => setModal(null)}
            >
              {t("common.cancel")}
            </button>
            <button
              className="button danger"
              disabled={task.busy}
              onClick={() =>
                void task.run(
                  () => api.remove(modal.entry.resultId),
                  () => {
                    setModal(null);
                    history.refresh();
                    setToast("history.deleted");
                  },
                )
              }
            >
              {t("history.deleteAction")}
            </button>
          </div>
        </Dialog>
      )}
    </div>
  );
}
