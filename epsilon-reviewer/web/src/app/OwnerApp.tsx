import { useCallback, useMemo, useReducer, type Dispatch } from "react";
import type { ImportedRecord } from "../../../protocol/model.ts";
import type { ResultDocument } from "../../../protocol/result.ts";
import type { ExportOutcome } from "../api/client";
import { api } from "../api/platform";
import { AnalysisProgress } from "../features/analysis/AnalysisProgress";
import { ConfirmAnalysis } from "../features/analysis/ConfirmAnalysis";
import { HistoryList } from "../features/history/HistoryList";
import { useHistory } from "../features/history/useHistory";
import { ImportCard } from "../features/home/ImportCard";
import { useModels, type Models } from "../features/models/useModels";
import { ReviewView } from "../features/replay/ReviewView";
import { SettingsDialog } from "../features/settings/SettingsDialog";
import { StorageDialog } from "../features/settings/StorageDialog";
import { ShareDialog } from "../features/sharing/ShareDialog";
import { useTask } from "../hooks/useTask";
import type { MessageKey } from "../language/catalog";
import { modelKey } from "../language/format";
import { AppHeader } from "./AppHeader";
import { DeleteDialog } from "./DeleteDialog";
import { ErrorBanner } from "./ErrorBanner";
import { INITIAL_NAVIGATION, navigate, type NavigationAction } from "./navigation";
import { Toast } from "./Toast";
import { useToast } from "./useToast";

const EXPORT_TOASTS: Record<ExportOutcome, MessageKey | null> = {
  downloaded: "history.downloadStarted",
  saved: "history.exported",
  cancelled: null,
};

/**
 * 持ち主の画面（Web の Cookie の持ち主とデスクトップ）。取り込み・開く・書き出し・解析し直し・解析の開始・削除は
 * 1 つの useTask を使い、後から始めた操作が前の操作の結果を捨てる。
 */
export function OwnerApp() {
  const models = useModels();
  const history = useHistory();
  const task = useTask();
  const toast = useToast();
  const [{ screen, modal }, dispatch] = useReducer(navigate, INITIAL_NAVIGATION);
  const web = api.platform === "web";
  const { reload: reloadHistory } = history;
  const { clearError } = task;
  const completed = useCallback(
    (document: ResultDocument) => {
      dispatch({ type: "completed", document });
      reloadHistory();
    },
    [reloadHistory],
  );
  const home = useCallback(() => {
    dispatch({ type: "back" });
    clearError();
  }, [clearError]);
  const imported = (record: ImportedRecord) =>
    dispatch({ type: "imported", record });
  const closeModal = () => dispatch({ type: "modalClosed" });

  if (screen.name === "review")
    return (
      <OwnerReview
        document={screen.document}
        seat={screen.seat}
        sharing={modal?.name === "share"}
        models={models}
        dispatch={dispatch}
        onBack={home}
      />
    );
  return (
    <div className="app-shell">
      <AppHeader
        busy={task.busy}
        onSettings={
          api.desktop
            ? () => dispatch({ type: "modalOpened", modal: { name: "settings" } })
            : null
        }
        onStorage={() =>
          dispatch({ type: "modalOpened", modal: { name: "storage" } })
        }
      />
      <main className="app-main">
        {task.error !== null && modal?.name !== "delete" && (
          <ErrorBanner error={task.error} onClose={clearError} />
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
                share: web
                  ? (entry) =>
                      dispatch({
                        type: "modalOpened",
                        modal: { name: "share", target: entry },
                      })
                  : null,
                open: (entry) =>
                  void task.run(
                    () => api.result(entry.resultId),
                    (document) => dispatch({ type: "opened", document }),
                  ),
                export: (entry) =>
                  void task.run(
                    () => api.export(entry.resultId),
                    (outcome) => {
                      const message = EXPORT_TOASTS[outcome];
                      if (message !== null) toast.show(message);
                    },
                  ),
                reanalyse: (entry) =>
                  void task.run(
                    () => api.reanalyse(entry.resultId),
                    (record) => {
                      imported(record);
                      models.select(modelKey(entry.model));
                    },
                  ),
                delete: (entry) => {
                  clearError();
                  dispatch({
                    type: "modalOpened",
                    modal: { name: "delete", entry },
                  });
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
            onSeat={(seat) => dispatch({ type: "seatChosen", seat })}
            onBack={home}
            onStart={() => {
              const model = models.selected;
              if (model === undefined) return;
              void task.run(
                () => api.start(screen.record.recordId, model),
                (started) => dispatch({ type: "started", started }),
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
            onBack={() => dispatch({ type: "back" })}
          />
        )}
      </main>
      <Toast message={toast.message} />
      {modal?.name === "storage" && <StorageDialog onClose={closeModal} />}
      {modal?.name === "settings" && api.desktop && (
        <SettingsDialog
          desktop={api.desktop}
          onClose={closeModal}
          onFolderChanged={() => {
            reloadHistory();
            dispatch({ type: "folderChanged" });
          }}
        />
      )}
      {modal?.name === "share" && (
        <ShareDialog
          resultId={modal.target.resultId}
          names={modal.target.record.names}
          onClose={closeModal}
        />
      )}
      {modal?.name === "delete" && (
        <DeleteDialog
          busy={task.busy}
          error={task.error}
          shareable={web}
          onCancel={closeModal}
          onDelete={() =>
            void task.run(
              () => api.remove(modal.entry.resultId),
              () => {
                closeModal();
                reloadHistory();
                toast.show("history.deleted");
              },
            )
          }
        />
      )}
    </div>
  );
}

/** 解析結果の画面。もしもは結果のモデルが今のカタログで使える時だけ、共有は Web だけ。 */
function OwnerReview({
  document,
  seat,
  sharing,
  models,
  dispatch,
  onBack,
}: {
  document: ResultDocument;
  seat: number;
  sharing: boolean;
  models: Models;
  dispatch: Dispatch<NavigationAction>;
  onBack: () => void;
}) {
  const { resultId } = document;
  const available = models.isAvailable(document.model);
  const scenario = useMemo(
    () => (available ? api.scenarioClient(resultId, "owner") : null),
    [available, resultId],
  );
  const share = useCallback(
    () =>
      dispatch({
        type: "modalOpened",
        modal: { name: "share", target: document },
      }),
    [dispatch, document],
  );
  return (
    <>
      <ReviewView
        key={resultId}
        document={document}
        initialSeat={seat}
        scenario={scenario}
        onShare={api.platform === "web" ? share : null}
        back="history"
        onBack={onBack}
      />
      {sharing && (
        <ShareDialog
          resultId={resultId}
          names={document.record.names}
          onClose={() => dispatch({ type: "modalClosed" })}
        />
      )}
    </>
  );
}
