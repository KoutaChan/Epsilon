import { ChevronDown, Cpu, RefreshCw } from "lucide-react";
import { errorKey } from "../../language/catalog";
import { modelKey, modelLabel } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import type { Models } from "./useModels";

export function ModelSelect({ models, id }: { models: Models; id: string }) {
  const { t } = useLanguage();
  const { state, selected, selectedKey, select, refresh } = models;
  return (
    <div className="model-control">
      <label className="filled-field">
        <Cpu aria-hidden="true" />
        <span>
          <span className="field-label">{t("model.label")}</span>
          <select
            id={id}
            value={selectedKey}
            disabled={state.status !== "ready"}
            onChange={(event) => select(event.target.value)}
            aria-describedby={`${id}-status`}
          >
            <option value="">
              {t(state.status === "loading" ? "model.loading" : "model.select")}
            </option>
            {state.status === "ready" &&
              state.catalog.models.map((model) => (
                <option
                  key={modelKey(model)}
                  value={modelKey(model)}
                  disabled={model.availability !== "available"}
                >
                  {modelLabel(model)}
                  {model.availability !== "available"
                    ? ` — ${t("model.unavailable")}`
                    : ""}
                </option>
              ))}
          </select>
        </span>
        <ChevronDown className="field-chevron" aria-hidden="true" />
      </label>
      <div
        id={`${id}-status`}
        className={`model-feedback ${state.status === "error" ? "error" : ""}`}
        role="status"
      >
        {state.status === "error" ? (
          <>
            {t(errorKey(state.error))}
            <button className="text-button" onClick={refresh}>
              <RefreshCw />
              {t("common.retry")}
            </button>
          </>
        ) : state.status === "ready" &&
          !state.catalog.models.some(
            (model) => model.availability === "available",
          ) ? (
          t("model.empty")
        ) : state.status === "ready" && !selected ? (
          t("model.hint")
        ) : (
          ""
        )}
      </div>
    </div>
  );
}
