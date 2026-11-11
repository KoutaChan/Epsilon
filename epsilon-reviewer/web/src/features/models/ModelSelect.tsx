import { ChevronDown, Cpu, RefreshCw } from "lucide-react";
import { errorKey } from "../../language/catalog";
import { modelKey, modelLabel } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import type { CatalogModel } from "../../../../protocol/model.ts";
import type { Models } from "./useModels";

/** モデルを選ぶ欄。使えないモデルは選べない。人数（seats）を渡すと、その人数の牌譜を読めないモデルも選べない。 */
export function ModelSelect({
  models,
  id,
  seats,
}: {
  models: Models;
  id: string;
  seats?: number;
}) {
  const { t } = useLanguage();
  const { state, selected, select, reload } = models;
  const usable = (model: CatalogModel) =>
    model.available && (seats === undefined || model.seats === seats);
  return (
    <div className="model-control">
      <label className="filled-field">
        <Cpu aria-hidden="true" />
        <span>
          <span className="field-label">{t("model.label")}</span>
          <select
            id={id}
            value={selected ? modelKey(selected) : ""}
            disabled={state.status !== "ready"}
            onChange={(event) => select(event.target.value)}
            aria-describedby={`${id}-status`}
          >
            <option value="">
              {t(state.status === "loading" ? "model.loading" : "model.select")}
            </option>
            {state.status === "ready" &&
              state.value.models.map((model) => (
                <option
                  key={modelKey(model)}
                  value={modelKey(model)}
                  disabled={!usable(model)}
                >
                  {usable(model)
                    ? modelLabel(model)
                    : t("model.unavailableOption", {
                        model: modelLabel(model),
                      })}
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
            <button className="text-button" onClick={reload}>
              <RefreshCw />
              {t("common.retry")}
            </button>
          </>
        ) : state.status === "ready" && !state.value.models.some(usable) ? (
          t("model.empty")
        ) : state.status === "ready" && selected && !usable(selected) ? (
          t("model.otherSeats")
        ) : state.status === "ready" && !selected ? (
          t("model.hint")
        ) : (
          ""
        )}
      </div>
    </div>
  );
}
