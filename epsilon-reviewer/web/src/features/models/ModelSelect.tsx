import { ChevronDown, Cpu, RefreshCw } from "lucide-react";
import { errorKey } from "../../language/catalog";
import { modelKey, modelLabel } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import type { CatalogModel } from "@epsilon-reviewer/protocol/model";
import type { Models } from "./useModels";

/**
 * その人数（seats）の牌譜を解析するモデルを選ぶ欄。使えないモデルと、その人数の牌譜を読めないモデルは選べない。onlySeats なら、
 * その人数のモデルだけを並べる。
 */
export function ModelSelect({
  models,
  id,
  seats,
  label,
  onlySeats = false,
}: {
  models: Models;
  id: string;
  seats: number;
  label?: string;
  onlySeats?: boolean;
}) {
  const { t } = useLanguage();
  const { state, select, reload } = models;
  const selected = models.selected(seats);
  const usable = (model: CatalogModel) =>
    model.available && model.seats === seats;
  return (
    <div className="model-control">
      <label className="filled-field">
        <Cpu aria-hidden="true" />
        <span>
          <span className="field-label">{label ?? t("model.label")}</span>
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
              state.value.models
                .filter((model) => !onlySeats || model.seats === seats)
                .map((model) => (
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
        ) : (
          ""
        )}
      </div>
    </div>
  );
}
