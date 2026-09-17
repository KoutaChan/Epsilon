import { ArrowLeft, ArrowRight, Check, LoaderCircle } from "lucide-react";
import type { ImportedRecord } from "../../api/types";
import { playerName, seatName, sourceLabel } from "../../language/format";
import { useLanguage } from "../../language/LanguageProvider";
import { ModelSelect } from "../models/ModelSelect";
import type { Models } from "../models/useModels";

export function ConfirmAnalysis({
  record,
  seat,
  onSeat,
  models,
  busy,
  onBack,
  onStart,
}: {
  record: ImportedRecord;
  seat: number;
  onSeat: (seat: number) => void;
  models: Models;
  busy: boolean;
  onBack: () => void;
  onStart: () => void;
}) {
  const { t } = useLanguage();
  return (
    <section className="content-card confirm-card">
      <button className="text-button" disabled={busy} onClick={onBack}>
        <ArrowLeft />
        {t("common.back")}
      </button>
      <h2>{t("confirm.title")}</h2>
      <p className="record-caption">
        {sourceLabel(record.metadata.source, t)} ·{" "}
        {t("confirm.rounds", { count: record.metadata.roundCount })}
      </p>
      <ModelSelect models={models} id="confirm-model" />
      <fieldset className="player-options">
        <legend>{t("confirm.view")}</legend>
        <div>
          {record.metadata.names.map((name, index) => (
            <button
              type="button"
              key={index}
              aria-pressed={seat === index}
              onClick={() => onSeat(index)}
            >
              <span>{seatName(index, 0, t)}</span>
              <strong>{playerName(name, index, t)}</strong>
              {seat === index && <Check />}
            </button>
          ))}
        </div>
        <p>{t("confirm.allPlayers")}</p>
      </fieldset>
      <div className="confirm-footer">
        <p>{t("confirm.saved")}</p>
        <button
          className="button primary"
          onClick={onStart}
          disabled={busy || !models.selected}
        >
          {busy ? (
            <LoaderCircle className="spin" />
          ) : (
            <>
              {t("home.analyse")}
              <ArrowRight />
            </>
          )}
        </button>
      </div>
    </section>
  );
}
