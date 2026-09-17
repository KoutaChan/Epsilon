import { useRef, useState } from "react";
import { ArrowRight, Link, LoaderCircle, Upload } from "lucide-react";
import { ModelSelect } from "../models/ModelSelect";
import type { Models } from "../models/useModels";
import { useLanguage } from "../../language/LanguageProvider";

export function ImportCard({
  models,
  busy,
  onFile,
  onUrl,
}: {
  models: Models;
  busy: boolean;
  onFile: (file: File) => void;
  onUrl: (url: string) => void;
}) {
  const { t } = useLanguage();
  const [url, setUrl] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  return (
    <section
      className={`import-card ${dragging ? "dragging" : ""}`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        const file = event.dataTransfer.files[0];
        if (!busy && file) onFile(file);
      }}
    >
      <h2>{t("home.title")}</h2>
      <p className="section-copy">{t("home.description")}</p>
      <form
        onSubmit={(event) => {
          event.preventDefault();
          onUrl(url.trim());
        }}
      >
        <label className="filled-field">
          <Link aria-hidden="true" />
          <span>
            <span className="field-label">{t("home.url")}</span>
            <input
              type="url"
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              placeholder={t("home.urlPlaceholder")}
              autoComplete="off"
              required
            />
          </span>
        </label>
        <button
          className="button primary import-submit"
          disabled={busy || !url.trim() || !models.selected}
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
      </form>
      <div className="file-row">
        <button
          className="button secondary"
          onClick={() => fileInput.current!.click()}
          disabled={busy}
        >
          <Upload />
          {t("home.file")}
        </button>
        <small>{t("home.formats")}</small>
        <input
          ref={fileInput}
          type="file"
          accept=".jsonl,.mjson,.mjai,.gz,.xml,.json,.bin,.dat,.majsoul"
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onFile(file);
            event.currentTarget.value = "";
          }}
        />
      </div>
      <ModelSelect models={models} id="home-model" />
    </section>
  );
}
