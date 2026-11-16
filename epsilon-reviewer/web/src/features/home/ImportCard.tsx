import { useRef, useState } from "react";
import { ArrowRight, Link, LoaderCircle, Upload } from "lucide-react";
import { useLanguage } from "../../language/LanguageProvider";

/** ファイルの選択で選べる牌譜の拡張子（mjai・天鳳の XML・雀魂。gzip で圧縮したものも選べる）。 */
const RECORD_FILE_TYPES =
  ".jsonl,.mjson,.mjai,.gz,.xml,.json,.bin,.dat,.majsoul";

/** 牌譜を取り込むカード。天鳳の URL か、牌譜のファイル（選ぶかドロップする）から取り込む。モデルは、読んだ牌譜の人数で後から選ぶ。 */
export function ImportCard({
  busy,
  onFile,
  onUrl,
}: {
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
          disabled={busy || !url.trim()}
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
          onClick={() => fileInput.current?.click()}
          disabled={busy}
        >
          <Upload />
          {t("home.file")}
        </button>
        <small>{t("home.formats")}</small>
        <input
          ref={fileInput}
          type="file"
          accept={RECORD_FILE_TYPES}
          hidden
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onFile(file);
            event.currentTarget.value = "";
          }}
        />
      </div>
    </section>
  );
}
