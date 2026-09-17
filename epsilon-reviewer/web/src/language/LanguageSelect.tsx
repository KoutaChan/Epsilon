import { Languages } from "lucide-react";
import { useState } from "react";
import { languages, errorKey } from "./catalog";
import { useLanguage } from "./LanguageProvider";

export function LanguageSelect() {
  const { language, setLanguage, t } = useLanguage();
  const [error, setError] = useState<unknown>(null);
  return (
    <div className="language-control">
      <label className="language-select">
        <Languages aria-hidden="true" />
        <select
          value={language}
          aria-label={t("language.label")}
          onChange={(event) => {
            setError(null);
            void setLanguage(event.target.value).catch(setError);
          }}
        >
          {languages.map((entry) => (
            <option key={entry.id} value={entry.id}>
              {entry.name}
            </option>
          ))}
        </select>
      </label>
      {error !== null && <span role="alert">{t(errorKey(error))}</span>}
    </div>
  );
}
