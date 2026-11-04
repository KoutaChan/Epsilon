import { CircleAlert } from "lucide-react";
import { errorKey } from "../language/catalog";
import { useLanguage } from "../language/LanguageProvider";

export function ErrorBanner({
  error,
  onClose,
}: {
  error: unknown;
  onClose: () => void;
}) {
  const { t } = useLanguage();
  return (
    <div className="error-banner" role="alert">
      <CircleAlert />
      <span>{t(errorKey(error))}</span>
      <button className="text-button" onClick={onClose}>
        {t("common.close")}
      </button>
    </div>
  );
}
