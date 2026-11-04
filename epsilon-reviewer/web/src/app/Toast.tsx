import type { MessageKey } from "../language/catalog";
import { useLanguage } from "../language/LanguageProvider";

export function Toast({ message }: { message: MessageKey | null }) {
  const { t } = useLanguage();
  return (
    message !== null && (
      <div className="toast" role="status">
        {t(message)}
      </div>
    )
  );
}
