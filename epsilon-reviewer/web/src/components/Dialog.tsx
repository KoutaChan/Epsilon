import { useEffect, useRef, type ReactNode } from "react";
import { X } from "lucide-react";
import { useLanguage } from "../language/LanguageProvider";

/** モーダルの窓。開いた時に showModal し、閉じたら前のフォーカスへ戻す。Esc と背景の押下は onClose。 */
export function Dialog({
  title,
  children,
  onClose,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
}) {
  const { t } = useLanguage();
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    if (dialog === null) return;
    const trigger =
      document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
    dialog.showModal();
    return () => {
      dialog.close();
      trigger?.focus();
    };
  }, []);
  return (
    <dialog
      className="dialog"
      ref={ref}
      aria-label={title}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="dialog-body">
        <div className="dialog-heading">
          <h2>{title}</h2>
          <button
            className="icon-button"
            aria-label={t("common.close")}
            onClick={onClose}
          >
            <X />
          </button>
        </div>
        {children}
      </div>
    </dialog>
  );
}
