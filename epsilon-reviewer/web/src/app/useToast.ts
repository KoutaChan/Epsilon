import { useEffect, useState } from "react";
import type { MessageKey } from "../language/catalog";

const TOAST_DURATION_MS = 4200;

export interface Toasts {
  message: MessageKey | null;
  show(message: MessageKey): void;
}

export function useToast(): Toasts {
  const [message, setMessage] = useState<MessageKey | null>(null);
  useEffect(() => {
    if (message === null) return;
    const timer = setTimeout(() => setMessage(null), TOAST_DURATION_MS);
    return () => clearTimeout(timer);
  }, [message]);
  return { message, show: setMessage };
}
