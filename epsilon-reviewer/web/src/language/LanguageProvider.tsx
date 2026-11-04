import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { api } from "../api/platform";
import { isLanguage, translator, type Translate } from "./catalog";

const STORAGE_KEY = "epsilon-reviewer.language";
const DEFAULT_LANGUAGE = "ja";

export interface LanguageState {
  language: string;
  /** Intl の書式の言語（言語の id と同じ）。 */
  locale: string;
  t: Translate;
  /** デスクトップは設定に保存してから切り替える（失敗は投げる）。 */
  setLanguage(id: string): Promise<void>;
}

const Context = createContext<LanguageState | null>(null);

function storedLanguage(): string {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    if (value !== null && isLanguage(value)) return value;
  } catch {
    // ストレージを使えない窓（プライベートの窓など）は既定の言語で動く。
  }
  return DEFAULT_LANGUAGE;
}

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setCurrent] = useState(storedLanguage);
  useEffect(() => {
    const desktop = api.desktop;
    if (desktop === null) return;
    let active = true;
    desktop.settings().then(
      (settings) => {
        if (active && isLanguage(settings.language))
          setCurrent(settings.language);
      },
      (error: unknown) =>
        console.error("Could not read the desktop language setting.", error),
    );
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, language);
    } catch {
      // 覚えられなくても今の画面の言語は変わる。
    }
    document.documentElement.lang = language;
    document.title = translator(language)("app.title");
  }, [language]);
  const setLanguage = useCallback(async (id: string) => {
    if (api.desktop !== null) await api.desktop.setLanguage(id);
    setCurrent(id);
  }, []);
  const value = useMemo(
    () => ({ language, locale: language, t: translator(language), setLanguage }),
    [language, setLanguage],
  );
  return <Context value={value}>{children}</Context>;
}

export function useLanguage(): LanguageState {
  const value = useContext(Context);
  if (value === null)
    throw new Error("useLanguage must be used inside LanguageProvider.");
  return value;
}
