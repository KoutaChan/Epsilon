import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { languages, translate, type Translate } from "./catalog";

const storageKey = "epsilon-reviewer.language";
const defaultLanguage = "ja";
function storedLanguage() {
  const value = localStorage.getItem(storageKey) ?? defaultLanguage;
  return languages.some((entry) => entry.id === value)
    ? value
    : defaultLanguage;
}
interface LanguageContext {
  language: string;
  locale: string;
  t: Translate;
  setLanguage: (language: string) => Promise<void>;
}
const Context = createContext<LanguageContext>(null!);
export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, updateLanguage] = useState(storedLanguage);
  useEffect(() => {
    if (!window.epsilonReviewer) return;
    let active = true;
    window.epsilonReviewer
      .getSettings()
      .then((settings) => {
        if (active) updateLanguage(settings.language);
      })
      .catch((error: unknown) =>
        console.error("Failed to load the language preference.", error),
      );
    return () => {
      active = false;
    };
  }, []);
  useEffect(() => {
    localStorage.setItem(storageKey, language);
    document.documentElement.lang = language;
    document.title = translate(language)("app.title");
  }, [language]);
  async function setLanguage(value: string) {
    if (window.epsilonReviewer) await window.epsilonReviewer.setLanguage(value);
    updateLanguage(value);
  }
  return (
    <Context
      value={{
        language,
        locale: language,
        t: translate(language),
        setLanguage,
      }}
    >
      {children}
    </Context>
  );
}
export const useLanguage = () => useContext(Context);
