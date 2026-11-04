import { createRoot } from "react-dom/client";
import { App } from "./App";
import { LanguageProvider } from "./language/LanguageProvider";
// 共通の規則は画面の部品の CSS（sharing.css）の後に置く。同じ強さの規則（.dialog-body p と .share-error p）は後が勝つ。
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <LanguageProvider>
    <App />
  </LanguageProvider>,
);
