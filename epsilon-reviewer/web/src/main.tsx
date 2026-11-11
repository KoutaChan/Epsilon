import { createRoot } from "react-dom/client";
import { App } from "./App";
import { LanguageProvider } from "./language/LanguageProvider";
// 共通のスタイルは、画面の部品の CSS（sharing.css など）より後に読み込む。詳細度が同じルール（.dialog-body p と .share-error p）では、後に読み込んだ方が優先される。
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <LanguageProvider>
    <App />
  </LanguageProvider>,
);
