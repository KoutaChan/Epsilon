import { createRoot } from "react-dom/client";
import { App } from "./App";
import { LanguageProvider } from "./language/LanguageProvider";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <LanguageProvider>
    <App />
  </LanguageProvider>,
);
