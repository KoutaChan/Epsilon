import path from "node:path";
import { app, dialog, ipcMain, type BrowserWindow } from "electron";
import { resultFileName } from "../../protocol/result.ts";
import {
  APP_URL,
  registerAppProtocol,
  registerAppScheme,
} from "./appProtocol.ts";
import { registerBridge } from "./bridge.ts";
import { EngineProcess, engineLaunch } from "./engine/process.ts";
import { Languages, type MessageKey } from "./language.ts";
import { appPaths } from "./paths.ts";
import { SettingsStore } from "./settings.ts";
import { errorMessage } from "./values.ts";
import {
  createMainWindow,
  denyPermissions,
  developmentUrl,
  untrustedSender,
} from "./window.ts";

const APP_NAME = "Epsilon AI Reviewer";

app.setName(APP_NAME);
const profile = process.env.EPSILON_REVIEWER_PROFILE;
const profileValid = !profile || path.isAbsolute(profile);
if (profileValid)
  app.setPath(
    "userData",
    profile || path.join(app.getPath("appData"), "epsilon-reviewer"),
  );
registerAppScheme();

/** 起動に失敗したときのダイアログの文言を引く関数。言語を読む前に失敗したら null のままで、英語のエラーメッセージをそのまま出す。 */
let message: ((key: MessageKey) => string) | null = null;
let engine: EngineProcess | null = null;
let window: BrowserWindow | null = null;
let openWindow: (() => Promise<void>) | null = null;
let engineStopped = false;

async function start(): Promise<void> {
  const paths = appPaths(
    app.isPackaged,
    process.resourcesPath,
    app.getAppPath(),
    app.getPath("userData"),
  );
  const languages = Languages.load(paths.language);
  const locale = languages.select(app.getLocale());
  message = (key) => languages.message(locale, key);
  if (!profileValid)
    throw new Error("EPSILON_REVIEWER_PROFILE must be an absolute path.");
  const development = developmentUrl(process.env.EPSILON_REVIEWER_DEV_URL);
  const settings = await SettingsStore.open(
    path.join(paths.userData, "settings.json"),
    {
      resultsDirectory: path.join(app.getPath("documents"), "epsilon-reviewer"),
      language: locale,
    },
    (language) => languages.has(language),
  );
  const text = (key: MessageKey): string =>
    languages.message(settings.value.language, key);
  message = text;
  // Java はウィンドウと並行して起動する。JVM の起動と保存先・カタログの読み込みを、ウィンドウの読み込みと同時に進めるため。
  const engineProcess = new EngineProcess(() =>
    engineLaunch(paths.engine, paths.userData, settings.value.resultsDirectory),
  );
  engine = engineProcess;
  engineProcess.start();
  registerAppProtocol(paths.web);
  denyPermissions();
  registerBridge(ipcMain, {
    engine: engineProcess,
    settings,
    languages,
    host: {
      checkSender: (event) => untrustedSender(window, event, development),
      chooseDirectory: async (current) => {
        const choice = await dialog.showOpenDialog(mainWindow(), {
          title: text("desktop.directoryTitle"),
          defaultPath: current,
          properties: ["openDirectory", "createDirectory"],
        });
        return choice.canceled ? null : (choice.filePaths[0] ?? null);
      },
      chooseExportPath: async (resultId) => {
        const choice = await dialog.showSaveDialog(mainWindow(), {
          title: text("desktop.exportTitle"),
          defaultPath: resultFileName(resultId),
          filters: [
            {
              name: text("desktop.exportFilter"),
              extensions: ["epsilon-reviewer.json.gz"],
            },
          ],
        });
        return choice.canceled || choice.filePath === ""
          ? null
          : choice.filePath;
      },
      setTitle: (title) => window?.setTitle(title),
    },
  });
  openWindow = async () => {
    const created = createMainWindow({
      title: text("app.title"),
      icon: paths.icon,
      preload: paths.preload,
    });
    window = created;
    created.on("closed", () => {
      window = null;
    });
    await created.loadURL(development ?? APP_URL);
  };
  await openWindow();
}

function mainWindow(): BrowserWindow {
  if (window === null) throw new Error("The main window is closed.");
  return window;
}

function startupFailed(error: unknown): void {
  console.error("Application startup failed:", error);
  if (message !== null)
    dialog.showErrorBox(
      message("desktop.startupTitle"),
      message("desktop.startupFailure"),
    );
  else dialog.showErrorBox(APP_NAME, errorMessage(error));
  app.quit();
}

if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on("second-instance", () => {
    window?.restore();
    window?.focus();
  });
  app.whenReady().then(start).catch(startupFailed);
  app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
  });
  app.on("activate", () => {
    if (window === null && openWindow !== null)
      openWindow().catch(startupFailed);
  });
  // アプリを終える前に Java を止める。stdin を閉じて 5 秒待ち、終わらなければ kill する。
  app.on("will-quit", (event) => {
    if (engine === null || engineStopped) return;
    event.preventDefault();
    void engine.stop().finally(() => {
      engineStopped = true;
      app.quit();
    });
  });
}
