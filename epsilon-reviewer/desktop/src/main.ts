import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  net,
  protocol,
  session,
} from "electron";
import { stat } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { writeBytesAtomically, ResultStore } from "./result-store";
import { errorResponse, DesktopError } from "./errors";
import { Languages } from "./language";
import { SettingsStore, parseServerOrigin } from "./settings";
import { RemoteClient } from "./remote-client";
import { ReviewSession, parseDesktopApiRequest } from "./review-session";

const scheme = "epsilon-reviewer";
const profile = process.env.EPSILON_REVIEWER_PROFILE;
if (profile && !path.isAbsolute(profile))
  throw new Error("EPSILON_REVIEWER_PROFILE must be an absolute path.");
app.setName("Epsilon AI Reviewer");
app.setPath(
  "userData",
  profile || path.join(app.getPath("appData"), "epsilon-reviewer"),
);
protocol.registerSchemesAsPrivileged([
  {
    scheme,
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
    },
  },
]);

let window: BrowserWindow | null = null;
let settings: SettingsStore;
let languages: Languages;

function messageInCurrentLanguage(key: string): string {
  return languages.getMessage(settings.value.language, key);
}

function requireTrustedRendererFrame(event: Electron.IpcMainInvokeEvent): void {
  if (
    !window ||
    event.sender !== window.webContents ||
    event.senderFrame !== window.webContents.mainFrame
  )
    throw new DesktopError(
      "invalid_request",
      "IPC is restricted to the application main frame.",
      403,
    );
  const url = new URL(event.senderFrame.url);
  if (url.protocol === scheme + ":" && url.hostname === "app") return;
  const developmentUrl = process.env.EPSILON_REVIEWER_DEV_URL;
  if (developmentUrl && url.origin === new URL(developmentUrl).origin) return;
  throw new DesktopError(
    "invalid_request",
    "IPC sender origin is not requireTrustedRendererFrame.",
    403,
  );
}

function registerAssets(): void {
  const webRoot = app.isPackaged
    ? path.join(process.resourcesPath, "web")
    : path.resolve(__dirname, "../../web/dist");
  protocol.handle(scheme, async (request) => {
    const url = new URL(request.url);
    if (url.hostname !== "app" || request.method !== "GET")
      return new Response(null, { status: 403 });
    let file: string;
    try {
      file = path.resolve(
        webRoot,
        "." +
          decodeURIComponent(
            url.pathname === "/" ? "/index.html" : url.pathname,
          ),
      );
    } catch {
      return new Response(null, { status: 400 });
    }
    if (!file.startsWith(webRoot + path.sep))
      return new Response(null, { status: 403 });
    try {
      if (!(await stat(file)).isFile())
        return new Response(null, { status: 404 });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT")
        return new Response(null, { status: 404 });
      throw error;
    }
    return net.fetch(pathToFileURL(file).toString());
  });
}

async function initialize(): Promise<void> {
  languages = new Languages(
    app.isPackaged
      ? path.join(process.resourcesPath, "language")
      : path.resolve(__dirname, "../../language"),
  );
  const indexDirectory = path.join(app.getPath("userData"), "review");
  settings = await SettingsStore.open(
    path.join(indexDirectory, "settings.json"),
    {
      serverUrl: parseServerOrigin(
        process.env.EPSILON_REVIEWER_SERVER || "http://127.0.0.1:8080",
      ),
      resultsDirectory: path.join(
        app.getPath("documents"),
        "epsilon-reviewer",
        "results",
      ),
      language: languages.selectLanguageForLocale(app.getLocale()),
    },
    languages,
  );
  registerAssets();
  session.defaultSession.setPermissionRequestHandler(
    (_contents, _permission, callback) => callback(false),
  );
  session.defaultSession.setPermissionCheckHandler(() => false);
  const apiSession = session.fromPartition("persist:reviewer-api");
  const review = new ReviewSession(
    settings,
    new ResultStore(indexDirectory),
    new RemoteClient((url, init) => apiSession.fetch(url, init)),
    async (id, bytes) => {
      const choice = await dialog.showSaveDialog(window!, {
        title: messageInCurrentLanguage("desktop.exportTitle"),
        defaultPath: id + ".epsilon-reviewer.json.gz",
        filters: [
          {
            name: messageInCurrentLanguage("desktop.exportFilter"),
            extensions: ["epsilon-reviewer.json.gz"],
          },
        ],
      });
      if (choice.canceled) return { status: 200, body: { cancelled: true } };
      await writeBytesAtomically(choice.filePath!, bytes);
      return { status: 200, body: { saved: true } };
    },
  );
  ipcMain.handle("reviewer:request", async (event, value: unknown) => {
    requireTrustedRendererFrame(event);
    try {
      return await review.request(parseDesktopApiRequest(value));
    } catch (error) {
      console.error("Desktop request failed:", error);
      return errorResponse(error);
    }
  });
  ipcMain.handle("reviewer:settings", (event) => {
    requireTrustedRendererFrame(event);
    return settings.value;
  });
  ipcMain.handle("reviewer:server", (event, value: string) => {
    requireTrustedRendererFrame(event);
    return settings.setServerUrl(value);
  });
  ipcMain.handle("reviewer:language", async (event, value: string) => {
    requireTrustedRendererFrame(event);
    const next = await settings.setLanguage(value);
    window!.setTitle(messageInCurrentLanguage("app.title"));
    return next;
  });
  ipcMain.handle("reviewer:directory", async (event) => {
    requireTrustedRendererFrame(event);
    const selected = await dialog.showOpenDialog(window!, {
      title: messageInCurrentLanguage("desktop.directoryTitle"),
      defaultPath: settings.value.resultsDirectory,
      properties: ["openDirectory", "createDirectory"],
    });
    return selected.canceled
      ? settings.value
      : settings.setResultsDirectory(selected.filePaths[0]!);
  });
  await createWindow();
}

function createWindow(): Promise<void> {
  const developmentUrl = process.env.EPSILON_REVIEWER_DEV_URL;
  if (
    developmentUrl &&
    !/^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?\/?$/.test(developmentUrl)
  )
    throw new Error(
      "EPSILON_REVIEWER_DEV_URL must point to a local Vite server.",
    );
  window = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 360,
    minHeight: 500,
    icon: app.isPackaged
      ? path.join(process.resourcesPath, "icon.png")
      : path.resolve(__dirname, "../../assets/icon.png"),
    title: messageInCurrentLanguage("app.title"),
    autoHideMenuBar: true,
    backgroundColor: "#f5f6fa",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("will-navigate", (event) => event.preventDefault());
  window.on("closed", () => {
    window = null;
  });
  return window.loadURL(developmentUrl || scheme + "://app/");
}

function startupFailed(error: unknown): void {
  console.error("Application startup failed:", error);
  dialog.showErrorBox(
    settings
      ? messageInCurrentLanguage("desktop.startupTitle")
      : "Application startup failed",
    settings
      ? messageInCurrentLanguage("desktop.startupFailure")
      : "The application could not start. See the application log for details.",
  );
  app.quit();
}

if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on("second-instance", () => {
    window?.restore();
    window?.focus();
  });
  app.whenReady().then(initialize).catch(startupFailed);
  app.on("window-all-closed", () => {
    if (process.platform !== "darwin") app.quit();
  });
  app.on("activate", () => {
    if (!window && settings) void createWindow().catch(startupFailed);
  });
}
