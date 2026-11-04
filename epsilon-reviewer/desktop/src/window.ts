import { BrowserWindow, session, type IpcMainInvokeEvent } from "electron";
import type { ErrorBody } from "../../protocol/errors.ts";
import { APP_HOST, APP_SCHEME } from "./appProtocol.ts";

export interface WindowOptions {
  title: string;
  icon: string;
  preload: string;
}

/** 主の窓。新しい窓と移動は拒む。 */
export function createMainWindow({
  title,
  icon,
  preload,
}: WindowOptions): BrowserWindow {
  const window = new BrowserWindow({
    width: 1440,
    height: 960,
    minWidth: 360,
    minHeight: 500,
    icon,
    title,
    autoHideMenuBar: true,
    backgroundColor: "#f5f6fa",
    webPreferences: {
      preload,
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  window.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  window.webContents.on("will-navigate", (event) => event.preventDefault());
  return window;
}

/** 既定のセッションの権限の求めと確かめを全て断る。 */
export function denyPermissions(): void {
  session.defaultSession.setPermissionRequestHandler(
    (_contents, _permission, callback) => callback(false),
  );
  session.defaultSession.setPermissionCheckHandler(() => false);
}

/**
 * EPSILON_REVIEWER_DEV_URL（開発の Vite）。無ければ null。
 *
 * @throws Error localhost か 127.0.0.1 の http でない
 */
export function developmentUrl(value: string | undefined): string | null {
  if (!value) return null;
  if (!/^http:\/\/(localhost|127\.0\.0\.1)(:\d+)?\/?$/.test(value))
    throw new Error(
      "EPSILON_REVIEWER_DEV_URL must point to a local Vite server.",
    );
  return value;
}

/**
 * IPC の送り手を確かめる。主の窓の主の枠で、epsilon-reviewer://app か開発の URL の origin なら null、
 * 違えば失敗の中身。
 */
export function untrustedSender(
  window: BrowserWindow | null,
  event: IpcMainInvokeEvent,
  development: string | null,
): ErrorBody | null {
  const frame = event.senderFrame;
  if (
    window === null ||
    event.sender !== window.webContents ||
    frame === null ||
    frame !== window.webContents.mainFrame
  )
    return {
      code: "invalid_request",
      message: "IPC is restricted to the application main frame.",
    };
  const url = new URL(frame.url);
  if (url.protocol === `${APP_SCHEME}:` && url.hostname === APP_HOST)
    return null;
  if (development !== null && url.origin === new URL(development).origin)
    return null;
  return {
    code: "invalid_request",
    message: "IPC sender origin is not trusted.",
  };
}
