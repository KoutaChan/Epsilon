import { BrowserWindow, session, type IpcMainInvokeEvent } from "electron";
import type { ErrorBody } from "../../protocol/errors.ts";
import { APP_HOST, APP_SCHEME } from "./appProtocol.ts";

export interface WindowOptions {
  title: string;
  icon: string;
  preload: string;
}

/** メインウィンドウを作る。新しいウィンドウを開くことと、ほかのページへの移動は断る。 */
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

/** 既定のセッションで、権限の要求と権限の確認を全て断る。 */
export function denyPermissions(): void {
  session.defaultSession.setPermissionRequestHandler(
    (_contents, _permission, callback) => callback(false),
  );
  session.defaultSession.setPermissionCheckHandler(() => false);
}

/**
 * 開発用の Vite の URL（EPSILON_REVIEWER_DEV_URL の値）を確かめて返す。設定されていなければ null。
 *
 * @throws Error localhost か 127.0.0.1 を指す http の URL でないとき
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
 * IPC の送信元を確かめる。メインウィンドウのメインフレームから来ていて、その origin が epsilon-reviewer://app か
 * 開発用の URL なら null を返す。そうでなければ返すエラーの中身を返す。
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
