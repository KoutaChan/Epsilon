import { stat } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { net, protocol } from "electron";

export const APP_SCHEME = "epsilon-reviewer";
export const APP_HOST = "app";
export const APP_URL = `${APP_SCHEME}://${APP_HOST}/`;

/** アプリのスキームを特権付きで登録する。アプリが ready になる前に呼ぶ。 */
export function registerAppScheme(): void {
  protocol.registerSchemesAsPrivileged([
    {
      scheme: APP_SCHEME,
      privileges: {
        standard: true,
        secure: true,
        supportFetchAPI: true,
        corsEnabled: true,
      },
    },
  ]);
}

/**
 * epsilon-reviewer://app/ の下にある画面のファイルを返す。受けるのは GET だけで、"/" には index.html を返す。
 * ほかのホスト、GET 以外、画面のフォルダーの外を指す要求には 403 を、パスを読めなければ 400 を、ファイルが無ければ
 * 404 を返す。
 */
export function registerAppProtocol(webRoot: string): void {
  protocol.handle(APP_SCHEME, async (request) => {
    const url = new URL(request.url);
    if (url.hostname !== APP_HOST || request.method !== "GET")
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
