import { DesktopError, type ApiResponse } from "./errors";

export type Fetch = (url: string, init: RequestInit) => Promise<Response>;
export interface Upload {
  bytes: Uint8Array;
  fileName: string;
  recordSource?: string;
}

/** HTTP を所有する。保存・画面・現在の接続設定には依存しない。 */
export class RemoteClient {
  constructor(private readonly fetch: Fetch) {}

  async request(
    origin: string,
    pathname: string,
    method = "GET",
    body?: unknown,
    upload?: Upload,
  ): Promise<Response> {
    const headers: Record<string, string> = { Accept: "application/json" };
    let data: BodyInit | undefined;
    if (upload) {
      data = upload.bytes as Uint8Array<ArrayBuffer>;
      headers["Content-Type"] = "application/octet-stream";
      headers["X-File-Name"] = encodeURIComponent(upload.fileName);
      if (upload.recordSource) headers["X-Record-Source"] = upload.recordSource;
    } else if (body !== undefined) {
      data = JSON.stringify(body);
      headers["Content-Type"] = "application/json";
    }
    try {
      return await this.fetch(origin + pathname, {
        method,
        headers,
        body: data,
        credentials: "include",
        redirect: "error",
        signal: AbortSignal.timeout(120_000),
      });
    } catch (cause) {
      throw new DesktopError(
        "server_unavailable",
        "The analysis server could not be reached.",
        502,
        { cause },
      );
    }
  }

  async json<T = unknown>(
    origin: string,
    pathname: string,
    method = "GET",
    body?: unknown,
    upload?: Upload,
  ): Promise<ApiResponse<T>> {
    const response = await this.request(origin, pathname, method, body, upload);
    if (response.status === 204) return { status: 204, body: null as T };
    try {
      return { status: response.status, body: (await response.json()) as T };
    } catch (cause) {
      throw new DesktopError(
        "invalid_response",
        "The server returned invalid JSON (HTTP " + response.status + ").",
        502,
        { cause },
      );
    }
  }
}
