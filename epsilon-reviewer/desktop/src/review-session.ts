import type { SettingsStore } from "./settings";
import { ResultStore } from "./result-store";
import { RemoteClient } from "./remote-client";
import { DesktopError, type ApiResponse } from "./errors";

export interface RequestInput {
  path: string;
  method: "GET" | "POST" | "DELETE";
  body?: unknown;
  fileName?: string;
}
interface Job {
  jobId: string;
  status: string;
  resultId?: string;
}

export function parseDesktopApiRequest(value: unknown): RequestInput {
  const input = value as Partial<RequestInput> | null;
  if (
    !input ||
    typeof input.path !== "string" ||
    !["GET", "POST", "DELETE"].includes(input.method ?? "") ||
    (input.fileName !== undefined && typeof input.fileName !== "string")
  ) {
    throw new DesktopError("invalid_request", "Invalid desktop request.");
  }
  return input as RequestInput;
}

/** ジョブの接続先と回収を所有する。履歴アクセスはネットワークを待たない。 */
export class ReviewSession {
  private readonly jobs = new Map<string, string>();
  private readonly transfers = new Map<string, Promise<void>>();
  private readonly cleanup = new Map<string, string>();
  private cleanupRunning = false;

  constructor(
    private readonly settings: SettingsStore,
    private readonly store: ResultStore,
    private readonly remote: RemoteClient,
    private readonly exportResult: (
      id: string,
      bytes: Uint8Array,
    ) => Promise<ApiResponse>,
  ) {}

  async request(input: RequestInput): Promise<ApiResponse> {
    const { path: pathname, method, body } = input;
    void this.releaseRemoteResults();

    if (pathname === "/api/history" && method === "GET")
      return { status: 200, body: { results: await this.store.list() } };

    const resultRoute =
      /^\/api\/results\/([a-zA-Z0-9_-]{1,128})(\/export)?$/.exec(pathname);
    if (resultRoute) {
      const id = resultRoute[1]!;
      if (method === "DELETE" && !resultRoute[2]) {
        await this.store.remove(id);
        return { status: 204, body: null };
      }
      if (method === "GET") {
        return resultRoute[2]
          ? this.exportResult(id, await this.store.readArchive(id))
          : { status: 200, body: await this.store.get(id) };
      }
      throw new DesktopError(
        "invalid_request",
        "This result operation is not supported.",
        405,
      );
    }

    const origin = this.settings.value.serverUrl;
    if (pathname === "/api/models" && method === "GET")
      return this.remote.json(origin, pathname);

    if (pathname === "/api/records" && method === "POST") {
      if (input.fileName) {
        const bytes = (body as { bytes?: unknown } | null)?.bytes;
        if (!(bytes instanceof ArrayBuffer))
          throw new DesktopError(
            "invalid_request",
            "A record upload must contain an ArrayBuffer.",
          );
        if (bytes.byteLength > 16 * 1024 * 1024)
          throw new DesktopError(
            "file_too_large",
            "The record exceeds the 16 MiB limit.",
            413,
          );
        return this.remote.json(origin, pathname, method, undefined, {
          bytes: new Uint8Array(bytes),
          fileName: input.fileName,
        });
      }
      if (body && typeof body === "object" && "resultId" in body) {
        const source = await this.store.getSource(String(body.resultId));
        return this.remote.json(origin, pathname, method, undefined, {
          bytes: Buffer.from(source.base64, "base64"),
          fileName: source.fileName,
          recordSource: source.recordSource,
        });
      }
      return this.remote.json(origin, pathname, method, body);
    }

    if (pathname === "/api/analyses" && method === "POST") {
      const { recordId, modelId, revision } = body as {
        recordId: string;
        modelId: string;
        revision: string;
      };
      const response = await this.remote.json<Job>(origin, pathname, method, {
        recordId,
        modelId,
        revision,
        storage: "desktop",
      });
      if (response.status === 202) this.jobs.set(response.body.jobId, origin);
      return response;
    }

    const jobRoute = /^\/api\/jobs\/([a-zA-Z0-9_-]{1,128})(\/cancel)?$/.exec(
      pathname,
    );
    if (
      jobRoute &&
      ((method === "GET" && !jobRoute[2]) || (method === "POST" && jobRoute[2]))
    ) {
      const jobOrigin = this.jobs.get(jobRoute[1]!);
      if (!jobOrigin)
        throw new DesktopError(
          "invalid_request",
          "The job does not belong to this desktop session.",
          404,
        );
      const response = await this.remote.json<Job>(jobOrigin, pathname, method);
      if (
        method === "GET" &&
        response.status === 200 &&
        response.body.status === "completed" &&
        response.body.resultId
      ) {
        await this.collect(response.body.resultId, jobOrigin);
      }
      return response;
    }
    throw new DesktopError(
      "invalid_request",
      "This desktop API route is not supported.",
      404,
    );
  }

  private async collect(id: string, origin: string): Promise<void> {
    if (await this.store.has(id)) return;
    let pending = this.transfers.get(id);
    if (!pending) {
      pending = this.download(id, origin).finally(() =>
        this.transfers.delete(id),
      );
      this.transfers.set(id, pending);
    }
    await pending;
  }

  private async download(id: string, origin: string): Promise<void> {
    const archive = await this.remote.request(
      origin,
      "/api/results/" + id + "/export",
    );
    if (archive.status !== 200)
      throw new DesktopError(
        "result_transfer_failed",
        "The completed result could not be downloaded.",
        502,
      );
    const source = await this.remote.request(
      origin,
      "/api/results/" + id + "/source",
    );
    if (!source.ok)
      throw new DesktopError(
        "source_transfer_failed",
        "The original record could not be downloaded.",
        502,
      );
    const fileName = source.headers.get("X-File-Name");
    if (!fileName)
      throw new DesktopError(
        "invalid_response",
        "The original record response has no file name.",
        502,
      );
    await this.store.saveArchive(
      id,
      new Uint8Array(await archive.arrayBuffer()),
      this.settings.value.resultsDirectory,
      {
        fileName: decodeURIComponent(fileName),
        recordSource: source.headers.get("X-Record-Source") ?? undefined,
        base64: Buffer.from(await source.arrayBuffer()).toString("base64"),
      },
    );
    this.cleanup.set(id, origin);
    void this.releaseRemoteResults();
  }

  private async releaseRemoteResults(): Promise<void> {
    if (this.cleanupRunning || !this.cleanup.size) return;
    this.cleanupRunning = true;
    try {
      for (const [id, origin] of this.cleanup) {
        try {
          const response = await this.remote.request(
            origin,
            "/api/results/" + id,
            "DELETE",
          );
          if ([200, 204, 404].includes(response.status))
            this.cleanup.delete(id);
          await response.body?.cancel();
        } catch {
          // ローカル保存済みの結果だけを再試行する。未回収分はサーバー TTL でも整理する。
        }
      }
    } finally {
      this.cleanupRunning = false;
    }
  }
}
