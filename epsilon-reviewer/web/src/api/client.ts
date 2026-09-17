import type {
  BridgeRequest,
  HistoryEntry,
  ImportedRecord,
  Job,
  ModelCatalog,
  ReviewResult,
  ShareLink,
} from "./types";

import { ApiError } from "./errors.ts";
import { readCurrentReviewResult } from "./reviewFormat.ts";
export { ApiError } from "./errors.ts";

export const isDesktop = Boolean(window.epsilonReviewer);

interface RequestOptions {
  method?: BridgeRequest["method"];
  body?: unknown;
  fileName?: string;
  signal?: AbortSignal;
  credentials?: RequestCredentials;
}

async function request<T>(
  path: string,
  {
    method = "GET",
    body,
    fileName,
    signal,
    credentials = "same-origin",
  }: RequestOptions = {},
): Promise<T> {
  let status: number, data: unknown;
  if (window.epsilonReviewer) {
    ({ status, body: data } = await window.epsilonReviewer.request({
      path,
      method,
      body: body instanceof ArrayBuffer ? { bytes: body } : body,
      fileName,
    }));
    signal?.throwIfAborted();
  } else {
    const raw = body instanceof ArrayBuffer;
    const headers: Record<string, string> = {};
    if (body !== undefined)
      headers["Content-Type"] = raw
        ? "application/octet-stream"
        : "application/json";
    if (fileName) headers["X-File-Name"] = encodeURIComponent(fileName);
    const response = await fetch(path, {
      method,
      headers,
      credentials,
      signal,
      body: body === undefined ? undefined : raw ? body : JSON.stringify(body),
    }).catch((error: unknown) => {
      if (signal?.aborted) throw error;
      throw new ApiError(
        "Could not connect to the analysis server.",
        "connection_failed",
      );
    });
    status = response.status;
    const text = await response.text();
    try {
      data = text ? JSON.parse(text) : undefined;
    } catch {
      throw new ApiError(
        "The server returned invalid JSON.",
        "invalid_response",
        status,
      );
    }
  }
  if (status < 200 || status >= 300) {
    const { error } = data as { error: { message: string; code: string } };
    throw new ApiError(error.message, error.code, status);
  }
  return data as T;
}

export const api = {
  models: (signal?: AbortSignal) =>
    request<ModelCatalog>("/api/models", { signal }),
  history: (signal?: AbortSignal) =>
    request<{ results: HistoryEntry[] }>("/api/history", { signal }),
  importUrl: (url: string) =>
    request<ImportedRecord>("/api/records", { method: "POST", body: { url } }),
  importFile: async (file: File) =>
    request<ImportedRecord>("/api/records", {
      method: "POST",
      body: await file.arrayBuffer(),
      fileName: file.name,
    }),
  reanalyse: (resultId: string) =>
    request<ImportedRecord>("/api/records", {
      method: "POST",
      body: { resultId },
    }),
  start: (recordId: string, modelId: string, revision: string) =>
    request<{ jobId: string; model: ReviewResult["model"] }>("/api/analyses", {
      method: "POST",
      body: {
        recordId,
        modelId,
        revision,
        storage: isDesktop ? "desktop" : "web",
      },
    }),
  job: (id: string, signal?: AbortSignal) =>
    request<Job>(`/api/jobs/${encodeURIComponent(id)}`, { signal }),
  cancel: (id: string) =>
    request<void>(`/api/jobs/${encodeURIComponent(id)}/cancel`, {
      method: "POST",
    }),
  result: async (id: string, signal?: AbortSignal) =>
    readCurrentReviewResult(
      await request<unknown>(`/api/results/${encodeURIComponent(id)}`, {
        signal,
      }),
    ),
  shareLink: (id: string, signal?: AbortSignal) =>
    request<ShareLink>(`/api/results/${encodeURIComponent(id)}/share`, {
      signal,
    }),
  sharedResult: async (id: string, signal?: AbortSignal) => {
    if (
      !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(id)
    )
      throw new ApiError(
        "The shared analysis link is invalid.",
        "share_not_found",
        404,
      );
    return readCurrentReviewResult(
      await request<unknown>(`/api/shares/${id}`, {
        signal,
        credentials: "omit",
      }),
    );
  },
  remove: (id: string) =>
    request<void>(`/api/results/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  export: async (id: string) => {
    const path = `/api/results/${encodeURIComponent(id)}/export`;
    if (window.epsilonReviewer) {
      const result = await request<{ saved?: boolean; cancelled?: boolean }>(
        path,
      );
      if (result.cancelled === true) return false;
      if (typeof result.saved === "boolean") return result.saved;
      throw new ApiError(
        "The desktop export response is invalid.",
        "invalid_response",
      );
    }
    const response = await fetch(path, { credentials: "same-origin" }).catch(
      () => {
        throw new ApiError(
          "Could not connect to the analysis server.",
          "connection_failed",
        );
      },
    );
    if (!response.ok) {
      const { error } = await response.json();
      throw new ApiError(error.message, error.code, response.status);
    }
    const url = URL.createObjectURL(await response.blob());
    const link = document.createElement("a");
    link.href = url;
    link.download = `${id}.epsilon-reviewer.json.gz`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    return true;
  },
};
