import type {
  AnalysisStarted,
  HistoryList,
  ImportedRecord,
  JobSnapshot,
  ModelCatalog,
  NoFields,
  ScenarioAnswer,
  ScenarioQuery,
  ShareLink,
} from "./model.ts";
import type { ResultDocument } from "./result.ts";

// 操作の表。Java の Command と同じ名前・道・入口を持ち、Web の HTTP とデスクトップの stdio の両方がこの表で操作を送る。
// Java の試験がこのファイルを読んで突き合わせるので、1 つの操作は下の書き方のまま 1 つの塊に書く。

/**
 * 誰が呼べるか。public は Cookie を読まない公開の道、owner は Cookie の持ち主、share は共有リンク
 * （id を知る誰でも）、local はデスクトップの Java の子プロセス。
 */
export type Access = "public" | "owner" | "share" | "local";

/** 要求の本文。none は道の欄だけ、json は道の欄を除いた欄の JSON、binary は牌譜のバイト列。 */
export type RequestBody = "none" | "json" | "binary";

/**
 * 応答の本文。json は JSON、empty は本文なし（HTTP は 204）、document は gzip の解析結果の文書
 * （HTTP は Content-Encoding: gzip で、stdio は gzip のバイト列の本文で送る）、attachment は HTTP の
 * ダウンロード。
 */
export type ResponseBody = "json" | "empty" | "document" | "attachment";

export interface HttpRoute {
  method: "GET" | "POST" | "DELETE";
  /** {名前} は要求の同じ名前の欄（encodeURIComponent して埋める）。 */
  path: string;
}

export interface CommandSpec {
  /** HTTP の道。デスクトップだけの操作は null。 */
  http: HttpRoute | null;
  /** stdio で Java が受けるか。 */
  stdio: boolean;
  /** デスクトップの画面が橋渡しで直接呼べるか（false はメインプロセスだけが呼ぶ）。 */
  renderer: boolean;
  access: readonly Access[];
  request: RequestBody;
  response: ResponseBody;
}

export interface ResultRef {
  resultId: string;
}
export interface JobRef {
  jobId: string;
}
export interface FileImportRequest {
  /** 絶対パス。Java が 16 MiB まで読む。 */
  path: string;
}
export interface TenhouImportRequest {
  url: string;
}
export interface AnalysisRequest {
  recordId: string;
  modelId: string;
  revision: string;
}
export interface ExportRequest {
  resultId: string;
  /** 保存のダイアログで選んだ絶対パス。 */
  path: string;
}
export type ScenarioRequest = ResultRef & ScenarioQuery;

/** 操作ごとの要求と応答の型。 */
export interface CommandTypes {
  models: { request: NoFields; response: ModelCatalog };
  history: { request: NoFields; response: HistoryList };
  importBytes: { request: NoFields; response: ImportedRecord };
  importFile: { request: FileImportRequest; response: ImportedRecord };
  importTenhou: { request: TenhouImportRequest; response: ImportedRecord };
  importStored: { request: ResultRef; response: ImportedRecord };
  startAnalysis: { request: AnalysisRequest; response: AnalysisStarted };
  job: { request: JobRef; response: JobSnapshot };
  cancelJob: { request: JobRef; response: JobSnapshot };
  result: { request: ResultRef; response: ResultDocument };
  deleteResult: { request: ResultRef; response: void };
  downloadResult: { request: ResultRef; response: Blob };
  exportResult: { request: ExportRequest; response: void };
  shareLink: { request: ResultRef; response: ShareLink };
  sharedResult: { request: ResultRef; response: ResultDocument };
  scenario: { request: ScenarioRequest; response: ScenarioAnswer };
  sharedScenario: { request: ScenarioRequest; response: ScenarioAnswer };
}

export type CommandName = keyof CommandTypes;
export type CommandRequest<N extends CommandName> = CommandTypes[N]["request"];
export type CommandResponse<N extends CommandName> = CommandTypes[N]["response"];

export const COMMANDS = {
  models: {
    http: { method: "GET", path: "/api/models" },
    stdio: true,
    renderer: true,
    access: ["public", "local"],
    request: "none",
    response: "json",
  },
  history: {
    http: { method: "GET", path: "/api/history" },
    stdio: true,
    renderer: true,
    access: ["owner", "local"],
    request: "none",
    response: "json",
  },
  importBytes: {
    http: { method: "POST", path: "/api/records" },
    stdio: true,
    renderer: true,
    access: ["owner", "local"],
    request: "binary",
    response: "json",
  },
  importFile: {
    http: null,
    stdio: true,
    renderer: true,
    access: ["local"],
    request: "json",
    response: "json",
  },
  importTenhou: {
    http: { method: "POST", path: "/api/records/tenhou" },
    stdio: true,
    renderer: true,
    access: ["owner", "local"],
    request: "json",
    response: "json",
  },
  importStored: {
    http: { method: "POST", path: "/api/results/{resultId}/record" },
    stdio: true,
    renderer: true,
    access: ["owner", "local"],
    request: "none",
    response: "json",
  },
  startAnalysis: {
    http: { method: "POST", path: "/api/analyses" },
    stdio: true,
    renderer: true,
    access: ["owner", "local"],
    request: "json",
    response: "json",
  },
  job: {
    http: { method: "GET", path: "/api/jobs/{jobId}" },
    stdio: true,
    renderer: true,
    access: ["owner", "local"],
    request: "none",
    response: "json",
  },
  cancelJob: {
    http: { method: "POST", path: "/api/jobs/{jobId}/cancel" },
    stdio: true,
    renderer: true,
    access: ["owner", "local"],
    request: "none",
    response: "json",
  },
  result: {
    http: { method: "GET", path: "/api/results/{resultId}" },
    stdio: true,
    renderer: true,
    access: ["owner", "local"],
    request: "none",
    response: "document",
  },
  deleteResult: {
    http: { method: "DELETE", path: "/api/results/{resultId}" },
    stdio: true,
    renderer: true,
    access: ["owner", "local"],
    request: "none",
    response: "empty",
  },
  downloadResult: {
    http: { method: "GET", path: "/api/results/{resultId}/download" },
    stdio: false,
    renderer: false,
    access: ["owner"],
    request: "none",
    response: "attachment",
  },
  exportResult: {
    http: null,
    stdio: true,
    renderer: false,
    access: ["local"],
    request: "json",
    response: "empty",
  },
  shareLink: {
    http: { method: "GET", path: "/api/results/{resultId}/share" },
    stdio: false,
    renderer: false,
    access: ["owner"],
    request: "none",
    response: "json",
  },
  sharedResult: {
    http: { method: "GET", path: "/api/shares/{resultId}" },
    stdio: false,
    renderer: false,
    access: ["share"],
    request: "none",
    response: "document",
  },
  scenario: {
    http: { method: "POST", path: "/api/results/{resultId}/scenario" },
    stdio: true,
    renderer: true,
    access: ["owner", "local"],
    request: "json",
    response: "json",
  },
  sharedScenario: {
    http: { method: "POST", path: "/api/shares/{resultId}/scenario" },
    stdio: false,
    renderer: false,
    access: ["share"],
    request: "json",
    response: "json",
  },
} as const satisfies { readonly [N in CommandName]: CommandSpec };

type Table = typeof COMMANDS;

/** HTTP の道を持つ操作。 */
export type HttpCommandName = {
  [N in CommandName]: Table[N]["http"] extends null ? never : N;
}[CommandName];

/** stdio で Java が受ける操作。 */
export type StdioCommandName = {
  [N in CommandName]: Table[N]["stdio"] extends true ? N : never;
}[CommandName];

/** デスクトップの画面が橋渡しで呼べる操作。 */
export type RendererCommandName = {
  [N in CommandName]: Table[N]["renderer"] extends true ? N : never;
}[CommandName];

/** 牌譜の大きさの上限（アップロード・ファイル・天鳳の本文）。 */
export const RECORD_SIZE_LIMIT = 16 * 1024 * 1024;

/** JSON の要求の本文の上限（HTTP）。 */
export const JSON_BODY_LIMIT = 16 * 1024;

/** 解析の進みを聞き直す間隔。応答が来てからこれだけ待つ。 */
export const JOB_POLL_INTERVAL_MS = 750;

/**
 * HTTP の道の {名前} を要求の欄で埋め、残りの欄を返す（json の本文。残りが無ければ null）。
 *
 * @throws TypeError 道の欄が文字列でない
 */
export function httpTarget(
  route: HttpRoute,
  request: object,
): { path: string; body: Record<string, unknown> | null } {
  const rest: Record<string, unknown> = { ...request };
  const path = route.path.replace(/\{(\w+)\}/g, (_token, name: string) => {
    const value = rest[name];
    if (typeof value !== "string")
      throw new TypeError(`Missing path parameter: ${name}`);
    delete rest[name];
    return encodeURIComponent(value);
  });
  return { path, body: Object.keys(rest).length > 0 ? rest : null };
}
