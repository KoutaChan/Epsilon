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

// 操作の一覧。名前、HTTP のパス、stdio や画面から呼べるか、呼べる相手は Java の Command と同じにする。
// Web の HTTP もデスクトップ版の stdio も、この一覧に従って操作を送る。Java のテストがこのファイルを読んで
// 突き合わせるので、1 つの操作は下と同じ書き方で 1 つのブロックに書く。

/**
 * 誰が呼べるか。public は Cookie を読まない公開の API、owner は Cookie で決まるオーナー、share は共有リンク
 * （id を知っていれば誰でも）、local はデスクトップ版の Java の子プロセス。
 */
export type Access = "public" | "owner" | "share" | "local";

/**
 * 要求の本文。none はパスに入る値だけで本文なし、json はパスに入る値を除いたフィールドの JSON、
 * binary は牌譜のバイト列。
 */
export type RequestBody = "none" | "json" | "binary";

/**
 * 応答の本文。json は JSON、empty は本文なし（HTTP では 204）、document は gzip した解析結果の文書
 * （HTTP では Content-Encoding: gzip で、stdio では gzip のバイト列を本文にして送る）、attachment は HTTP の
 * ダウンロード。
 */
export type ResponseBody = "json" | "empty" | "document" | "attachment";

export interface HttpRoute {
  method: "GET" | "POST" | "DELETE";
  /** {名前} の所には、要求の同じ名前のフィールドを encodeURIComponent して入れる。 */
  path: string;
}

export interface CommandSpec {
  /** HTTP のメソッドとパス。デスクトップ版だけの操作では null。 */
  http: HttpRoute | null;
  /** stdio で Java が受けるか。 */
  stdio: boolean;
  /** デスクトップ版の画面がブリッジから直接呼べるか（false ならメインプロセスだけが呼ぶ）。 */
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
  /** 牌譜のファイルの絶対パス。Java が 16 MiB まで読む。 */
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
export type CommandResponse<N extends CommandName> =
  CommandTypes[N]["response"];

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

/** stdio で Java が受ける操作。 */
export type StdioCommandName = {
  [N in CommandName]: Table[N]["stdio"] extends true ? N : never;
}[CommandName];

/** デスクトップ版の画面がブリッジから呼べる操作。 */
export type RendererCommandName = {
  [N in CommandName]: Table[N]["renderer"] extends true ? N : never;
}[CommandName];

/** 牌譜のサイズの上限（バイト）。アップロード、ファイル、天鳳から取った本文のどれにも使う。 */
export const RECORD_SIZE_LIMIT = 16 * 1024 * 1024;

/** 解析の進み具合を聞き直す間隔（ミリ秒）。応答が来てから、これだけ待って次を聞く。 */
export const JOB_POLL_INTERVAL_MS = 750;

/**
 * HTTP のパスの {名前} を要求のフィールドで埋め、パスと、残ったフィールドを返す。残ったフィールドは json の本文に
 * なり、何も残らなければ null。
 *
 * @throws TypeError パスに入れるフィールドが文字列でないとき
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
