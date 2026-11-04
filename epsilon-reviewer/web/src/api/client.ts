import { RECORD_SIZE_LIMIT } from "../../../protocol/commands.ts";
import type {
  AnalysisStarted,
  HistoryList,
  ImportedRecord,
  JobSnapshot,
  ModelCatalog,
  ModelKey,
  ShareLink,
} from "../../../protocol/model.ts";
import { resultFileName } from "../../../protocol/result.ts";
import type { ResultDocument } from "../../../protocol/result.ts";
import type { ScenarioClient } from "../review-entry.ts";
import { ApiError } from "./errors.ts";
import type { DesktopTransport, Transport } from "./transport.ts";

/** 画面が覚えておく解析結果の文書の数（開き直しは送らない）。 */
const DOCUMENT_CACHE_SIZE = 2;
/** ダウンロードの blob の URL を返すまでの時間。 */
const DOWNLOAD_URL_LIFETIME_MS = 1000;
/** 共有の id（= 解析結果の id）の形。 */
const RESULT_ID =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

/** 書き出しの結果。downloaded は Web のダウンロードの開始、saved と cancelled はデスクトップの保存の画面。 */
export type ExportOutcome = "downloaded" | "saved" | "cancelled";

export type DesktopSettingsAccess = Pick<
  DesktopTransport,
  "settings" | "setLanguage" | "chooseResultsDirectory"
>;

export interface ReviewerApi {
  readonly platform: Transport["platform"];
  /** デスクトップの設定の手。Web は null。 */
  readonly desktop: DesktopSettingsAccess | null;
  models(signal?: AbortSignal): Promise<ModelCatalog>;
  history(signal?: AbortSignal): Promise<HistoryList>;
  importUrl(url: string): Promise<ImportedRecord>;
  importFile(file: File): Promise<ImportedRecord>;
  reanalyse(resultId: string): Promise<ImportedRecord>;
  start(recordId: string, model: ModelKey): Promise<AnalysisStarted>;
  job(jobId: string, signal?: AbortSignal): Promise<JobSnapshot>;
  cancel(jobId: string): Promise<JobSnapshot>;
  result(resultId: string, signal?: AbortSignal): Promise<ResultDocument>;
  remove(resultId: string): Promise<void>;
  export(resultId: string): Promise<ExportOutcome>;
  shareLink(resultId: string, signal?: AbortSignal): Promise<ShareLink>;
  sharedResult(resultId: string, signal?: AbortSignal): Promise<ResultDocument>;
  /** もしもの評価の先。share は共有リンクの閲覧、owner は持ち主（デスクトップも）。 */
  scenarioClient(resultId: string, access: "owner" | "share"): ScenarioClient;
}

export function createApi(transport: Transport): ReviewerApi {
  // 最近開いた順。Map の挿入順の先頭が一番古い。
  const documents = new Map<string, ResultDocument>();
  const remember = (resultId: string, document: ResultDocument) => {
    documents.delete(resultId);
    documents.set(resultId, document);
    if (documents.size > DOCUMENT_CACHE_SIZE)
      documents.delete(documents.keys().next().value!);
  };
  return {
    platform: transport.platform,
    desktop: transport.platform === "desktop" ? transport : null,
    models: (signal) => transport.call("models", {}, { signal }),
    history: (signal) => transport.call("history", {}, { signal }),
    importUrl: (url) => transport.call("importTenhou", { url }),
    async importFile(file) {
      if (file.size > RECORD_SIZE_LIMIT)
        throw new ApiError(
          "record_too_large",
          `The game record exceeds ${RECORD_SIZE_LIMIT} bytes.`,
        );
      if (transport.platform === "desktop") {
        const path = transport.pathForFile(file);
        if (path !== "") return transport.call("importFile", { path });
      }
      return transport.call("importBytes", {}, { body: file });
    },
    reanalyse: (resultId) => transport.call("importStored", { resultId }),
    start: (recordId, { modelId, revision }) =>
      transport.call("startAnalysis", { recordId, modelId, revision }),
    job: (jobId, signal) => transport.call("job", { jobId }, { signal }),
    cancel: (jobId) => transport.call("cancelJob", { jobId }),
    async result(resultId, signal) {
      const document =
        documents.get(resultId) ??
        (await transport.call("result", { resultId }, { signal }));
      remember(resultId, document);
      return document;
    },
    async remove(resultId) {
      await transport.call("deleteResult", { resultId });
      documents.delete(resultId);
    },
    async export(resultId) {
      if (transport.platform === "desktop")
        return (await transport.exportResult(resultId)) ? "saved" : "cancelled";
      const url = URL.createObjectURL(
        await transport.call("downloadResult", { resultId }),
      );
      const link = document.createElement("a");
      link.href = url;
      link.download = resultFileName(resultId);
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), DOWNLOAD_URL_LIFETIME_MS);
      return "downloaded";
    },
    shareLink: (resultId, signal) =>
      transport.call("shareLink", { resultId }, { signal }),
    async sharedResult(resultId, signal) {
      if (!RESULT_ID.test(resultId))
        throw new ApiError(
          "share_not_found",
          "The shared analysis link is invalid.",
          404,
        );
      return transport.call("sharedResult", { resultId }, { signal });
    },
    scenarioClient(resultId, access) {
      const command = access === "share" ? "sharedScenario" : "scenario";
      return {
        evaluate: (query, signal) =>
          transport.call(command, { resultId, ...query }, { signal }),
      };
    },
  };
}
