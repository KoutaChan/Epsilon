// エラーコードは Web 版・デスクトップ版・Java で同じ文字列を使う。画面に出すのは errors.<コード> の文言だけ。

/**
 * Java の解析側が返すエラーコード。並び順と文字列は Java の ErrorCode と同じ（HTTP のステータスは Java 側で決める）。
 * 文言は language/<言語>/errors.json の errors.<コード> にある。
 */
export const SERVER_ERROR_CODES = [
  "not_found",
  "invalid_request",
  "host_not_allowed",
  "origin_not_allowed",
  "server_error",
  "shutting_down",
  "data_locked",
  "busy",
  "invalid_url",
  "unsupported_url",
  "record_not_found",
  "record_access_denied",
  "fetch_failed",
  "fetch_timeout",
  "record_too_large",
  "invalid_record",
  "too_many_records",
  "record_expired",
  "analysis_active",
  "queue_full",
  "model_unavailable",
  "job_not_found",
  "analysis_failed",
  "result_not_found",
  "share_not_found",
  "invalid_edit",
  "decision_not_found",
] as const;

/** 画面とデスクトップ版のアプリの中だけで作るエラーコード。Java は返さない。 */
export const CLIENT_ERROR_CODES = [
  "connection_failed",
  "invalid_response",
  "invalid_result",
  "renderer_failed",
  "engine_failed",
  "operation_failed",
] as const;

export type ServerErrorCode = (typeof SERVER_ERROR_CODES)[number];
export type ClientErrorCode = (typeof CLIENT_ERROR_CODES)[number];
export type ErrorCode = ServerErrorCode | ClientErrorCode;

/** エラーの中身。message は英語で、画面には出さない（ログと調査のためのもの）。 */
export interface ErrorBody {
  code: ErrorCode;
  message: string;
}

const KNOWN: ReadonlySet<string> = new Set<string>([
  ...SERVER_ERROR_CODES,
  ...CLIENT_ERROR_CODES,
]);

export function isErrorCode(value: unknown): value is ErrorCode {
  return typeof value === "string" && KNOWN.has(value);
}
