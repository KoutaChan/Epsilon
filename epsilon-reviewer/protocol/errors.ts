// 失敗の符号は Web・デスクトップ・Java で同じ文字列を使い、画面は errors.<符号> の文言だけを出す。

/**
 * Java の解析の側が返す失敗の符号。並びと文字列は Java の ErrorCode と同じ（HTTP の状態は Java の側が持つ）。
 * 文言は language/<言語>/errors.json の errors.<符号>。
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

/** 画面とデスクトップのアプリの中だけで作る失敗の符号。Java は返さない。 */
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

/** 失敗の中身。message は英語で、画面には出さない（記録と調べ物のため）。 */
export interface ErrorBody {
  code: ErrorCode;
  message: string;
}

/** HTTP の 2xx 以外の本文。 */
export interface ErrorEnvelope {
  error: ErrorBody;
}

const KNOWN: ReadonlySet<string> = new Set<string>([
  ...SERVER_ERROR_CODES,
  ...CLIENT_ERROR_CODES,
]);

export function isErrorCode(value: unknown): value is ErrorCode {
  return typeof value === "string" && KNOWN.has(value);
}
