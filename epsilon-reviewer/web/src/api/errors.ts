import type { ErrorCode } from "../../../protocol/errors.ts";

/** 操作の失敗。画面は code の文言（errors.<code>）だけを出し、message は英語の調べ物の文。 */
export class ApiError extends Error {
  readonly code: ErrorCode;
  /** HTTP の状態。HTTP 以外で作った失敗は 0。 */
  readonly status: number;

  constructor(code: ErrorCode, message: string, status = 0) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}
