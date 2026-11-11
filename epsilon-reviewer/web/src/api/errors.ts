import type { ErrorCode } from "../../../protocol/errors.ts";

/** 操作の失敗。画面にはエラーコードに対応する文言（errors.<code>）だけを出す。message は原因を調べるための英語のメッセージ。 */
export class ApiError extends Error {
  readonly code: ErrorCode;
  /** HTTP のステータスコード。HTTP の応答から作ったのでなければ 0。 */
  readonly status: number;

  constructor(code: ErrorCode, message: string, status = 0) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}
