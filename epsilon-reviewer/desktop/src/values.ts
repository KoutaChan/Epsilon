// 型の分からない値を読むためのヘルパー。IPC、フレーム、ファイルの JSON、catch した例外に使う。

/** 配列でない JSON のオブジェクト。 */
export type Fields = Record<string, unknown>;

export function isFields(value: unknown): value is Fields {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** 例外の英語のメッセージを返す。ログと ErrorBody.message に使う。 */
export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
