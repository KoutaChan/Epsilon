// 型の分からない値（IPC・枠・ファイルの JSON、catch した例外）の読み方。

/** 配列でない JSON の物。 */
export type Fields = Record<string, unknown>;

export function isFields(value: unknown): value is Fields {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** 例外の英語の文（記録と ErrorBody.message のため）。 */
export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
