/** 例外の英語のメッセージを返す。ログと ErrorBody.message に使う。 */
export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
