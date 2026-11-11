/** 解析結果が読めないときのエラー。画面には、エラーコードに対応する文言（errors.invalid_result）を出す。メッセージは記録用で、英語で書く。 */
export class ReviewFormatError extends Error {
  readonly code = "invalid_result";
}
