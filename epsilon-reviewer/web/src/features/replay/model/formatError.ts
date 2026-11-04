/** 読めない解析結果。code で画面の errors.invalid_result の文言になる。message は英語（記録のため）。 */
export class ReviewFormatError extends Error {
  readonly code = "invalid_result";
}
