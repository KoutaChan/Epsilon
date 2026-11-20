/** 保存結果の読込エラー。英語のメッセージは不正なフィールドを画面に示す。 */
export class ReviewFormatError extends Error {
  readonly code = "invalid_result";
}
