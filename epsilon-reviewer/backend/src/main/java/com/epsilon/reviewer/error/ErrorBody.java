package com.epsilon.reviewer.error;

/** エラーの本文 {code, message}。HTTP のエラー応答、stdio の返答、解析ジョブの失敗で同じ形を使う。 */
public record ErrorBody(String code, String message) {
  public static ErrorBody of(ErrorCode code, String message) {
    return new ErrorBody(code.code(), message);
  }

  public static ErrorBody of(ErrorCode code) {
    return of(code, code.message());
  }

  public static ErrorBody of(ReviewerException failure) {
    return of(failure.code(), failure.getMessage());
  }
}
