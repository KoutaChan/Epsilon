package com.epsilon.reviewer.error;

/** 失敗の本文 {code, message}。HTTP の封筒・stdio の答え・解析の仕事の失敗で同じ形。 */
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
