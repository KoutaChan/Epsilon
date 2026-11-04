package com.epsilon.reviewer.error;

/** engine から入口までの唯一の失敗。符号と英語の文を持つ。 */
public final class ReviewerException extends RuntimeException {
  private final ErrorCode code;

  /** 符号の既定の文で作る。 */
  public ReviewerException(ErrorCode code) {
    this(code, code.message());
  }

  public ReviewerException(ErrorCode code, String message) {
    super(message);
    this.code = code;
  }

  public ErrorCode code() {
    return code;
  }
}
