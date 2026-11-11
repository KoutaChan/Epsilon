package com.epsilon.reviewer.error;

/** エンジンから入口（HTTP・stdio）まで失敗を伝える、ただ 1 つの例外。エラーコードと英語のメッセージを持つ。 */
public final class ReviewerException extends RuntimeException {
  private final ErrorCode code;

  /** エラーコードの既定のメッセージで作る。 */
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
