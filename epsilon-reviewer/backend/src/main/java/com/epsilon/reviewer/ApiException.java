package com.epsilon.reviewer;

/** HTTPステータスと、クライアントへ公開できる診断情報を持つ例外。 */
public final class ApiException extends RuntimeException {
  private final int status;
  private final String code;

  public ApiException(int status, String code, String message) {
    super(message);
    this.status = status;
    this.code = code;
  }

  public int status() {
    return status;
  }

  public String code() {
    return code;
  }

  public ApiError error() {
    return new ApiError(code, getMessage());
  }
}
