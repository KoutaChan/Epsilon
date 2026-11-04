package com.epsilon.reviewer.error;

/**
 * 失敗の符号。code は画面の errors.&lt;code&gt;、status は HTTP の状態、message は英語の既定の文。並びは
 * protocol/errors.ts の SERVER_ERROR_CODES と同じ。
 */
public enum ErrorCode {
  /** 道の表に無い道・方法。 */
  NOT_FOUND("not_found", 404, "API route not found."),
  /** 要求の形・型・本文の種類・大きさが違う。知らない stdio の操作も含む。 */
  INVALID_REQUEST("invalid_request", 400, "The request is invalid."),
  /** Host が公開の origin と合わない。 */
  HOST_NOT_ALLOWED("host_not_allowed", 403, "Host is not allowed."),
  /** Origin・Sec-Fetch-Site が他のサイトからの要求を示す。 */
  ORIGIN_NOT_ALLOWED("origin_not_allowed", 403, "Cross-site requests are not allowed."),
  /** 予期しない例外。記録に積み跡を残す。 */
  SERVER_ERROR("server_error", 500, "The reviewer could not complete the operation."),
  /** 終わりの途中か、割り込まれた。 */
  SHUTTING_DOWN("shutting_down", 503, "The reviewer is shutting down."),
  /** 保存先の鍵を他のレビュアーが持つ（起動の失敗）。 */
  DATA_LOCKED("data_locked", 409, "The data directory is used by another reviewer."),
  /** もしもの待ちが上限を超えたか、3 秒待っても空かない。 */
  BUSY("busy", 429, "What-if analysis is busy."),
  /** URL の形か天鳳の牌譜 id の形が違う。 */
  INVALID_URL("invalid_url", 400, "Expected a Tenhou record URL."),
  /** 天鳳以外の主の URL。 */
  UNSUPPORTED_URL("unsupported_url", 400, "Only Tenhou record URLs are supported."),
  /** 天鳳が 404 を返した。 */
  RECORD_NOT_FOUND("record_not_found", 404, "The game record was not found at its source."),
  /** 天鳳が 401・403 を返した。 */
  RECORD_ACCESS_DENIED("record_access_denied", 422, "The game record provider denied access."),
  /** 転送・DNS・TLS・転送先の決まり・その他の状態で牌譜を取れない。 */
  FETCH_FAILED("fetch_failed", 502, "Could not download the game record."),
  /** 天鳳の応答が 30 秒で終わらない。 */
  FETCH_TIMEOUT("fetch_timeout", 504, "The game record download timed out."),
  /** 牌譜が 16 MiB を超える。 */
  RECORD_TOO_LARGE("record_too_large", 413, "The game record exceeds 16 MiB."),
  /** 牌譜を読めない・再生できない。文は読み手の英語。 */
  INVALID_RECORD("invalid_record", 422, "The game record could not be read."),
  /** 読み込んだ牌譜が持ち主ごと 4 件・全体 64 件・128 MiB を超える。 */
  TOO_MANY_RECORDS("too_many_records", 429, "Too many imported records are waiting."),
  /** 読み込んだ牌譜が 30 分を過ぎたか、知らない recordId。 */
  RECORD_EXPIRED("record_expired", 404, "The imported record has expired."),
  /** 持ち主の解析が既に待ちか走りにある。 */
  ANALYSIS_ACTIVE("analysis_active", 409, "An analysis is already queued or running."),
  /** 解析の待ちが埋まっている。 */
  QUEUE_FULL("queue_full", 429, "The analysis queue is full."),
  /** モデルがカタログに無いか、使えないか、開けない。 */
  MODEL_UNAVAILABLE("model_unavailable", 409, "The model is not available."),
  /** 解析の仕事が無い（知らない・他の持ち主・終わってから 24 時間後）。 */
  JOB_NOT_FOUND("job_not_found", 404, "The analysis job was not found."),
  /** 解析の仕事が失敗した（JobSnapshot.error だけに出る）。 */
  ANALYSIS_FAILED("analysis_failed", 500, "The analysis failed."),
  /** 解析結果が無い（知らない・他の持ち主・削除の後）。 */
  RESULT_NOT_FOUND("result_not_found", 404, "The analysis result was not found."),
  /** 共有の道の解析結果が無い。 */
  SHARE_NOT_FOUND("share_not_found", 404, "The shared analysis was not found."),
  /** もしもの変更が判断の局面に合わない。文は欄の道で始まる。 */
  INVALID_EDIT("invalid_edit", 400, "The edits do not match the position."),
  /** その席のその判断が解析に無い。 */
  DECISION_NOT_FOUND("decision_not_found", 404, "Decision not found in the analysis.");

  private final String code;
  private final int status;
  private final String message;

  ErrorCode(String code, int status, String message) {
    this.code = code;
    this.status = status;
    this.message = message;
  }

  public String code() {
    return code;
  }

  public int status() {
    return status;
  }

  public String message() {
    return message;
  }
}
