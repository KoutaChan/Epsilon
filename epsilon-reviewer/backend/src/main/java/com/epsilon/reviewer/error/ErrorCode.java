package com.epsilon.reviewer.error;

/**
 * エラーコード。{@code code} は画面の errors.&lt;code&gt; のキー、{@code status} は HTTP のステータス、{@code message} は英語の
 * 既定のメッセージ。並び順は protocol/errors.ts の SERVER_ERROR_CODES と同じ。
 */
public enum ErrorCode {
  /** ルートの表に無いパスかメソッド。 */
  NOT_FOUND("not_found", 404, "API route not found."),
  /** リクエストの形・型・本文の種類・大きさが正しくない。stdio で知らない操作を受けたときも同じ。 */
  INVALID_REQUEST("invalid_request", 400, "The request is invalid."),
  /** Host ヘッダーが公開の origin と合わない。 */
  HOST_NOT_ALLOWED("host_not_allowed", 403, "Host is not allowed."),
  /** Origin か Sec-Fetch-Site が、他のサイトからのリクエストであることを示す。 */
  ORIGIN_NOT_ALLOWED("origin_not_allowed", 403, "Cross-site requests are not allowed."),
  /** 予期しない例外。ログにスタックトレースを残す。 */
  SERVER_ERROR("server_error", 500, "The reviewer could not complete the operation."),
  /** シャットダウンの途中か、スレッドが割り込まれた。 */
  SHUTTING_DOWN("shutting_down", 503, "The reviewer is shutting down."),
  /** データのディレクトリのロックを別のレビュアーが持っている（起動に失敗する）。 */
  DATA_LOCKED("data_locked", 409, "The data directory is used by another reviewer."),
  /** もしもの順番待ちが上限を超えたか、3 秒待っても空きが出ない。 */
  BUSY("busy", 429, "What-if analysis is busy."),
  /** URL か天鳳の牌譜 id の形が正しくない。 */
  INVALID_URL("invalid_url", 400, "Expected a Tenhou record URL."),
  /** 天鳳以外のホストの URL。 */
  UNSUPPORTED_URL("unsupported_url", 400, "Only Tenhou record URLs are supported."),
  /** 天鳳が 404 を返した。 */
  RECORD_NOT_FOUND("record_not_found", 404, "The game record was not found at its source."),
  /** 天鳳が 401・403 を返した。 */
  RECORD_ACCESS_DENIED("record_access_denied", 422, "The game record provider denied access."),
  /** 牌譜を取れない。通信・DNS・TLS の失敗、リダイレクトの制限（回数・転送先）、200 以外のステータスのどれか。 */
  FETCH_FAILED("fetch_failed", 502, "Could not download the game record."),
  /** 天鳳の応答が 30 秒で終わらない。 */
  FETCH_TIMEOUT("fetch_timeout", 504, "The game record download timed out."),
  /** 牌譜が 16 MiB を超える。 */
  RECORD_TOO_LARGE("record_too_large", 413, "The game record exceeds 16 MiB."),
  /** 牌譜を読めないか再生できない。メッセージは牌譜を読むコードが出した英語のまま。 */
  INVALID_RECORD("invalid_record", 422, "The game record could not be read."),
  /** 読み込んだ牌譜が、オーナーごとに 4 件、全体で 64 件か 128 MiB を超える。 */
  TOO_MANY_RECORDS("too_many_records", 429, "Too many imported records are waiting."),
  /** 読み込んでから 30 分が過ぎたか、知らない牌譜の id。 */
  RECORD_EXPIRED("record_expired", 404, "The imported record has expired."),
  /** そのオーナーの解析がすでに順番待ちか実行中。 */
  ANALYSIS_ACTIVE("analysis_active", 409, "An analysis is already queued or running."),
  /** 解析の待ち行列が埋まっている。 */
  QUEUE_FULL("queue_full", 429, "The analysis queue is full."),
  /** モデルがカタログに無いか、使えないか、開けない。 */
  MODEL_UNAVAILABLE("model_unavailable", 409, "The model is not available."),
  /** 解析ジョブが無い（知らない id、別のオーナーのジョブ、終わってから 24 時間が過ぎた）。 */
  JOB_NOT_FOUND("job_not_found", 404, "The analysis job was not found."),
  /** 解析ジョブが失敗した。HTTP のエラー応答には使わず、ジョブの状態（{@code JobSnapshot.error}）にだけ出る。 */
  ANALYSIS_FAILED("analysis_failed", 500, "The analysis failed."),
  /** 解析結果が無い（知らない id、別のオーナーの結果、削除済み）。 */
  RESULT_NOT_FOUND("result_not_found", 404, "The analysis result was not found."),
  /** 共有リンクで指した解析結果が無い。 */
  SHARE_NOT_FOUND("share_not_found", 404, "The shared analysis was not found."),
  /** もしもの変更が判断の局面に合わない。メッセージは変更したフィールドのパスで始まる。 */
  INVALID_EDIT("invalid_edit", 400, "The edits do not match the position."),
  /** 指定した席の、指定した判断が解析に無い。 */
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
