package com.epsilon.reviewer.api;

import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * 操作の表。protocol/commands.ts の COMMANDS と同じ名前・並び・HTTP の道・stdio・入口・本文の種類
 * （ProtocolContractTest が突き合わせる）。HTTP の道の {名前} は要求の record の同じ名前の欄。
 */
public enum Command {
  MODELS(
      "models",
      "GET",
      "/api/models",
      true,
      EnumSet.of(Access.PUBLIC, Access.LOCAL),
      RequestBody.NONE,
      ResponseBody.JSON,
      NoRequest.class),
  /** 持ち主の履歴（新しい順）。 */
  HISTORY(
      "history",
      "GET",
      "/api/history",
      true,
      EnumSet.of(Access.OWNER, Access.LOCAL),
      RequestBody.NONE,
      ResponseBody.JSON,
      NoRequest.class),
  IMPORT_BYTES(
      "importBytes",
      "POST",
      "/api/records",
      true,
      EnumSet.of(Access.OWNER, Access.LOCAL),
      RequestBody.BINARY,
      ResponseBody.JSON,
      NoRequest.class),
  /** デスクトップのファイルの取り込み（絶対パス）。 */
  IMPORT_FILE(
      "importFile",
      null,
      null,
      true,
      EnumSet.of(Access.LOCAL),
      RequestBody.JSON,
      ResponseBody.JSON,
      FileImportRequest.class),
  IMPORT_TENHOU(
      "importTenhou",
      "POST",
      "/api/records/tenhou",
      true,
      EnumSet.of(Access.OWNER, Access.LOCAL),
      RequestBody.JSON,
      ResponseBody.JSON,
      TenhouImportRequest.class),
  /** 保存した結果の元の牌譜の取り込み（解析し直し）。 */
  IMPORT_STORED(
      "importStored",
      "POST",
      "/api/results/{resultId}/record",
      true,
      EnumSet.of(Access.OWNER, Access.LOCAL),
      RequestBody.NONE,
      ResponseBody.JSON,
      ResultRequest.class),
  START_ANALYSIS(
      "startAnalysis",
      "POST",
      "/api/analyses",
      true,
      EnumSet.of(Access.OWNER, Access.LOCAL),
      RequestBody.JSON,
      ResponseBody.JSON,
      AnalysisRequest.class),
  /** 解析の進み。 */
  JOB(
      "job",
      "GET",
      "/api/jobs/{jobId}",
      true,
      EnumSet.of(Access.OWNER, Access.LOCAL),
      RequestBody.NONE,
      ResponseBody.JSON,
      JobRequest.class),
  CANCEL_JOB(
      "cancelJob",
      "POST",
      "/api/jobs/{jobId}/cancel",
      true,
      EnumSet.of(Access.OWNER, Access.LOCAL),
      RequestBody.NONE,
      ResponseBody.JSON,
      JobRequest.class),
  RESULT(
      "result",
      "GET",
      "/api/results/{resultId}",
      true,
      EnumSet.of(Access.OWNER, Access.LOCAL),
      RequestBody.NONE,
      ResponseBody.DOCUMENT,
      ResultRequest.class),
  /** 解析結果の削除（共有も止まる）。 */
  DELETE_RESULT(
      "deleteResult",
      "DELETE",
      "/api/results/{resultId}",
      true,
      EnumSet.of(Access.OWNER, Access.LOCAL),
      RequestBody.NONE,
      ResponseBody.EMPTY,
      ResultRequest.class),
  /** 解析結果のダウンロード（Web の書き出し）。 */
  DOWNLOAD_RESULT(
      "downloadResult",
      "GET",
      "/api/results/{resultId}/download",
      false,
      EnumSet.of(Access.OWNER),
      RequestBody.NONE,
      ResponseBody.ATTACHMENT,
      ResultRequest.class),
  /** デスクトップの解析結果の書き出し（メインプロセスだけが呼ぶ）。 */
  EXPORT_RESULT(
      "exportResult",
      null,
      null,
      true,
      EnumSet.of(Access.LOCAL),
      RequestBody.JSON,
      ResponseBody.EMPTY,
      ExportRequest.class),
  SHARE_LINK(
      "shareLink",
      "GET",
      "/api/results/{resultId}/share",
      false,
      EnumSet.of(Access.OWNER),
      RequestBody.NONE,
      ResponseBody.JSON,
      ResultRequest.class),
  SHARED_RESULT(
      "sharedResult",
      "GET",
      "/api/shares/{resultId}",
      false,
      EnumSet.of(Access.SHARE),
      RequestBody.NONE,
      ResponseBody.DOCUMENT,
      ResultRequest.class),
  SCENARIO(
      "scenario",
      "POST",
      "/api/results/{resultId}/scenario",
      true,
      EnumSet.of(Access.OWNER, Access.LOCAL),
      RequestBody.JSON,
      ResponseBody.JSON,
      ScenarioRequest.class),
  /** 共有リンクのもしも（保存の牌譜をサーバで評価し、牌譜は閲覧者へ送らない）。 */
  SHARED_SCENARIO(
      "sharedScenario",
      "POST",
      "/api/shares/{resultId}/scenario",
      false,
      EnumSet.of(Access.SHARE),
      RequestBody.JSON,
      ResponseBody.JSON,
      ScenarioRequest.class);

  private static final Map<String, Command> BY_WIRE_NAME =
      Arrays.stream(values())
          .collect(Collectors.toUnmodifiableMap(Command::wireName, Function.identity()));

  private final String wireName;
  private final String httpMethod;
  private final String httpPath;
  private final boolean stdio;
  private final Set<Access> access;
  private final RequestBody requestBody;
  private final ResponseBody responseBody;
  private final Class<? extends Record> requestType;

  Command(
      String wireName,
      String httpMethod,
      String httpPath,
      boolean stdio,
      EnumSet<Access> access,
      RequestBody requestBody,
      ResponseBody responseBody,
      Class<? extends Record> requestType) {
    this.wireName = wireName;
    this.httpMethod = httpMethod;
    this.httpPath = httpPath;
    this.stdio = stdio;
    this.access = Collections.unmodifiableSet(access);
    this.requestBody = requestBody;
    this.responseBody = responseBody;
    this.requestType = requestType;
  }

  public String wireName() {
    return wireName;
  }

  /** HTTP の方法。デスクトップだけの操作は null。 */
  public String httpMethod() {
    return httpMethod;
  }

  /** HTTP の道の型（{名前} は要求の欄）。デスクトップだけの操作は null。 */
  public String httpPath() {
    return httpPath;
  }

  /** stdio で受けるか。 */
  public boolean stdio() {
    return stdio;
  }

  /** 入口（並びは Access の宣言の順）。 */
  public Set<Access> access() {
    return access;
  }

  public RequestBody requestBody() {
    return requestBody;
  }

  public ResponseBody responseBody() {
    return responseBody;
  }

  public Class<? extends Record> requestType() {
    return requestType;
  }

  public static Optional<Command> byWireName(String name) {
    return Optional.ofNullable(BY_WIRE_NAME.get(name));
  }
}
