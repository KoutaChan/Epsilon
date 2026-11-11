package com.epsilon.reviewer.http;

import com.epsilon.reviewer.StoredDocument;
import com.epsilon.reviewer.api.Access;
import com.epsilon.reviewer.api.Caller;
import com.epsilon.reviewer.api.Command;
import com.epsilon.reviewer.api.Dispatcher;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.http.RouteTable.Route;
import com.epsilon.reviewer.json.Json;
import com.epsilon.reviewer.json.StrictRecords;
import com.epsilon.reviewer.record.RecordImporter;
import com.google.gson.JsonObject;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.util.Locale;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 全てのリクエストを受けるハンドラー。ヘッダーを確かめてから、API と静的ファイルに振り分ける。API はルートの表で操作を探し、本文を
 * 読んで Dispatcher に渡し、応答を書く。Cookie を読む（無ければ発行する）のはオーナーのルートだけで、ヘッダーを確かめた後に行う。
 * 共有リンクの API で失敗したときは、共有の id を残さないよう、ログには例外のクラス名だけを書く。
 */
final class HttpRouter implements HttpHandler {
  private static final Logger LOG = LoggerFactory.getLogger(HttpRouter.class);

  /** JSON のリクエスト本文の上限（バイト）。 */
  static final int JSON_LIMIT = 16 * 1024;

  private final Dispatcher dispatcher;
  private final RequestGuard guard;
  private final OwnerCookie cookies;
  private final RouteTable routes = new RouteTable();
  private final StaticFiles files;

  HttpRouter(Dispatcher dispatcher, RequestGuard guard, OwnerCookie cookies, StaticFiles files) {
    this.dispatcher = dispatcher;
    this.guard = guard;
    this.cookies = cookies;
    this.files = files;
  }

  @Override
  public void handle(HttpExchange exchange) {
    var replies = new HttpReplies(exchange);
    URI uri = exchange.getRequestURI();
    String path = uri.getPath();
    try {
      guard.writeHeaders(uri, exchange.getResponseHeaders());
      guard.require(
          exchange.getRequestMethod(),
          uri,
          exchange.getRequestHeaders(),
          exchange.getRemoteAddress().getAddress().isLoopbackAddress());
      if (path.startsWith("/api/")) api(exchange, replies, uri.getRawPath());
      else files.serve(exchange, replies, path);
    } catch (ReviewerException failure) {
      fail(replies, failure.code(), failure.getMessage());
    } catch (Exception failure) {
      if (RequestGuard.isShareApi(path))
        LOG.error("Could not serve a shared analysis ({}).", failure.getClass().getSimpleName());
      else if (replies.committed())
        LOG.warn(
            "The response to {} {} failed after it started: {}",
            exchange.getRequestMethod(),
            path,
            failure.toString());
      else LOG.error("Could not process {} {}.", exchange.getRequestMethod(), path, failure);
      fail(replies, ErrorCode.SERVER_ERROR, ErrorCode.SERVER_ERROR.message());
    } finally {
      exchange.close();
    }
  }

  private void api(HttpExchange exchange, HttpReplies replies, String rawPath) throws IOException {
    Route route =
        routes
            .match(exchange.getRequestMethod(), rawPath)
            .orElseThrow(() -> new ReviewerException(ErrorCode.NOT_FOUND));
    Command command = route.command();
    Caller caller =
        command.access().contains(Access.OWNER)
            ? new Caller.Owner(cookies.resolve(exchange))
            : new Caller.Visitor();
    JsonObject fields = new JsonObject();
    byte[] body = null;
    switch (command.requestBody()) {
      case NONE -> {}
      case JSON -> {
        requireType(exchange, "application/json");
        byte[] json = read(exchange, JSON_LIMIT, ErrorCode.INVALID_REQUEST);
        fields = Json.readObject(json, JSON_LIMIT);
      }
      case BINARY -> {
        requireType(exchange, "application/octet-stream");
        body = read(exchange, RecordImporter.RECORD_LIMIT, ErrorCode.RECORD_TOO_LARGE);
      }
    }
    for (Map.Entry<String, String> parameter : route.parameters().entrySet()) {
      if (fields.has(parameter.getKey()))
        throw new ReviewerException(
            ErrorCode.INVALID_REQUEST,
            "The request has an unknown field: " + parameter.getKey() + ".");
      fields.addProperty(parameter.getKey(), parameter.getValue());
    }
    Record request = StrictRecords.read(fields, command.requestType());
    Object response = dispatcher.dispatch(command, caller, request, body);
    switch (command.responseBody()) {
      case JSON -> replies.json(Json.write(response));
      case EMPTY -> replies.empty();
      case DOCUMENT -> replies.document((StoredDocument) response);
      case ATTACHMENT -> replies.attachment((StoredDocument) response);
    }
  }

  private static void requireType(HttpExchange exchange, String type) {
    String header = exchange.getRequestHeaders().getFirst("Content-Type");
    if (header == null || !header.toLowerCase(Locale.ROOT).startsWith(type))
      throw new ReviewerException(ErrorCode.INVALID_REQUEST, "Expected " + type + " content.");
  }

  /** 本文を上限まで読む。Content-Length があれば、その大きさの配列へ一度に読む。上限を超えたら {@code tooLarge} の例外を投げる。 */
  private static byte[] read(HttpExchange exchange, int limit, ErrorCode tooLarge)
      throws IOException {
    String declared = exchange.getRequestHeaders().getFirst("Content-Length");
    try (InputStream in = exchange.getRequestBody()) {
      if (declared == null) {
        byte[] body = in.readNBytes(limit + 1);
        if (body.length > limit) throw new ReviewerException(tooLarge);
        return body;
      }
      long length;
      try {
        length = Long.parseLong(declared.trim());
      } catch (NumberFormatException failure) {
        throw new ReviewerException(ErrorCode.INVALID_REQUEST, "Invalid Content-Length.");
      }
      if (length > limit) throw new ReviewerException(tooLarge);
      byte[] body = new byte[(int) length];
      if (in.readNBytes(body, 0, body.length) != body.length)
        throw new ReviewerException(ErrorCode.INVALID_REQUEST, "The request body ended early.");
      return body;
    }
  }

  private static void fail(HttpReplies replies, ErrorCode code, String message) {
    try {
      if (!replies.error(code, message))
        LOG.debug("The response had started; the {} error was not written.", code.code());
    } catch (IOException failure) {
      LOG.debug("Could not write the {} error: {}", code.code(), failure.toString());
    }
  }
}
