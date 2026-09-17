package com.epsilon.reviewer;

import static com.epsilon.reviewer.ReviewerHttp.*;

import com.epsilon.reviewer.RecordFetcher.SourceRecord;
import com.epsilon.reviewer.model.ModelCatalog;
import com.google.gson.*;
import com.sun.net.httpserver.HttpExchange;
import java.io.IOException;
import java.net.URLDecoder;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/** HTTP の経路を型付きの解析サービスへ接続する。 */
final class ReviewerApi {
  private final ModelCatalog catalog;
  private final ResultStore store;
  private final ReviewService service;
  private final RecordFetcher fetcher;
  private final ReviewerHttp http;

  ReviewerApi(
      ModelCatalog catalog,
      ResultStore store,
      ReviewService service,
      RecordFetcher fetcher,
      ReviewerHttp http) {
    this.catalog = catalog;
    this.store = store;
    this.service = service;
    this.fetcher = fetcher;
    this.http = http;
  }

  void readSharedResult(HttpExchange exchange) throws IOException {
    String path = exchange.getRequestURI().getPath();
    if (!exchange.getRequestMethod().equals("GET") || !path.startsWith("/api/shares/"))
      throw new ApiException(404, "share_not_found", "Shared analysis not found.");
    try (var content = store.openSharedResult(path.substring("/api/shares/".length()))) {
      streamResult(exchange, content, null);
    }
  }

  void handle(HttpExchange exchange, String owner) throws IOException, InterruptedException {
    String path = exchange.getRequestURI().getPath();
    String method = exchange.getRequestMethod();
    var headers = exchange.getResponseHeaders();
    if (path.equals("/api/models") && method.equals("GET")) {
      json(exchange, 200, catalog.publicCatalog());
      return;
    }
    if (path.equals("/api/history") && method.equals("GET")) {
      json(exchange, 200, Map.of("results", store.history(owner)));
      return;
    }
    if (path.equals("/api/records") && method.equals("POST")) {
      SourceRecord source;
      String contentType = exchange.getRequestHeaders().getFirst("Content-Type");
      if (contentType != null
          && contentType.toLowerCase(Locale.ROOT).startsWith("application/json")) {
        JsonObject request = requestJson(exchange);
        if (request.has("url") && request.has("resultId"))
          throw new ApiException(
              400, "invalid_input", "Specify exactly one URL or stored record reference.");
        if (request.has("url")) source = fetcher.fetch(readRequiredString(request, "url"));
        else if (request.has("resultId") && request.size() == 1)
          source = store.source(owner, readRequiredString(request, "resultId"));
        else
          throw new ApiException(
              415,
              "result_import_disabled",
              "Analysis result import is disabled; provide the original record URL or file.");
      } else if (contentType != null
          && contentType.toLowerCase(Locale.ROOT).startsWith("application/octet-stream")) {
        String fileName = exchange.getRequestHeaders().getFirst("X-File-Name");
        if (fileName != null) fileName = URLDecoder.decode(fileName, StandardCharsets.UTF_8);
        byte[] bytes =
            RecordFetcher.readBounded(exchange.getRequestBody(), RecordFetcher.MAX_UPLOAD);
        source = RecordFetcher.uploaded(bytes, fileName);
      } else
        throw new ApiException(
            415, "unsupported_content_type", "Expected an original record file.");
      json(exchange, 201, service.prepare(owner, source));
      return;
    }
    if (path.equals("/api/analyses") && method.equals("POST")) {
      JsonObject body = requestJson(exchange);
      String storage = body.has("storage") ? readRequiredString(body, "storage") : "web";
      if (!Set.of("web", "desktop").contains(storage))
        throw new ApiException(400, "invalid_storage", "Storage must be web or desktop.");
      json(
          exchange,
          202,
          service.analyze(
              owner,
              readRequiredString(body, "recordId"),
              readRequiredString(body, "modelId"),
              readRequiredString(body, "revision"),
              storage.equals("desktop")));
      return;
    }
    String[] route = path.split("/");
    if (route.length >= 4 && route[2].equals("jobs")) {
      if (route.length == 4 && method.equals("GET")) {
        json(exchange, 200, service.job(owner, route[3]));
        return;
      }
      if (route.length == 5 && route[4].equals("cancel") && method.equals("POST")) {
        json(exchange, 200, service.cancel(owner, route[3]));
        return;
      }
    }
    if (route.length >= 4 && route[2].equals("results")) {
      String id = route[3];
      if (route.length == 5 && route[4].equals("share") && method.equals("GET")) {
        json(
            exchange,
            200,
            Map.of("url", http.sharedResultUrl(store.requireShareableResultId(owner, id))));
        return;
      }
      if (route.length == 4 && method.equals("DELETE")) {
        store.delete(owner, id);
        exchange.sendResponseHeaders(204, -1);
        return;
      }
      if (route.length == 5 && route[4].equals("source") && method.equals("GET")) {
        SourceRecord source = store.source(owner, id);
        headers.set("X-File-Name", URLEncoder.encode(source.fileName(), StandardCharsets.UTF_8));
        headers.set("X-Record-Source", source.source());
        bytes(exchange, 200, "application/octet-stream", source.bytes());
        return;
      }
      if (method.equals("GET")
          && (route.length == 4 || route.length == 5 && route[4].equals("export"))) {
        try (var content = store.openResult(owner, id)) {
          streamResult(
              exchange, content, route.length == 5 ? id + ".epsilon-reviewer.json.gz" : null);
        }
        return;
      }
    }
    throw new ApiException(404, "not_found", "API route not found.");
  }
}
