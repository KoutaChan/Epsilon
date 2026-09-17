package com.epsilon.reviewer;

import com.epsilon.reviewer.engine.ReviewEngine;
import com.epsilon.reviewer.model.ModelCatalog;
import com.google.gson.JsonParseException;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.file.Path;
import java.time.Instant;
import java.util.concurrent.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** サーバー資源の作成・解放と、HTTPリクエスト処理中に発生した例外への応答を管理する。 */
public final class ReviewerServer implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(ReviewerServer.class);
  private final HttpServer server;
  private final ExecutorService requests;
  private final RecordFetcher fetcher;
  private final ReviewService service;
  private final ReviewerHttp http;
  private final ReviewerApi api;
  private final DataDirectoryLease directoryLease;

  public ReviewerServer(
      InetSocketAddress bind, URI publicOrigin, Path models, Path dataDirectory, Path webDirectory)
      throws IOException {
    directoryLease = new DataDirectoryLease(dataDirectory);
    requests = Executors.newFixedThreadPool(8);
    fetcher = new RecordFetcher();
    HttpServer created = null;
    try {
      var catalog = new ModelCatalog(models);
      var store = new ResultStore(dataDirectory.resolve("results"));
      store.cleanTemporary(Instant.now());
      Path snapshots = dataDirectory.resolve("temporary-models");
      ModelCatalog.cleanSnapshots(snapshots);
      created = HttpServer.create(bind, 64);
      server = created;
      service = new ReviewService(new ReviewEngine(), catalog, store, snapshots);
      http = new ReviewerHttp(publicOrigin, webDirectory, this::port);
      api = new ReviewerApi(catalog, store, service, fetcher, http);
      server.setExecutor(requests);
      server.createContext("/", this::handle);
    } catch (IOException | RuntimeException failure) {
      if (created != null) created.stop(0);
      requests.shutdownNow();
      fetcher.close();
      directoryLease.close();
      throw failure;
    }
  }

  public void start() {
    server.start();
  }

  public int port() {
    return server.getAddress().getPort();
  }

  private void handle(HttpExchange exchange) throws IOException {
    String path = exchange.getRequestURI().getPath();
    boolean sharedApi = ReviewerHttp.isSharedResultApi(path);
    try {
      http.prepare(exchange);
      if (sharedApi) api.readSharedResult(exchange);
      else if (path.startsWith("/api/")) api.handle(exchange, http.owner(exchange));
      else http.serveUi(exchange, path);
    } catch (ApiException failure) {
      ReviewerHttp.error(exchange, failure.status(), failure.code(), failure.getMessage());
    } catch (IllegalArgumentException | JsonParseException failure) {
      ReviewerHttp.error(exchange, 400, "invalid_input", "Invalid request input.");
    } catch (InterruptedException failure) {
      Thread.currentThread().interrupt();
      ReviewerHttp.error(exchange, 503, "interrupted", "Request processing was interrupted.");
    } catch (Exception failure) {
      if (sharedApi) {
        // 共有 URL と、結果 ID を含み得るファイル例外をログへ出さない。
        LOG.error("Could not read shared analysis ({}).", failure.getClass().getSimpleName());
      } else {
        LOG.error("Could not process {} {}.", exchange.getRequestMethod(), path, failure);
      }
      ReviewerHttp.error(exchange, 500, "server_error", "Request processing failed.");
    } finally {
      exchange.close();
    }
  }

  @Override
  public void close() {
    server.stop(1);
    requests.shutdown();
    boolean interrupted = false;
    while (!requests.isTerminated()) {
      try {
        requests.awaitTermination(1, TimeUnit.SECONDS);
      } catch (InterruptedException failure) {
        interrupted = true;
      }
    }
    service.close();
    fetcher.close();
    try {
      directoryLease.close();
    } catch (IOException failure) {
      LOG.error("Could not release the reviewer data directory lock.", failure);
    }
    if (interrupted) Thread.currentThread().interrupt();
  }
}
