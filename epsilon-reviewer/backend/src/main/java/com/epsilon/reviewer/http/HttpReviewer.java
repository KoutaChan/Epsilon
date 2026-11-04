package com.epsilon.reviewer.http;

import com.epsilon.reviewer.Reviewer;
import com.epsilon.reviewer.api.Dispatcher;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.URI;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** JDK の HTTP のサーバ（要求のスレッド 8 本 reviewer-http-&lt;n&gt;、backlog 64）。 */
public final class HttpReviewer implements AutoCloseable {
  private static final int THREADS = 8;
  private static final int BACKLOG = 64;

  private final HttpServer server;
  private final ExecutorService requests;

  private HttpReviewer(HttpServer server, ExecutorService requests) {
    this.server = server;
    this.requests = requests;
  }

  /** 静的ファイルの索引を作り、結んで受け始める。 */
  public static HttpReviewer start(ServerConfig config, Reviewer reviewer) throws IOException {
    StaticFiles files = StaticFiles.index(config.webDir());
    HttpServer server =
        HttpServer.create(new InetSocketAddress(config.host(), config.port()), BACKLOG);
    int port = server.getAddress().getPort();
    URI origin = config.origin();
    String authority = origin.getRawAuthority();
    if (authority.endsWith(":0"))
      authority = authority.substring(0, authority.length() - 1) + port;
    var dispatcher = new Dispatcher(reviewer, URI.create(origin.getScheme() + "://" + authority));
    var router =
        new HttpRouter(
            dispatcher,
            new RequestGuard(origin.getScheme(), authority, port),
            new OwnerCookie(origin.getScheme().equals("https")),
            files);
    var counter = new AtomicInteger();
    ExecutorService requests =
        Executors.newFixedThreadPool(
            THREADS, task -> new Thread(task, "reviewer-http-" + counter.incrementAndGet()));
    server.setExecutor(requests);
    server.createContext("/", router);
    server.start();
    return new HttpReviewer(server, requests);
  }

  /** 結んだポート。 */
  public int port() {
    return server.getAddress().getPort();
  }

  /** 受けるのを止め、走る要求を待つ。 */
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
    if (interrupted) Thread.currentThread().interrupt();
  }
}
