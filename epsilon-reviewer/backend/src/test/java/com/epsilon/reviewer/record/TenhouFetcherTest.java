package com.epsilon.reviewer.record;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.zip.GZIPOutputStream;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * 手元で動かす HTTP のホストから牌譜を取るテスト。確かめるのは、ステータスからエラーコードへの対応、gzip の展開、リダイレクトの扱い、
 * 本文の上限、繋がらないホスト。
 */
public class TenhouFetcherTest {
  private static final String PREFIX = "2026091512gm-0089-0000-0000000";
  private static final byte[] RECORD = "<mjloggm ver=\"2.3\"/>".getBytes(StandardCharsets.UTF_8);
  private HttpServer server;
  private TenhouFetcher fetcher;

  @BeforeClass
  public void start() throws IOException {
    server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    server.createContext("/0/log/", this::answer);
    server.start();
    URI origin = URI.create("http://127.0.0.1:" + server.getAddress().getPort());
    fetcher = new TenhouFetcher(origin, false);
  }

  @AfterClass(alwaysRun = true)
  public void stop() {
    fetcher.close();
    server.stop(0);
  }

  @Test
  public void bodiesArriveAsRecordsAndGzipIsInflated() {
    Assert.assertEquals(fetcher.fetch(id(1)), RECORD);
    Assert.assertEquals(fetcher.fetch(id(2)), RECORD);
    Assert.assertEquals(fetcher.fetch(id(6)), RECORD, "a same-host redirect is followed");
  }

  @Test
  public void statusesAndRedirectsMapToStableCodes() {
    refused(3, ErrorCode.RECORD_NOT_FOUND);
    refused(4, ErrorCode.RECORD_ACCESS_DENIED);
    refused(5, ErrorCode.FETCH_FAILED);
    refused(7, ErrorCode.FETCH_FAILED);
    refused(8, ErrorCode.FETCH_FAILED);
    refused(9, ErrorCode.RECORD_TOO_LARGE);
    refused(0, ErrorCode.FETCH_FAILED);
  }

  @Test
  public void anUnreachableHostFailsToFetch() throws IOException {
    int port;
    try (var closed = new ServerSocket(0, 1, InetAddress.getLoopbackAddress())) {
      port = closed.getLocalPort();
    }
    try (var unreachable = new TenhouFetcher(URI.create("http://127.0.0.1:" + port), false)) {
      ReviewerException failure =
          Assert.expectThrows(ReviewerException.class, () -> unreachable.fetch(id(1)));
      Assert.assertEquals(failure.code(), ErrorCode.FETCH_FAILED);
    }
    try (var tenhou = new TenhouFetcher(URI.create("http://127.0.0.1:" + port), true)) {
      ReviewerException failure =
          Assert.expectThrows(ReviewerException.class, () -> tenhou.fetch(id(1)));
      Assert.assertEquals(failure.code(), ErrorCode.FETCH_FAILED, "loopback is not public");
    }
  }

  /** 牌譜の id の最後の数字で、返す応答を選ぶ。 */
  private void answer(HttpExchange exchange) throws IOException {
    String query = exchange.getRequestURI().getRawQuery();
    int kind = Character.digit(query.charAt(query.length() - 1), 16);
    String self = "/0/log/?" + query;
    switch (kind) {
      case 1 -> send(exchange, 200, RECORD);
      case 2 -> {
        Assert.assertEquals(exchange.getRequestHeaders().getFirst("Accept-Encoding"), "gzip");
        exchange.getResponseHeaders().set("Content-Encoding", "gzip");
        send(exchange, 200, gzip(RECORD));
      }
      case 3 -> send(exchange, 404, new byte[0]);
      case 4 -> send(exchange, 403, new byte[0]);
      case 5 -> send(exchange, 500, new byte[0]);
      case 6 -> redirect(exchange, "/0/log/?" + PREFIX + "1");
      case 7 -> redirect(exchange, "http://other.example/0/log/?" + PREFIX + "1");
      case 8 -> redirect(exchange, self);
      case 9 -> {
        exchange.sendResponseHeaders(200, 0);
        try (OutputStream out = exchange.getResponseBody()) {
          byte[] chunk = new byte[1024 * 1024];
          for (int i = 0; i <= 16; i++) out.write(chunk);
        } catch (IOException ignored) {
          // 取得する側が上限に達した所で接続を切る。
        }
      }
      default -> {
        exchange.sendResponseHeaders(302, -1);
        exchange.close();
      }
    }
  }

  private static void redirect(HttpExchange exchange, String location) throws IOException {
    exchange.getResponseHeaders().set("Location", location);
    send(exchange, 302, new byte[0]);
  }

  private static void send(HttpExchange exchange, int status, byte[] body) throws IOException {
    exchange.sendResponseHeaders(status, body.length == 0 ? -1 : body.length);
    try (OutputStream out = exchange.getResponseBody()) {
      out.write(body);
    }
  }

  private static byte[] gzip(byte[] bytes) throws IOException {
    var out = new ByteArrayOutputStream();
    try (var gzip = new GZIPOutputStream(out)) {
      gzip.write(bytes);
    }
    return out.toByteArray();
  }

  private static TenhouLogId id(int kind) {
    return new TenhouLogId(PREFIX + Integer.toHexString(kind));
  }

  private void refused(int kind, ErrorCode code) {
    ReviewerException failure =
        Assert.expectThrows(ReviewerException.class, () -> fetcher.fetch(id(kind)));
    Assert.assertEquals(failure.code(), code, "kind " + kind + ": " + failure.getMessage());
  }
}
