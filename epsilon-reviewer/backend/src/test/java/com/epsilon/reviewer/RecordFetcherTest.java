package com.epsilon.reviewer;

import com.sun.net.httpserver.HttpServer;
import java.io.*;
import java.net.*;
import java.net.http.*;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.zip.GZIPOutputStream;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** Tests supported record inputs, original upload bytes, and download size limits. */
public final class RecordFetcherTest {
  @DataProvider
  public Object[][] mahjongSoulHosts() {
    return new Object[][] {
      {"game.mahjongsoul.com"},
      {"mahjongsoul.game.yo-star.com"},
      {"game.maj-soul.com"},
      {"game.maj-soul.net"}
    };
  }

  @Test(dataProvider = "mahjongSoulHosts")
  public void mahjongSoulUrlsAreRejectedBeforeAnyNetworkRequest(String host) {
    try (var fetcher = new RecordFetcher()) {
      for (String scheme : List.of("http", "https")) {
        String url =
            scheme + "://" + host + "/?paipu=260915-12345678-1234-1234-1234-123456789abc_a123";
        var failure = Assert.expectThrows(ApiException.class, () -> fetcher.fetch(url));
        Assert.assertEquals(failure.status(), 400);
        Assert.assertEquals(failure.code(), "unsupported_url");
      }
    }
  }

  @Test
  public void compressedUploadsKeepOriginalBytesAndNameForTheSharedReader() throws Exception {
    byte[] record = "{\"type\":\"start_game\"}\n".getBytes(StandardCharsets.UTF_8);
    var compressed = new ByteArrayOutputStream();
    try (var gzip = new GZIPOutputStream(compressed)) {
      gzip.write(record);
    }
    byte[] bytes = compressed.toByteArray();
    var uploaded = RecordFetcher.uploaded(bytes, "game.mjson.gz");
    Assert.assertSame(uploaded.bytes(), bytes);
    Assert.assertEquals(uploaded.fileName(), "game.mjson.gz");
    Assert.assertEquals(uploaded.source(), "auto");
  }

  @Test
  public void uploadNamesAndRecognizableResultDocumentsAreRejected() {
    byte[] record = "{\"type\":\"start_game\"}\n".getBytes(StandardCharsets.UTF_8);
    Assert.expectThrows(
        ApiException.class, () -> RecordFetcher.uploaded(record, "game.epsilon-reviewer.json"));
    Assert.expectThrows(
        ApiException.class, () -> RecordFetcher.uploaded(record, "game.epsilon-reviewer.json.gz"));
    Assert.expectThrows(ApiException.class, () -> RecordFetcher.uploaded(record, "../game.mjson"));
    Assert.expectThrows(
        ApiException.class,
        () ->
            RecordFetcher.uploaded(
                "{\"formatVersion\":1}".getBytes(StandardCharsets.UTF_8), "renamed.mjson"));
  }

  @Test
  public void httpDownloadRejectsBodyThatExceedsLimitWhileReceiving() throws Exception {
    HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
    server.createContext(
        "/",
        exchange -> {
          exchange.sendResponseHeaders(200, 0);
          try (var out = exchange.getResponseBody()) {
            out.write(new byte[100]);
          }
        });
    server.start();
    try (var client = HttpClient.newHttpClient()) {
      var request =
          HttpRequest.newBuilder(URI.create("http://127.0.0.1:" + server.getAddress().getPort()))
              .build();
      Assert.expectThrows(ApiException.class, () -> BoundedHttp.get(client, request, 10));
      Assert.assertEquals(BoundedHttp.get(client, request, 100).body().length, 100);
    } finally {
      server.stop(0);
    }
  }
}
