package com.epsilon.reviewer.http;

import com.epsilon.reviewer.TestReviewer;
import com.epsilon.reviewer.store.OwnerId;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.Map;
import org.testng.Assert;

/** テスト用のモデルを使う Reviewer を、127.0.0.1 の空いたポートで動かす HTTP のサーバ。 */
final class TestServer implements AutoCloseable {
  /** {@code TestReviewer.OWNER} の Cookie。 */
  static final String OWNER_COOKIE = OwnerCookie.NAME + "=" + "A".repeat(43);

  /** {@code TestReviewer.OTHER} の Cookie。 */
  static final String OTHER_COOKIE = OwnerCookie.NAME + "=" + "B".repeat(43);

  static final Map<OwnerId, String> COOKIES =
      Map.of(TestReviewer.OWNER, OWNER_COOKIE, TestReviewer.OTHER, OTHER_COOKIE);

  final TestReviewer reviewer;
  final HttpReviewer server;
  final HttpClient client = HttpClient.newHttpClient();

  private TestServer(TestReviewer reviewer, HttpReviewer server) {
    this.reviewer = reviewer;
    this.server = server;
  }

  static TestServer start() {
    return start(URI.create("http://127.0.0.1:0"), null);
  }

  /** @param web 画面のファイルを置いたフォルダー。無ければ null */
  static TestServer start(URI origin, Path web) {
    TestReviewer reviewer = TestReviewer.open();
    try {
      return new TestServer(
          reviewer,
          HttpReviewer.start(
              new ServerConfig("127.0.0.1", 0, origin, web), reviewer.reviewer));
    } catch (IOException failure) {
      reviewer.close();
      throw new UncheckedIOException(failure);
    }
  }

  URI base() {
    return URI.create("http://127.0.0.1:" + server.port());
  }

  /** @param owner Cookie を付けるオーナー。無ければ null */
  HttpRequest.Builder request(String path, OwnerId owner) {
    var request = HttpRequest.newBuilder(base().resolve(path));
    if (owner != null) request.header("Cookie", COOKIES.get(owner));
    return request;
  }

  HttpRequest.Builder json(String path, OwnerId owner, String body) {
    return request(path, owner)
        .header("Content-Type", "application/json")
        .POST(HttpRequest.BodyPublishers.ofString(body));
  }

  HttpResponse<String> send(HttpRequest.Builder request) {
    try {
      return client.send(request.build(), HttpResponse.BodyHandlers.ofString());
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    } catch (InterruptedException failure) {
      Thread.currentThread().interrupt();
      throw new AssertionError(failure);
    }
  }

  HttpResponse<byte[]> bytes(HttpRequest.Builder request) {
    try {
      return client.send(request.build(), HttpResponse.BodyHandlers.ofByteArray());
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    } catch (InterruptedException failure) {
      Thread.currentThread().interrupt();
      throw new AssertionError(failure);
    }
  }

  static JsonObject body(HttpResponse<String> response) {
    return JsonParser.parseString(response.body()).getAsJsonObject();
  }

  static void assertError(HttpResponse<String> response, int status, String code) {
    Assert.assertEquals(response.statusCode(), status, response.body());
    Assert.assertEquals(
        body(response).getAsJsonObject("error").get("code").getAsString(), code, response.body());
  }

  @Override
  public void close() {
    server.close();
    client.close();
    reviewer.close();
  }
}
