package com.epsilon.reviewer.http;

import com.epsilon.reviewer.TestFiles;
import java.io.ByteArrayOutputStream;
import java.net.URI;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.List;
import java.util.zip.GZIPOutputStream;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * 画面のファイルを配るテスト。確かめるのは、事前に圧縮した写しの選び方と古い写しの扱い、キャッシュのヘッダー、ETag と 304、SPA、
 * 点を含むパスの 404、HEAD、シンボリックリンク。
 */
public class StaticFilesTest {
  private static final String SCRIPT = "console.log('epsilon');".repeat(100);
  private Path web;
  private Path outside;
  private TestServer server;
  private byte[] gzip;

  @BeforeClass
  public void start() throws Exception {
    web = TestFiles.directory("reviewer-static-");
    outside = TestFiles.directory("reviewer-outside-");
    Files.createDirectories(web.resolve("assets"));
    Files.createDirectories(web.resolve("tiles"));
    Files.writeString(web.resolve("index.html"), "<main>index</main>");
    Files.writeString(web.resolve("assets/app.js"), SCRIPT);
    Files.writeString(web.resolve("assets/app.js.br"), "BROTLI");
    var bytes = new ByteArrayOutputStream();
    try (var out = new GZIPOutputStream(bytes)) {
      out.write(SCRIPT.getBytes(StandardCharsets.UTF_8));
    }
    gzip = bytes.toByteArray();
    Files.write(web.resolve("assets/app.js.gz"), gzip);
    Files.writeString(web.resolve("assets/rebuilt.js"), SCRIPT);
    Path stale = Files.writeString(web.resolve("assets/rebuilt.js.br"), "STALE");
    Files.setLastModifiedTime(
        stale,
        FileTime.fromMillis(
            Files.getLastModifiedTime(web.resolve("assets/rebuilt.js")).toMillis() - 60_000));
    Files.writeString(web.resolve("tiles/a.svg"), "<svg/>");
    Files.writeString(web.resolve("data.json"), "{}");
    Files.writeString(web.resolve("notes.md"), "# notes");
    Files.writeString(outside.resolve("secret.txt"), "secret");
    Files.createSymbolicLink(web.resolve("secret.txt"), outside.resolve("secret.txt"));
    server = TestServer.start(URI.create("http://127.0.0.1:0"), web);
  }

  @AfterClass(alwaysRun = true)
  public void stop() {
    server.close();
    TestFiles.delete(web);
    TestFiles.delete(outside);
  }

  @Test
  public void precompressedCopiesFollowAcceptEncoding() {
    var brotli = server.bytes(asset().header("Accept-Encoding", "gzip, br"));
    Assert.assertEquals(new String(brotli.body(), StandardCharsets.UTF_8), "BROTLI");
    Assert.assertEquals(brotli.headers().firstValue("Content-Encoding").orElseThrow(), "br");
    var gzipped = server.bytes(asset().header("Accept-Encoding", "gzip"));
    Assert.assertEquals(gzipped.body(), gzip);
    Assert.assertEquals(gzipped.headers().firstValue("Content-Encoding").orElseThrow(), "gzip");
    var plain = server.bytes(asset());
    Assert.assertEquals(new String(plain.body(), StandardCharsets.UTF_8), SCRIPT);
    Assert.assertFalse(plain.headers().firstValue("Content-Encoding").isPresent());
    for (HttpResponse<byte[]> response : List.of(brotli, gzipped, plain)) {
      Assert.assertEquals(response.headers().firstValue("Vary").orElseThrow(), "Accept-Encoding");
      Assert.assertEquals(
          response.headers().firstValue("Cache-Control").orElseThrow(),
          "public, max-age=31536000, immutable");
      Assert.assertEquals(
          response.headers().firstValue("Content-Type").orElseThrow(),
          "text/javascript; charset=utf-8");
    }
    String plainTag = plain.headers().firstValue("ETag").orElseThrow();
    Assert.assertTrue(plainTag.matches("\"[0-9a-f]{16}\""), plainTag);
    Assert.assertNotEquals(brotli.headers().firstValue("ETag").orElseThrow(), plainTag);
    Assert.assertEquals(
        server.bytes(asset().header("If-None-Match", plainTag)).statusCode(), 304);
    Assert.assertEquals(
        server
            .bytes(asset().header("If-None-Match", plainTag).header("Accept-Encoding", "br"))
            .statusCode(),
        200);
  }

  /** 前のビルドの写しが残っていても、元のファイルより古ければ送らない。 */
  @Test
  public void copiesOlderThanTheirFileAreIgnored() {
    var response =
        server.bytes(server.request("/assets/rebuilt.js", null).header("Accept-Encoding", "br"));
    Assert.assertEquals(new String(response.body(), StandardCharsets.UTF_8), SCRIPT);
    Assert.assertFalse(response.headers().firstValue("Content-Encoding").isPresent());
  }

  @Test
  public void cacheRulesTypesAndTheSinglePageFallback() {
    var tile = server.bytes(server.request("/tiles/a.svg", null));
    Assert.assertEquals(tile.headers().firstValue("Content-Type").orElseThrow(), "image/svg+xml");
    Assert.assertEquals(
        tile.headers().firstValue("Cache-Control").orElseThrow(), "public, max-age=604800");
    Assert.assertFalse(tile.headers().firstValue("Vary").isPresent());
    for (String path : new String[] {"/", "/history", "/deep/route"}) {
      var page = server.send(server.request(path, null));
      Assert.assertEquals(page.statusCode(), 200, path);
      Assert.assertEquals(page.body(), "<main>index</main>");
      Assert.assertEquals(
          page.headers().firstValue("Content-Type").orElseThrow(), "text/html; charset=utf-8");
      Assert.assertEquals(page.headers().firstValue("Cache-Control").orElseThrow(), "no-cache");
    }
    Assert.assertEquals(type("/data.json"), "application/json");
    Assert.assertEquals(type("/notes.md"), "application/octet-stream");
    for (String path :
        new String[] {"/missing.js", "/assets/app.js.gz", "/secret.txt", "/../x.txt"})
      TestServer.assertError(server.send(server.request(path, null)), 404, "not_found");
    TestServer.assertError(
        server.send(server.request("/", null).POST(HttpRequest.BodyPublishers.noBody())),
        404,
        "not_found");
  }

  @Test
  public void headSendsTheLengthWithoutTheBody() {
    var head = server.bytes(asset().method("HEAD", HttpRequest.BodyPublishers.noBody()));
    Assert.assertEquals(head.statusCode(), 200);
    Assert.assertEquals(head.body().length, 0);
    Assert.assertEquals(
        head.headers().firstValueAsLong("Content-Length").orElseThrow(), SCRIPT.length());
  }

  @Test
  public void anUnconfiguredWebDirectoryAnswersNotFound() {
    try (var bare = TestServer.start()) {
      TestServer.assertError(bare.send(bare.request("/", null)), 404, "not_found");
    }
  }

  private String type(String path) {
    var response = server.send(server.request(path, null));
    return response.headers().firstValue("Content-Type").orElseThrow();
  }

  private HttpRequest.Builder asset() {
    return server.request("/assets/app.js", null);
  }
}
