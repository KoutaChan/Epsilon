package com.epsilon.reviewer.http;

import com.epsilon.reviewer.TestReviewer;
import com.epsilon.reviewer.store.ResultId;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 文書の ETag と 304、gzip の交渉（Accept-Encoding の q と *）、ダウンロードの頭。 */
public class DocumentCachingTest {
  @Test
  public void documentsAreSentAsStoredOrInflatedAndRevalidatedByTheResultId() throws Exception {
    try (var server = TestServer.start()) {
      ResultId id = server.reviewer.analyze(TestReviewer.OWNER, "kakan-a", TestReviewer.FIXTURE);
      Path file =
          server.reviewer.directory.resolve(
              Path.of("data", "results", id.value(), "result.json.gz"));
      byte[] stored = Files.readAllBytes(file);
      byte[] inflated = HttpApiTest.gunzip(stored).getBytes(StandardCharsets.UTF_8);
      String tag = "W/\"" + id.value() + "\"";
      for (String path : List.of("/api/results/" + id.value(), "/api/shares/" + id.value())) {
        var owner = path.startsWith("/api/shares/") ? null : TestReviewer.OWNER;
        for (String encoding : List.of("gzip", "br, gzip;q=0.5", "*;q=1", "GZIP")) {
          var response =
              server.bytes(server.request(path, owner).header("Accept-Encoding", encoding));
          Assert.assertEquals(response.statusCode(), 200);
          Assert.assertEquals(response.body(), stored, encoding);
          Assert.assertEquals(header(response, "Content-Encoding"), "gzip");
          Assert.assertEquals(
              response.headers().firstValueAsLong("Content-Length").orElseThrow(), stored.length);
          Assert.assertEquals(header(response, "Vary"), "Accept-Encoding");
          Assert.assertEquals(response.headers().firstValue("ETag").orElseThrow(), tag);
          Assert.assertEquals(
              response.headers().firstValue("Cache-Control").orElseThrow(), "private, no-cache");
        }
        for (String encoding : List.of("identity", "br", "gzip;q=0, *;q=1", "gzip;q=0.000")) {
          var response =
              server.bytes(server.request(path, owner).header("Accept-Encoding", encoding));
          Assert.assertEquals(response.statusCode(), 200);
          Assert.assertEquals(response.body(), inflated, encoding);
          Assert.assertFalse(response.headers().firstValue("Content-Encoding").isPresent());
          Assert.assertFalse(response.headers().firstValue("Content-Length").isPresent());
          Assert.assertEquals(header(response, "Vary"), "Accept-Encoding");
        }
        for (String condition : List.of(tag, "\"" + id.value() + "\"", "*", "\"x\", " + tag)) {
          var response =
              server.bytes(server.request(path, owner).header("If-None-Match", condition));
          Assert.assertEquals(response.statusCode(), 304, condition);
          Assert.assertEquals(response.body().length, 0);
          Assert.assertEquals(response.headers().firstValue("ETag").orElseThrow(), tag);
        }
        var changed = server.request(path, owner).header("If-None-Match", "W/\"other\"");
        Assert.assertEquals(server.bytes(changed).statusCode(), 200);
      }
      for (String encoding : List.of("identity", "gzip")) {
        var download =
            server.bytes(
                server
                    .request("/api/results/" + id.value() + "/download", TestReviewer.OWNER)
                    .header("Accept-Encoding", encoding));
        Assert.assertEquals(download.statusCode(), 200);
        Assert.assertEquals(download.body(), stored);
        Assert.assertEquals(
            download.headers().firstValue("Content-Type").orElseThrow(), "application/gzip");
        Assert.assertFalse(download.headers().firstValue("Content-Encoding").isPresent());
        Assert.assertEquals(
            download.headers().firstValue("Content-Disposition").orElseThrow(),
            "attachment; filename=\"" + id.value() + ".epsilon-reviewer.json.gz\"");
        Assert.assertEquals(
            download.headers().firstValueAsLong("Content-Length").orElseThrow(), stored.length);
      }
    }
  }

  private static String header(HttpResponse<byte[]> response, String name) {
    return response.headers().firstValue(name).orElseThrow();
  }
}
