package com.epsilon.reviewer.http;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.sun.net.httpserver.Headers;
import java.io.InputStream;
import java.net.Socket;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import org.testng.Assert;
import org.testng.annotations.Test;

/** Host・Origin・Sec-Fetch-Site の確かめ、共有リンクの移動の例外、安全の頭。 */
public class RequestGuardTest {
  private static final RequestGuard GUARD = new RequestGuard("http", "reviewer.example:8080", 8080);

  @Test
  public void hostMustMatchThePublicAuthorityOrALoopbackAlias() {
    allowed("GET", "/api/models", false, "Host", "reviewer.example:8080");
    for (String alias : new String[] {"localhost:8080", "127.0.0.1:8080", "[::1]:8080"}) {
      allowed("GET", "/api/models", true, "Host", alias);
      refused(ErrorCode.HOST_NOT_ALLOWED, "GET", "/api/models", false, "Host", alias);
    }
    refused(ErrorCode.HOST_NOT_ALLOWED, "GET", "/", true, "Host", "localhost:8081");
    refused(ErrorCode.HOST_NOT_ALLOWED, "GET", "/", false, "Host", "Reviewer.example:8080");
    refused(ErrorCode.HOST_NOT_ALLOWED, "GET", "/", false);
  }

  @Test
  public void originAndFetchSiteMustBeTheSameSite() {
    String host = "reviewer.example:8080";
    allowed("POST", "/api/analyses", false, "Host", host, "Origin", "http://" + host);
    allowed("POST", "/api/analyses", false, "Host", host, "Sec-Fetch-Site", "same-origin");
    allowed("GET", "/", false, "Host", host, "Sec-Fetch-Site", "none");
    refused(
        ErrorCode.ORIGIN_NOT_ALLOWED, "POST", "/api/analyses", false, "Host", host, "Origin",
        "https://" + host);
    refused(
        ErrorCode.ORIGIN_NOT_ALLOWED, "POST", "/api/analyses", false, "Host", host, "Origin",
        "https://example.com");
    for (String site : new String[] {"same-site", "cross-site"})
      refused(
          ErrorCode.ORIGIN_NOT_ALLOWED, "GET", "/api/models", false, "Host", host,
          "Sec-Fetch-Site", site);
  }

  @Test
  public void onlyACrossSiteNavigationToASharedPageSkipsTheFetchSite() {
    String host = "reviewer.example:8080";
    for (String path : new String[] {"/?share=x", "/?%73hare=x", "/?share=", "/?a=1&share=x"})
      allowed(
          "GET", path, false, "Host", host, "Sec-Fetch-Site", "cross-site", "Sec-Fetch-Mode",
          "navigate", "Sec-Fetch-Dest", "document");
    String[][] refusedContexts = {
      {"GET", "/", "navigate", "document"},
      {"GET", "/api/shares/x", "navigate", "document"},
      {"GET", "/?share=x", "cors", "document"},
      {"GET", "/?share=x", "navigate", "iframe"},
      {"POST", "/?share=x", "navigate", "document"}
    };
    for (String[] context : refusedContexts)
      refused(
          ErrorCode.ORIGIN_NOT_ALLOWED, context[0], context[1], false, "Host", host,
          "Sec-Fetch-Site", "cross-site", "Sec-Fetch-Mode", context[2], "Sec-Fetch-Dest",
          context[3]);
    refused(
        ErrorCode.ORIGIN_NOT_ALLOWED, "GET", "/?share=x", false, "Host", host, "Origin",
        "https://example.com", "Sec-Fetch-Site", "cross-site", "Sec-Fetch-Mode", "navigate",
        "Sec-Fetch-Dest", "document");
  }

  @Test
  public void securityHeadersAreWrittenAndSharedPagesAreNotIndexed() {
    for (String path :
        new String[] {"/?share=x", "/api/shares/x", "/api/shares", "/", "/api/models"}) {
      var headers = new Headers();
      GUARD.writeHeaders(URI.create(path), headers);
      Assert.assertEquals(headers.getFirst("X-Content-Type-Options"), "nosniff");
      Assert.assertEquals(headers.getFirst("Referrer-Policy"), "no-referrer");
      Assert.assertEquals(
          headers.getFirst("Content-Security-Policy"),
          "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self'"
              + " data: blob:; font-src 'self'; connect-src 'self'; worker-src 'self' blob:;"
              + " object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'");
      Assert.assertEquals(
          headers.getFirst("X-Robots-Tag"),
          path.contains("share") ? "noindex" : null,
          path);
    }
  }

  @Test
  public void aWrongHostIsRefusedOnTheWire() throws Exception {
    try (var server = TestServer.start()) {
      for (String host :
          new String[] {"attacker.example", "localhost:" + (server.server.port() + 1)})
        try (var socket = new Socket("127.0.0.1", server.server.port())) {
          socket.setSoTimeout(5000);
          socket
              .getOutputStream()
              .write(
                  ("GET /api/models HTTP/1.1\r\nHost: " + host + "\r\nConnection: close\r\n\r\n")
                      .getBytes(StandardCharsets.US_ASCII));
          InputStream in = socket.getInputStream();
          String reply = new String(in.readAllBytes(), StandardCharsets.UTF_8);
          Assert.assertTrue(reply.startsWith("HTTP/1.1 403"), reply);
          Assert.assertTrue(reply.contains("host_not_allowed"), reply);
          Assert.assertTrue(
              reply.toLowerCase(Locale.ROOT).contains("content-security-policy"), reply);
        }
    }
  }

  private static void allowed(String method, String path, boolean loopback, String... headers) {
    GUARD.require(method, URI.create(path), headers(headers), loopback);
  }

  private static void refused(
      ErrorCode code, String method, String path, boolean loopback, String... headers) {
    ReviewerException failure =
        Assert.expectThrows(
            ReviewerException.class,
            () -> GUARD.require(method, URI.create(path), headers(headers), loopback));
    Assert.assertEquals(failure.code(), code, method + " " + path);
  }

  private static Headers headers(String... pairs) {
    var headers = new Headers();
    for (int i = 0; i < pairs.length; i += 2) headers.add(pairs[i], pairs[i + 1]);
    return headers;
  }
}
