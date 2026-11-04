package com.epsilon.reviewer.http;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.sun.net.httpserver.Headers;
import java.net.URI;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.Set;

/**
 * 全ての応答の安全の頭と、全ての道（静的ファイルも）の Host・Origin・Sec-Fetch-Site の確かめ（頭だけを見る）。
 *
 * <p>Host は公開の権威と完全に一致（相手が loopback なら localhost・127.0.0.1・[::1] の同じポートも可）。
 * Origin があれば &lt;scheme&gt;://&lt;Host&gt;。Sec-Fetch-Site があれば same-origin か none。他のサイトからの
 * 共有リンクの移動（GET / の share の問い、Sec-Fetch-Mode: navigate、Sec-Fetch-Dest: document）だけは
 * Sec-Fetch-Site を見ない（Host と Origin は見る）。
 */
final class RequestGuard {
  static final String CONTENT_SECURITY_POLICY =
      "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self'"
          + " data: blob:; font-src 'self'; connect-src 'self'; worker-src 'self' blob:;"
          + " object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'";

  private static final Set<String> SAME_SITE = Set.of("same-origin", "none");

  private final String scheme;
  private final String authority;
  private final int port;

  /**
   * @param authority 公開の権威（:0 は結んだポートに読み替えた物）
   * @param port 結んだポート
   */
  RequestGuard(String scheme, String authority, int port) {
    this.scheme = scheme;
    this.authority = authority;
    this.port = port;
  }

  /** 安全の頭を置く。/?share= と /api/shares/* は検索に載せない。 */
  void writeHeaders(URI uri, Headers response) {
    response.set("X-Content-Type-Options", "nosniff");
    response.set("Referrer-Policy", "no-referrer");
    response.set("Content-Security-Policy", CONTENT_SECURITY_POLICY);
    if (isSharePage(uri) || isShareApi(uri.getPath())) response.set("X-Robots-Tag", "noindex");
  }

  /**
   * @param loopback 相手が loopback の番地か
   * @throws ReviewerException host_not_allowed・origin_not_allowed
   */
  void require(String method, URI uri, Headers request, boolean loopback) {
    String host = request.getFirst("Host");
    boolean allowed = Objects.equals(host, authority);
    if (loopback)
      allowed |=
          Objects.equals(host, "localhost:" + port)
              || Objects.equals(host, "127.0.0.1:" + port)
              || Objects.equals(host, "[::1]:" + port);
    if (!allowed) throw new ReviewerException(ErrorCode.HOST_NOT_ALLOWED);
    String origin = request.getFirst("Origin");
    if (origin != null && !origin.equals(scheme + "://" + host))
      throw new ReviewerException(ErrorCode.ORIGIN_NOT_ALLOWED);
    boolean sharedNavigation =
        isSharePage(uri)
            && method.equals("GET")
            && "navigate".equals(request.getFirst("Sec-Fetch-Mode"))
            && "document".equals(request.getFirst("Sec-Fetch-Dest"));
    if (sharedNavigation) return;
    String site = request.getFirst("Sec-Fetch-Site");
    if (site != null && !SAME_SITE.contains(site))
      throw new ReviewerException(ErrorCode.ORIGIN_NOT_ALLOWED);
  }

  static boolean isShareApi(String path) {
    return path.equals("/api/shares") || path.startsWith("/api/shares/");
  }

  /** 画面の / で、問いに URL を戻した鍵が share の欄がある（値は見ない）。 */
  static boolean isSharePage(URI uri) {
    if (!uri.getPath().equals("/") || uri.getRawQuery() == null) return false;
    for (String parameter : uri.getRawQuery().split("&")) {
      String key = parameter.split("=", 2)[0];
      try {
        if (URLDecoder.decode(key, StandardCharsets.UTF_8).equals("share")) return true;
      } catch (IllegalArgumentException ignored) {
        // 戻せない鍵は share でない。
      }
    }
    return false;
  }
}
