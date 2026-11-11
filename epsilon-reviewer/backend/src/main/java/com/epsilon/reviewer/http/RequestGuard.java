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
 * 全ての応答にセキュリティのヘッダーを付け、静的ファイルを含む全てのパスで Host・Origin・Sec-Fetch-Site を確かめる。
 * 確かめるときに見るのは、ヘッダーと、相手が loopback かどうかだけ。
 *
 * <p>Host は公開のホストとポートに完全に一致すること（相手が loopback なら、同じポートの localhost・127.0.0.1・[::1] も認める）。
 * Origin があれば &lt;scheme&gt;://&lt;Host&gt; であること。Sec-Fetch-Site があれば same-origin か none であること。ただし、
 * 他のサイトのリンクから共有ページを開くとき（share のクエリを持つ GET /、Sec-Fetch-Mode: navigate、Sec-Fetch-Dest: document）だけは
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
   * @param authority 公開のホストとポート（:0 はバインドしたポートに読み替えたもの）
   * @param port バインドしたポート
   */
  RequestGuard(String scheme, String authority, int port) {
    this.scheme = scheme;
    this.authority = authority;
    this.port = port;
  }

  /** セキュリティのヘッダーを置く。/?share= と /api/shares/* には、検索エンジンに載せない指定（noindex）も付ける。 */
  void writeHeaders(URI uri, Headers response) {
    response.set("X-Content-Type-Options", "nosniff");
    response.set("Referrer-Policy", "no-referrer");
    response.set("Content-Security-Policy", CONTENT_SECURITY_POLICY);
    if (isSharePage(uri) || isShareApi(uri.getPath())) response.set("X-Robots-Tag", "noindex");
  }

  /**
   * @param loopback 相手が loopback のアドレスなら true
   * @throws ReviewerException host_not_allowed か origin_not_allowed
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

  /** 画面の / へのリクエストで、クエリに share というキーがあるか（キーは URL デコードして比べ、値は見ない）。 */
  static boolean isSharePage(URI uri) {
    if (!uri.getPath().equals("/") || uri.getRawQuery() == null) return false;
    for (String parameter : uri.getRawQuery().split("&")) {
      String key = parameter.split("=", 2)[0];
      try {
        if (URLDecoder.decode(key, StandardCharsets.UTF_8).equals("share")) return true;
      } catch (IllegalArgumentException ignored) {
        // デコードできないキーは share ではない。
      }
    }
    return false;
  }
}
