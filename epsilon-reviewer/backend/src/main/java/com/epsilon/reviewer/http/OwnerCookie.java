package com.epsilon.reviewer.http;

import com.epsilon.reviewer.store.OwnerId;
import com.sun.net.httpserver.HttpExchange;
import java.security.SecureRandom;
import java.util.Base64;
import java.util.List;
import java.util.regex.Pattern;

/**
 * オーナーの Cookie（epsilon_reviewer_owner）。値は 32 バイトの乱数を base64url でパディングなしに書いた 43 文字。全ての Cookie ヘッダーの
 * うち最後に出てくる正しい形の値を使い、無ければ発行する。オーナーの id はこの値の SHA-256。ヘッダーを確かめた後に、オーナーの
 * ルートでだけ呼ぶ。
 */
final class OwnerCookie {
  static final String NAME = "epsilon_reviewer_owner";
  private static final Pattern KEY = Pattern.compile("[A-Za-z0-9_-]{43}");
  private static final String ATTRIBUTES = "; Path=/; HttpOnly; SameSite=Lax; Max-Age=31536000";

  private final boolean secure;
  private final SecureRandom random = new SecureRandom();

  /** @param secure true なら Secure を付ける（origin が https のとき） */
  OwnerCookie(boolean secure) {
    this.secure = secure;
  }

  OwnerId resolve(HttpExchange exchange) {
    String key = key(exchange.getRequestHeaders().getOrDefault("Cookie", List.of()));
    if (key == null) {
      byte[] bytes = new byte[32];
      random.nextBytes(bytes);
      key = Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
      exchange
          .getResponseHeaders()
          .add("Set-Cookie", NAME + "=" + key + ATTRIBUTES + (secure ? "; Secure" : ""));
    }
    return OwnerId.fromCookieKey(key);
  }

  /** Cookie ヘッダーから、最後に出てくる正しい形の値を返す。無ければ null。 */
  static String key(List<String> headers) {
    String key = null;
    for (String header : headers)
      for (String item : header.split(";")) {
        String[] pair = item.trim().split("=", 2);
        if (pair.length == 2 && pair[0].equals(NAME) && KEY.matcher(pair[1]).matches())
          key = pair[1];
      }
    return key;
  }
}
