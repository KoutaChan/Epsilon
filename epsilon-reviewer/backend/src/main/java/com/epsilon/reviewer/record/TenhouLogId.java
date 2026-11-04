package com.epsilon.reviewer.record;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Pattern;

/** 天鳳の牌譜の id（例 2026091512gm-0089-0000-1a2b3c4d）。 */
public record TenhouLogId(String value) {
  private static final Pattern FORM =
      Pattern.compile("[0-9]{10}gm-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{8}");
  private static final Set<String> HOSTS = Set.of("tenhou.net", "www.tenhou.net");

  /**
   * 天鳳の牌譜の URL から id を読む。http か https で、ポートと利用者の欄が無く、主は tenhou.net か www.tenhou.net。id は問いの log、
   * 道が /0/log/ なら問いの全体。
   *
   * @throws ReviewerException invalid_url（URL・id の形）・unsupported_url（天鳳でない主、雀魂の URL など）
   */
  public static TenhouLogId parse(String url) {
    URI input;
    try {
      input = new URI(url);
    } catch (URISyntaxException failure) {
      throw new ReviewerException(ErrorCode.INVALID_URL, "Invalid record URL.");
    }
    if (input.getHost() == null
        || input.getUserInfo() != null
        || input.getPort() != -1
        || !("https".equals(input.getScheme()) || "http".equals(input.getScheme())))
      throw new ReviewerException(ErrorCode.INVALID_URL);
    if (!HOSTS.contains(input.getHost().toLowerCase(Locale.ROOT)))
      throw new ReviewerException(ErrorCode.UNSUPPORTED_URL);
    String id = parameter(input.getRawQuery(), "log");
    if (id == null && "/0/log/".equals(input.getPath())) id = input.getRawQuery();
    if (id == null || !FORM.matcher(id).matches())
      throw new ReviewerException(ErrorCode.INVALID_URL, "Invalid Tenhou log ID.");
    return new TenhouLogId(id);
  }

  /** 牌譜を取る URL（https://tenhou.net/0/log/?&lt;id&gt;）。 */
  public URI downloadUri() {
    return URI.create("https://tenhou.net/0/log/?" + value);
  }

  /** 問いの name の値（URL を戻した名前と値）。無ければ null。 */
  private static String parameter(String rawQuery, String name) {
    if (rawQuery == null) return null;
    String value = null;
    try {
      for (String part : rawQuery.split("&")) {
        String[] pair = part.split("=", 2);
        if (pair.length == 2 && URLDecoder.decode(pair[0], StandardCharsets.UTF_8).equals(name))
          value = URLDecoder.decode(pair[1], StandardCharsets.UTF_8);
      }
    } catch (IllegalArgumentException failure) {
      throw new ReviewerException(ErrorCode.INVALID_URL, "Invalid record URL.");
    }
    return value;
  }
}
