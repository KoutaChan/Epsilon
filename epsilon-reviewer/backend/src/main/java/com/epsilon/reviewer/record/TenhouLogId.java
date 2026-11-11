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

/**
 * 天鳳の牌譜の id（例: 2026091512gm-0089-0000-1a2b3c4d）。牌譜を取るときの形で持つ。
 *
 * @param value 対局の日時 10 桁、gm、ルール 4 桁、ロビー 4〜5 桁、16 進 8 桁をつないだもの
 */
public record TenhouLogId(String value) {
  private static final Pattern FORM =
      Pattern.compile(
          "[0-9]{10}gm-[0-9a-fA-F]{4}-[0-9a-fA-F]{4,5}-(?:[0-9a-fA-F]{8}|x[0-9a-fA-F]{12})");
  private static final Set<String> HOSTS = Set.of("tenhou.net", "www.tenhou.net");

  /** 末尾が x の id を直すときの日時。これより後の id は、日時から表の位置を決める。 */
  private static final String KEYED_AFTER = "2010041111";

  /** 末尾が x の id を直すときに混ぜる 16 bit の値の表。 */
  private static final int[] KEYS = {
    22136, 52719, 55146, 42104, 59591, 46934, 9248, 28891, 49597, 52974, 62844, 4015,
    18311, 50730, 43056, 17939, 64838, 38145, 27008, 39128, 35652, 63407, 65535, 23473,
    35164, 55230, 27536, 4386, 64920, 29075, 42617, 17294, 18868, 2081
  };

  /**
   * 天鳳の牌譜の URL から id を読む。受け取る URL は、http か https で、ポートとユーザー情報を書かず、ホストが tenhou.net か
   * www.tenhou.net のもの。id はクエリの log の値で、log が無くパスが /0/log/ ならクエリ全体を id とする。自分の対局の牌譜を開いた
   * URL の id（末尾が x と 16 進 12 桁）は、牌譜を取るときの id に直す。
   *
   * @throws ReviewerException invalid_url（URL や id の形が違う）・unsupported_url（ホストが天鳳でない。雀魂の URL など）
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
    return new TenhouLogId(downloadForm(id));
  }

  /**
   * 末尾が x と 16 進 12 桁の id を、牌譜を取るときの id（末尾が 16 進 8 桁）に直す。12 桁を 16 bit の値 a・b・c に分け、表の値 k と
   * その次の値 k' を使って、a ^ b ^ k と b ^ c ^ k ^ k' を並べる。ほかの id はそのまま返す。
   */
  private static String downloadForm(String id) {
    int code = id.lastIndexOf('-') + 1;
    if (id.charAt(code) != 'x') return id;
    int a = Integer.parseInt(id, code + 1, code + 5, 16);
    int b = Integer.parseInt(id, code + 5, code + 9, 16);
    int c = Integer.parseInt(id, code + 9, code + 13, 16);
    int index = 0;
    if (id.substring(0, 10).compareTo(KEYED_AFTER) > 0)
      index = Integer.parseInt("3" + id.substring(4, 10)) % (33 - (id.charAt(9) - '0'));
    return id.substring(0, code)
        + String.format("%04x%04x", a ^ b ^ KEYS[index], b ^ c ^ KEYS[index] ^ KEYS[index + 1]);
  }

  /** 牌譜を取る URL（https://tenhou.net/0/log/?&lt;id&gt;）。 */
  public URI downloadUri() {
    return URI.create("https://tenhou.net/0/log/?" + value);
  }

  /** クエリから {@code name} の値を取り出す。名前と値は URL デコードしてから扱う。同じ名前が何度もあれば最後の値を返し、無ければ null。 */
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
