package com.epsilon.reviewer.http;

import java.util.Arrays;
import java.util.Optional;

/** HTTP のサーバの CLI の選択肢。 */
public enum ServerOption {
  /** 結ぶ番地。既定 127.0.0.1。 */
  HOST("--host"),
  /** 結ぶポート（1..65535）。既定 8080。 */
  PORT("--port"),
  /**
   * 公開の origin（http(s)://host[:port]、利用者・問い・断片なし、道は空か /）。Host の守りと共有リンクが使う。既定
   * http://127.0.0.1:&lt;port&gt;。
   */
  ORIGIN("--origin"),
  /** models.json。無ければ空のカタログ。 */
  MODELS("--models"),
  /** 保存先。既定 reviewer-data。 */
  DATA_DIR("--data-dir"),
  /** 画面の web/dist。無ければ画面の道は 404。 */
  WEB_DIR("--web-dir"),
  /** 使い方を記録に書いて終わる（値を取らない）。 */
  HELP("--help");

  private static final String USAGE =
      "Epsilon AI Reviewer\n"
          + "--host 127.0.0.1 --port 8080 --origin http://127.0.0.1:8080\n"
          + "--models models.json --data-dir ./reviewer-data --web-dir ../web/dist\n"
          + "Without --models the catalog is empty. Set --origin to the public HTTPS URL when"
          + " publishing.";

  private final String flag;

  ServerOption(String flag) {
    this.flag = flag;
  }

  public String flag() {
    return flag;
  }

  /** --help の文。 */
  public static String usage() {
    return USAGE;
  }

  static Optional<ServerOption> of(String flag) {
    return Arrays.stream(values()).filter(option -> option.flag.equals(flag)).findFirst();
  }
}
