package com.epsilon.reviewer.http;

import java.net.URI;
import java.nio.file.Path;
import java.util.Set;

/**
 * HTTP サーバの設定。
 *
 * @param host バインドするアドレス
 * @param port バインドするポート（0 なら空いているポートを使う）
 * @param origin 公開の origin（scheme://rawAuthority の形にそろえる）。ポートが :0 なら、バインドしたポートに読み替える
 * @param webDir 画面のファイルのフォルダー（web/dist）。無ければ null
 */
public record ServerConfig(String host, int port, URI origin, Path webDir) {

  /**
   * 公開の origin を読む。ユーザー情報・クエリ・フラグメントは持てず、パスは空か / だけ。
   *
   * @param text http(s)://host[:port]
   * @return scheme://rawAuthority の形にそろえた origin
   * @throws IllegalArgumentException 形が正しくないとき
   */
  public static URI origin(String text) {
    URI origin = URI.create(text);
    if (!Set.of("http", "https").contains(origin.getScheme())
        || origin.getHost() == null
        || origin.getUserInfo() != null
        || origin.getRawQuery() != null
        || origin.getFragment() != null
        || !(origin.getPath().isEmpty() || origin.getPath().equals("/")))
      throw new IllegalArgumentException("expected http(s)://host[:port]");
    return URI.create(origin.getScheme() + "://" + origin.getRawAuthority());
  }
}
