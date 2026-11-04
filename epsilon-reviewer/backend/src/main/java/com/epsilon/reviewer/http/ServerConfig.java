package com.epsilon.reviewer.http;

import java.net.URI;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.Map;
import java.util.Set;

/**
 * HTTP のサーバの設定。
 *
 * @param origin 公開の origin（scheme://rawAuthority に揃える）。ポートが :0 なら結んだポートに読み替える
 * @param models models.json。無ければ null
 * @param webDir 画面の web/dist。無ければ null
 */
public record ServerConfig(
    String host, int port, URI origin, Path models, Path dataDir, Path webDir) {
  /**
   * 選択肢を読む（--help は Main が先に扱う）。
   *
   * @throws IllegalArgumentException 知らない選択肢・値の無い選択肢・形の違う値
   */
  public static ServerConfig parse(String[] args) {
    Map<ServerOption, String> values = new EnumMap<>(ServerOption.class);
    for (int i = 0; i < args.length; i++) {
      ServerOption option =
          ServerOption.of(args[i]).filter(found -> found != ServerOption.HELP).orElse(null);
      if (option == null || i + 1 >= args.length)
        throw new IllegalArgumentException("Unknown command-line option: " + args[i]);
      values.put(option, args[++i]);
    }
    String host = values.getOrDefault(ServerOption.HOST, "127.0.0.1");
    int port = Integer.parseInt(values.getOrDefault(ServerOption.PORT, "8080"));
    if (port < 1 || port > 65535)
      throw new IllegalArgumentException("Port must be between 1 and 65535.");
    URI origin = URI.create(values.getOrDefault(ServerOption.ORIGIN, "http://127.0.0.1:" + port));
    if (!Set.of("http", "https").contains(origin.getScheme())
        || origin.getHost() == null
        || origin.getUserInfo() != null
        || origin.getRawQuery() != null
        || origin.getFragment() != null
        || !(origin.getPath().isEmpty() || origin.getPath().equals("/")))
      throw new IllegalArgumentException("Origin must use http(s)://host[:port].");
    return new ServerConfig(
        host,
        port,
        URI.create(origin.getScheme() + "://" + origin.getRawAuthority()),
        path(values.get(ServerOption.MODELS)),
        Path.of(values.getOrDefault(ServerOption.DATA_DIR, "reviewer-data")),
        path(values.get(ServerOption.WEB_DIR)));
  }

  private static Path path(String value) {
    return value == null ? null : Path.of(value);
  }
}
