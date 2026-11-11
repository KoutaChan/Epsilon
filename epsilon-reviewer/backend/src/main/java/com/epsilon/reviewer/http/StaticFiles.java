package com.epsilon.reviewer.http;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 画面のファイル（web/dist）を配る。起動時に 1 回だけインデックスを作り、中身と、事前に圧縮した .br・.gz をメモリに持つ。シンボリック
 * リンクはたどらない。強い ETag が合えば 304 を返す。キャッシュは、/assets/* を中身の
 * 変わらないもの（immutable）、/tiles/* を 1 週間とし、ほかは no-cache にする。ファイルが見つからないとき、パスに点が無ければ
 * index.html を返し（SPA）、点があれば 404 にする。
 */
final class StaticFiles {
  private static final Logger LOG = LoggerFactory.getLogger(StaticFiles.class);
  private static final String INDEX = "index.html";
  private static final Map<String, String> TYPES =
      Map.of(
          ".js", "text/javascript; charset=utf-8",
          ".css", "text/css; charset=utf-8",
          ".svg", "image/svg+xml",
          ".html", "text/html; charset=utf-8",
          ".woff2", "font/woff2",
          ".png", "image/png",
          ".json", "application/json");

  private final boolean configured;

  /** '/' で区切った相対パスから中身への表。 */
  private final Map<String, Asset> assets;

  private StaticFiles(boolean configured, Map<String, Asset> assets) {
    this.configured = configured;
    this.assets = assets;
  }

  /** @param directory 画面のファイルのフォルダー（web/dist）。無ければ null で、画面のパスは全て 404 になる */
  static StaticFiles index(Path directory) throws IOException {
    if (directory == null) return new StaticFiles(false, Map.of());
    Path root = directory.toAbsolutePath().normalize();
    if (!Files.isDirectory(root)) {
      LOG.warn("The web directory {} does not exist; the UI is unavailable.", root);
      return new StaticFiles(true, Map.of());
    }
    Map<String, Path> files = new HashMap<>();
    try (var walk = Files.walk(root)) {
      for (Path file : walk.toList())
        if (Files.isRegularFile(file, LinkOption.NOFOLLOW_LINKS))
          files.put(root.relativize(file).toString().replace('\\', '/'), file);
    }
    Map<String, Asset> assets = new HashMap<>();
    long bytes = 0;
    for (Map.Entry<String, Path> file : files.entrySet()) {
      String name = file.getKey();
      if ((name.endsWith(".br") || name.endsWith(".gz"))
          && files.containsKey(name.substring(0, name.length() - 3))) continue;
      byte[] body = Files.readAllBytes(file.getValue());
      String hash = hash(body);
      var asset =
          new Asset(
              type(name),
              new Representation(body, "\"" + hash + "\"", null),
              compressed(files.get(name + ".br"), file.getValue(), hash, "br"),
              compressed(files.get(name + ".gz"), file.getValue(), hash, "gzip"));
      assets.put(name, asset);
      bytes += asset.size();
    }
    LOG.info("Indexed {} web files ({} bytes) from {}.", assets.size(), bytes, root);
    return new StaticFiles(true, Map.copyOf(assets));
  }

  void serve(HttpExchange exchange, HttpReplies replies, String path) throws IOException {
    if (!configured)
      throw new ReviewerException(ErrorCode.NOT_FOUND, "Web UI directory is not configured.");
    String method = exchange.getRequestMethod();
    boolean head = method.equals("HEAD");
    if (!head && !method.equals("GET")) throw new ReviewerException(ErrorCode.NOT_FOUND);
    String name = path.equals("/") ? INDEX : path.substring(1);
    Asset asset = assets.get(name);
    if (asset == null) {
      if (name.contains(".")) throw new ReviewerException(ErrorCode.NOT_FOUND, "File not found.");
      name = INDEX;
      asset = assets.get(INDEX);
      if (asset == null)
        throw new ReviewerException(ErrorCode.NOT_FOUND, "Web UI build is unavailable.");
    }
    Headers request = exchange.getRequestHeaders();
    Representation chosen = asset.plain();
    if (asset.brotli() != null && HttpReplies.accepts(request, "br")) chosen = asset.brotli();
    else if (asset.gzip() != null && HttpReplies.accepts(request, "gzip")) chosen = asset.gzip();
    Headers headers = exchange.getResponseHeaders();
    headers.set("Content-Type", asset.type());
    headers.set("Cache-Control", cacheControl(name));
    headers.set("ETag", chosen.etag());
    if (asset.brotli() != null || asset.gzip() != null) headers.set("Vary", "Accept-Encoding");
    if (chosen.encoding() != null) headers.set("Content-Encoding", chosen.encoding());
    if (HttpReplies.notModified(request, chosen.etag())) replies.notModified();
    else replies.send(200, chosen.body(), head);
  }

  private static String cacheControl(String name) {
    if (name.startsWith("assets/")) return "public, max-age=31536000, immutable";
    if (name.startsWith("tiles/")) return "public, max-age=604800";
    return "no-cache";
  }

  private static String type(String name) {
    int dot = name.lastIndexOf('.');
    return dot < 0
        ? "application/octet-stream"
        : TYPES.getOrDefault(name.substring(dot), "application/octet-stream");
  }

  /** 事前に圧縮したファイルを読む。本体より古いものは前のビルドの残りなので使わない。 */
  private static Representation compressed(Path file, Path body, String hash, String encoding)
      throws IOException {
    if (file == null) return null;
    if (Files.getLastModifiedTime(file).compareTo(Files.getLastModifiedTime(body)) < 0) {
      LOG.warn("Ignoring {} because it is older than {}.", file, body);
      return null;
    }
    String tag = "\"" + hash + "-" + (encoding.equals("gzip") ? "gz" : encoding) + "\"";
    return new Representation(Files.readAllBytes(file), tag, encoding);
  }

  /** 中身の SHA-256 の先頭 8 バイトを 16 進で書いたもの（強い ETag に使う）。 */
  private static String hash(byte[] body) {
    try {
      byte[] digest = MessageDigest.getInstance("SHA-256").digest(body);
      return HexFormat.of().formatHex(digest, 0, 8);
    } catch (NoSuchAlgorithmException failure) {
      throw new IllegalStateException(failure);
    }
  }

  /**
   * 1 つのファイルと、その圧縮版。
   *
   * @param brotli .br の圧縮版。無ければ null
   * @param gzip .gz の圧縮版。無ければ null
   */
  private record Asset(
      String type, Representation plain, Representation brotli, Representation gzip) {
    long size() {
      return plain.body().length
          + (brotli == null ? 0 : brotli.body().length)
          + (gzip == null ? 0 : gzip.body().length);
    }
  }

  /** @param encoding Content-Encoding。圧縮していないものは null */
  private record Representation(byte[] body, String etag, String encoding) {}
}
