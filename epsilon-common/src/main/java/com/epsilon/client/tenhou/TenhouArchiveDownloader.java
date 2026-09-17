package com.epsilon.client.tenhou;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 天鳳公開アーカイブから鳳凰卓の牌譜をダウンロードする。
 *
 * <p>取得方法を自動選択する:
 *
 * <ul>
 *   <li>完了した過去年: 年別 ZIP ({@code scraw{yyyy}.zip})
 *   <li>当年・部分期間: 個別 scc ファイル ({@code dat/scc{yyyymmdd}.html.gz})
 *   <li>直近1週間: 時間別 scc ファイル ({@code dat/scc{yyyymmddhh}.html.gz}) にフォールバック
 * </ul>
 *
 * ZIP 間は天鳳の規約に従い20分以上の間隔を空ける。
 *
 * <p>使い方:
 *
 * <pre>{@code
 * java TenhouArchiveDownloader <outputDir> <from> [to] [--three]
 *   from/to: yyyy (年指定) または yyyy-MM-dd (日付指定)
 * }</pre>
 */
public final class TenhouArchiveDownloader {

  private static final Logger log = LoggerFactory.getLogger(TenhouArchiveDownloader.class);

  private static final String ARCHIVE_URL_PREFIX = "https://tenhou.net/sc/raw/scraw";
  private static final String LIST_OLD_URL = "https://tenhou.net/sc/raw/list.cgi?old";
  private static final String LIST_RECENT_URL = "https://tenhou.net/sc/raw/list.cgi";
  private static final String DAT_BASE_URL = "https://tenhou.net/sc/raw/dat/";
  private static final String LOG_URL_PREFIX = "https://tenhou.net/0/log/?";

  /** 天鳳アーカイブ規約: ダウンロード間隔20分以上 */
  private static final long ARCHIVE_MIN_GAP_MS = 20 * 60 * 1000L;

  // インデックス行パターン: ルール説明 | <a href="...?log=LOGID">牌譜</a>
  private static final Pattern ENTRY_PATTERN =
      Pattern.compile("\\|\\s*([^|]+?)\\s*\\|\\s*<a[^>]+log=([^\"&>]+)[^>]*>牌譜</a>");

  // list.cgi パターン: {file:'2026/scc20260101.html.gz',size:...} または
  // {file:'scc2026033000.html.gz',size:...}
  // グループ(1)=パス含むファイル名 (例: 2026/scc20260101.html.gz), グループ(2)=日付部分
  private static final Pattern LIST_SCC_PATTERN =
      Pattern.compile("\\{file:'((?:[^/]*/)?scc(\\d{8,10})\\.html\\.gz)',size:\\d+}");

  private static final String FOUR_PLAYER_HOUOU = "四鳳南喰赤－";
  private static final String THREE_PLAYER_HOUOU = "三鳳南喰赤－";

  private final HttpClient client;
  private final Path outputDir;
  private final long delayMs;
  private final boolean threePlayer;
  private long lastArchiveTime;

  /**
   * 保存先、牌譜ごとの待機時間、対象ルールを指定してダウンローダーを生成する。
   *
   * @param outputDir 保存先ディレクトリ
   * @param delayMs 牌譜ダウンロード間の待機時間 (ms)
   * @param threePlayer true で三麻鳳凰卓、false で四麻鳳凰卓
   */
  public TenhouArchiveDownloader(Path outputDir, long delayMs, boolean threePlayer) {
    this.client =
        HttpClient.newBuilder()
            .followRedirects(HttpClient.Redirect.NORMAL)
            .connectTimeout(Duration.ofSeconds(30))
            .build();
    this.outputDir = outputDir;
    this.delayMs = delayMs;
    this.threePlayer = threePlayer;
  }

  /**
   * 日付範囲の鳳凰卓牌譜をダウンロードする。 完了した年は ZIP、それ以外は個別 scc ファイルを使用する。
   *
   * @param from 対象期間の開始日（両端を含む）
   * @param to 対象期間の終了日（両端を含む）
   * @return ダウンロードした牌譜数
   * @throws IOException アーカイブ、インデックスまたは牌譜の読み書きに失敗した場合
   * @throws InterruptedException 通信間隔の制限待機または HTTP 通信中にスレッドが割り込まれた場合
   */
  public int download(LocalDate from, LocalDate to) throws IOException, InterruptedException {
    Files.createDirectories(outputDir);
    int totalDownloaded = 0;
    int currentYear = LocalDate.now().getYear();

    for (int year = from.getYear(); year <= to.getYear(); year++) {
      LocalDate yearStart = LocalDate.of(year, 1, 1);
      LocalDate yearEnd = LocalDate.of(year, 12, 31);
      LocalDate effectiveFrom = from.isAfter(yearStart) ? from : yearStart;
      LocalDate effectiveTo = to.isBefore(yearEnd) ? to : yearEnd;

      boolean fullYear = effectiveFrom.equals(yearStart) && effectiveTo.equals(yearEnd);
      boolean pastYear = year < currentYear;

      if (fullYear && pastYear) {
        totalDownloaded += downloadViaZip(year);
      } else {
        totalDownloaded += downloadViaScc(effectiveFrom, effectiveTo);
      }
    }

    log.info("Download complete: {} logs total", totalDownloaded);
    return totalDownloaded;
  }

  // セクション: ZIP アーカイブ経由

  private int downloadViaZip(int year) throws IOException, InterruptedException {
    enforceArchiveGap();

    Path zipPath = downloadArchiveZip(year);
    lastArchiveTime = System.currentTimeMillis();
    if (zipPath == null) {
      log.warn("Archive for {} not available, falling back to scc", year);
      return downloadViaScc(LocalDate.of(year, 1, 1), LocalDate.of(year, 12, 31));
    }

    log.info("Extracting houou log IDs from {} archive...", year);
    List<String> logIds = extractLogIdsFromArchive(zipPath);
    log.info("Found {} houou games in {}", logIds.size(), year);
    if (logIds.isEmpty()) {
      return 0;
    }

    return downloadLogs(logIds, year);
  }

  private Path downloadArchiveZip(int year) throws IOException, InterruptedException {
    Path cacheDir = outputDir.resolve("archive_cache");
    Files.createDirectories(cacheDir);
    Path zipPath = cacheDir.resolve("scraw" + year + ".zip");

    if (Files.exists(zipPath) && Files.size(zipPath) > 0) {
      log.info("Using cached archive: {} ({}MB)", zipPath, Files.size(zipPath) / (1024 * 1024));
      return zipPath;
    }

    String url = ARCHIVE_URL_PREFIX + year + ".zip";
    log.info("Downloading archive: {} ...", url);

    try {
      HttpRequest request =
          HttpRequest.newBuilder()
              .uri(URI.create(url))
              .timeout(Duration.ofMinutes(30))
              .GET()
              .build();
      HttpResponse<Path> response = client.send(request, HttpResponse.BodyHandlers.ofFile(zipPath));
      if (response.statusCode() != 200) {
        Files.deleteIfExists(zipPath);
        throw new IOException("HTTP " + response.statusCode());
      }
      log.info("Archive downloaded: {} ({}MB)", zipPath, Files.size(zipPath) / (1024 * 1024));
      return zipPath;
    } catch (IOException e) {
      Files.deleteIfExists(zipPath);
      log.warn("Failed to download archive for {}: {}", year, e.getMessage());
      return null;
    }
  }

  private List<String> extractLogIdsFromArchive(Path zipPath) throws IOException {
    List<String> allIds = new ArrayList<>();
    try (ZipFile zip = new ZipFile(zipPath.toFile())) {
      Enumeration<? extends ZipEntry> entries = zip.entries();
      int sccCount = 0;
      while (entries.hasMoreElements()) {
        ZipEntry entry = entries.nextElement();
        String name = entry.getName();
        String fileName = name.contains("/") ? name.substring(name.lastIndexOf('/') + 1) : name;
        if (!fileName.startsWith("scc") || !fileName.endsWith(".html.gz")) {
          continue;
        }

        sccCount++;
        try (InputStream is = zip.getInputStream(entry)) {
          byte[] gzipped = is.readAllBytes();
          String html = decompressGzip(gzipped);
          allIds.addAll(extractHououLogIds(html));
        }
      }
      log.info("Processed {} scc entries from archive", sccCount);
    }
    return allIds;
  }

  // セクション: 個別 scc ファイル経由(list.cgi ベース)

  private int downloadViaScc(LocalDate from, LocalDate to)
      throws IOException, InterruptedException {
    enforceArchiveGap();

    log.info("Fetching scc file list from list.cgi...");
    List<String> sccFiles = fetchSccFileList(from, to);
    log.info("Found {} scc files in date range {} to {}", sccFiles.size(), from, to);

    Path cacheDir = outputDir.resolve("scc_cache");
    Files.createDirectories(cacheDir);

    List<String> allLogIds = new ArrayList<>();
    for (String sccFile : sccFiles) {
      // 2026/scc20260101.html.gz → scc20260101.html (キャッシュ名)
      String cacheName = Path.of(sccFile).getFileName().toString().replace(".gz", "");
      Path cacheFile = cacheDir.resolve(cacheName);
      if (Files.exists(cacheFile)) {
        allLogIds.addAll(extractHououLogIds(Files.readString(cacheFile)));
        continue;
      }

      try {
        byte[] gzipped = downloadBytes(DAT_BASE_URL + sccFile);
        String html = decompressGzip(gzipped);
        Files.writeString(cacheFile, html);
        List<String> ids = extractHououLogIds(html);
        allLogIds.addAll(ids);
        log.debug("{}: {} houou games", sccFile, ids.size());
      } catch (IOException e) {
        log.warn("Failed to download {}: {}", sccFile, e.getMessage());
      }
    }

    lastArchiveTime = System.currentTimeMillis();

    log.info("Found {} houou games from scc ({} to {})", allLogIds.size(), from, to);
    if (allLogIds.isEmpty()) {
      return 0;
    }

    return downloadLogs(allLogIds, from.getYear());
  }

  /** {@code list.cgi}と{@code list.cgi?old}から、指定した日付範囲に含まれるsccファイル名を取得する。 */
  private List<String> fetchSccFileList(LocalDate from, LocalDate to)
      throws IOException, InterruptedException {
    String fromStr = from.format(DateTimeFormatter.BASIC_ISO_DATE);
    String toStr = to.format(DateTimeFormatter.BASIC_ISO_DATE);

    List<String> result = new ArrayList<>();

    // list.cgi?old (当年の古いデータ) とlist.cgi (直近1週間) の両方を取得
    for (String listUrl : List.of(LIST_OLD_URL, LIST_RECENT_URL)) {
      String body;
      try {
        body = downloadText(listUrl);
      } catch (IOException e) {
        log.warn("Failed to fetch {}: {}", listUrl, e.getMessage());
        continue;
      }

      Matcher m = LIST_SCC_PATTERN.matcher(body);
      while (m.find()) {
        String fileName = m.group(1); // scc20260101.html.gz
        String dateKey = m.group(2); // 20260101 or 2026010100
        String dateOnly = dateKey.substring(0, 8);
        if (dateOnly.compareTo(fromStr) >= 0 && dateOnly.compareTo(toStr) <= 0) {
          result.add(fileName);
        }
      }
    }

    return result;
  }

  // セクション: 共通の牌譜ダウンロード

  private int downloadLogs(List<String> logIds, int year) throws IOException, InterruptedException {
    Path logDir = outputDir.resolve("logs").resolve(String.valueOf(year));
    Files.createDirectories(logDir);

    int downloaded = 0;
    int skipped = 0;
    int total = logIds.size();

    for (String logId : logIds) {
      Path outFile = logDir.resolve(logId + ".xml.gz");
      if (Files.exists(outFile)) {
        skipped++;
        continue;
      }
      try {
        String xml = downloadText(LOG_URL_PREFIX + logId);
        writeGzip(outFile, xml);
        downloaded++;
        if (delayMs > 0) {
          Thread.sleep(delayMs);
        }
      } catch (IOException e) {
        log.warn("Failed to download log {}: {}", logId, e.getMessage());
      }

      int processed = downloaded + skipped;
      if (processed % 500 == 0) {
        log.info(
            "  {}/{} processed ({} downloaded, {} skipped) for {}",
            processed,
            total,
            downloaded,
            skipped,
            year);
      }
    }

    log.info(
        "{}: {} downloaded, {} skipped, {} total houou games", year, downloaded, skipped, total);
    return downloaded;
  }

  // セクション: ユーティリティ

  private void enforceArchiveGap() throws InterruptedException {
    if (lastArchiveTime > 0) {
      long elapsed = System.currentTimeMillis() - lastArchiveTime;
      if (elapsed < ARCHIVE_MIN_GAP_MS) {
        long waitMs = ARCHIVE_MIN_GAP_MS - elapsed;
        log.info(
            "Waiting {} min {} sec before next download (tenhou rate limit)",
            waitMs / 60_000,
            (waitMs % 60_000) / 1000);
        Thread.sleep(waitMs);
      }
    }
  }

  List<String> extractHououLogIds(String html) {
    String filter = threePlayer ? THREE_PLAYER_HOUOU : FOUR_PLAYER_HOUOU;
    List<String> ids = new ArrayList<>();
    Matcher m = ENTRY_PATTERN.matcher(html);
    while (m.find()) {
      String desc = m.group(1);
      String logId = m.group(2);
      if (desc.contains(filter)) {
        ids.add(logId);
      }
    }
    return ids;
  }

  private byte[] downloadBytes(String url) throws IOException, InterruptedException {
    HttpRequest request =
        HttpRequest.newBuilder().uri(URI.create(url)).timeout(Duration.ofSeconds(30)).GET().build();
    HttpResponse<byte[]> response = client.send(request, HttpResponse.BodyHandlers.ofByteArray());
    if (response.statusCode() != 200) {
      throw new IOException("HTTP " + response.statusCode() + " for " + url);
    }
    return response.body();
  }

  private String downloadText(String url) throws IOException, InterruptedException {
    HttpRequest request =
        HttpRequest.newBuilder().uri(URI.create(url)).timeout(Duration.ofSeconds(30)).GET().build();
    HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
    if (response.statusCode() != 200) {
      throw new IOException("HTTP " + response.statusCode() + " for " + url);
    }
    return response.body();
  }

  private static void writeGzip(Path path, String content) throws IOException {
    try (OutputStream os = new GZIPOutputStream(Files.newOutputStream(path))) {
      os.write(content.getBytes(StandardCharsets.UTF_8));
    }
  }

  private static String decompressGzip(byte[] gzipped) throws IOException {
    try (InputStream is = new GZIPInputStream(new ByteArrayInputStream(gzipped));
        ByteArrayOutputStream out = new ByteArrayOutputStream()) {
      is.transferTo(out);
      return out.toString("UTF-8");
    }
  }

  /**
   * 引数を日付 (yyyy-MM-dd) または年 (yyyy) としてパースする。 年の場合は1月1日を返す (isEnd=false) / 12月31日を返す (isEnd=true)。
   */
  static LocalDate parseArg(String arg, boolean isEnd) {
    if (arg.contains("-")) {
      return LocalDate.parse(arg);
    }
    int year = Integer.parseInt(arg);
    return isEnd ? LocalDate.of(year, 12, 31) : LocalDate.of(year, 1, 1);
  }

  /**
   * 指定期間の鳳凰卓牌譜を保存する CLI を実行する。
   *
   * @param args 保存先、開始日、任意の終了日と {@code --three}
   * @throws Exception HTTP 通信、保存または待機に失敗した場合
   */
  public static void main(String[] args) throws Exception {
    if (args.length < 2) {
      log.error(
          """
          Usage: TenhouArchiveDownloader <outputDir> <from> [to] [--three]
            from/to: yyyy or yyyy-MM-dd\
          """);
      System.exit(1);
    }

    Path outputDir = Path.of(args[0]);
    LocalDate from = parseArg(args[1], false);
    LocalDate to =
        args.length >= 3 && !args[2].startsWith("--")
            ? parseArg(args[2], true)
            : parseArg(args[1], true);
    boolean threePlayer = false;
    for (String arg : args) {
      if ("--three".equals(arg)) {
        threePlayer = true;
      }
    }

    // to が未来の場合は今日に切り詰め
    LocalDate today = LocalDate.now();
    if (to.isAfter(today)) {
      to = today;
    }

    log.info("Downloading houou logs: {} to {}", from, to);
    TenhouArchiveDownloader downloader = new TenhouArchiveDownloader(outputDir, 500, threePlayer);
    int count = downloader.download(from, to);
    log.info("Done. {} logs downloaded to {}", count, outputDir);
  }
}
