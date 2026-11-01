package com.epsilon.prism.policy;

import com.epsilon.prism.observe.Observation;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 測りの付録: 収集の受け付けを間を空けて抜き取り、行ごとの記録 O を書き出す（速さの試験が同じ O から塊を作り直して回す）。系の性質
 * {@value #PROPERTY} が {@code K:N:DIR} のときだけ働き、K 個目・2K 個目…の受け付けを N 個まで DIR に書く（収集の始めの短い行に偏らない
 * よう、最初は K 個目）。性質がなければ何もしない。
 *
 * <p>ファイル {@code prism-v<版>-<受け付けの番号>.batch} は big-endian で、{@link #MAGIC}・行の数・行ごとに（O の long の数・O の
 * long の列）。書くのは答えを書き終えた受け付けの完了のスレッドで、O は受け付けを返すまで書き換わらない。書けなければ記録に書いて抜き取りを
 * やめる（方策は止めない）。
 */
public final class BatchSamples {

  private static final Logger LOG = LoggerFactory.getLogger(BatchSamples.class);

  /** 設定の系の性質。 */
  static final String PROPERTY = "prism.batchSamples";

  /** ファイルの頭の印（"PRSB"）。 */
  static final int MAGIC = 0x50525342;

  private static final Pattern SETTING = Pattern.compile("(\\d+):(\\d+):(.+)");

  private final long every;
  private final long limit;
  private final Path directory;
  private final AtomicLong admitted = new AtomicLong();
  private volatile boolean failed;

  private BatchSamples(long every, long limit, Path directory) {
    this.every = every;
    this.limit = limit;
    this.directory = directory;
  }

  /**
   * 系の性質 {@value #PROPERTY} から作る。
   *
   * @return 抜き取り（性質がなければ何もしない）
   */
  static BatchSamples configured() {
    return of(System.getProperty(PROPERTY));
  }

  /**
   * 設定から作る。
   *
   * @param setting K:N:DIR（K ≥ 1、N ≥ 1）。null か空なら何もしない
   * @return 抜き取り
   * @throws IllegalArgumentException 形が違うとき
   */
  static BatchSamples of(String setting) {
    if (setting == null || setting.isBlank()) {
      return new BatchSamples(0, 0, null);
    }
    Matcher matcher = SETTING.matcher(setting.strip());
    if (!matcher.matches()
        || Long.parseLong(matcher.group(1)) < 1
        || Long.parseLong(matcher.group(2)) < 1) {
      throw new IllegalArgumentException(
          PROPERTY + " must be K:N:DIR with K >= 1 and N >= 1, got " + setting);
    }
    return new BatchSamples(
        Long.parseLong(matcher.group(1)), Long.parseLong(matcher.group(2)), Path.of(matcher.group(3)));
  }

  /**
   * 答えを書き終えた受け付けを数え、抜き取りの番なら行の O を書く。
   *
   * @param rows 受け付けの問い合わせ（O を捕捉したもの）
   * @param version 方策の版
   */
  void offer(List<PolicyRequest> rows, int version) {
    if (every == 0 || failed) {
      return;
    }
    long index = admitted.incrementAndGet();
    if (index % every != 0 || index / every > limit) {
      return;
    }
    Path file =
        directory.resolve(String.format(Locale.ROOT, "prism-v%d-%09d.batch", version, index));
    try {
      write(file, rows);
      LOG.info("PRISM batch sample: {} rows -> {}", rows.size(), file);
    } catch (IOException | RuntimeException problem) {
      failed = true;
      LOG.warn("PRISM batch samples stop: cannot write {}", file, problem);
    }
  }

  /** 行の O を書く（同じ名前のファイルがあれば書かない）。 */
  static void write(Path file, List<PolicyRequest> rows) throws IOException {
    Files.createDirectories(file.getParent());
    Observation read = new Observation();
    try (DataOutputStream out =
        new DataOutputStream(
            new BufferedOutputStream(
                Files.newOutputStream(file, StandardOpenOption.CREATE_NEW), 1 << 16))) {
      out.writeInt(MAGIC);
      out.writeInt(rows.size());
      for (PolicyRequest request : rows) {
        long[] words = request.recordWords();
        int base = request.recordBase();
        int longs = read.bind(words, base).longs();
        out.writeInt(longs);
        for (int word = base; word < base + longs; word++) {
          out.writeLong(words[word]);
        }
      }
    }
  }

  /**
   * 書き出した受け付けを読む。
   *
   * @param file {@link #offer} が書いたファイル
   * @return 行ごとの O（問い合わせの順。各配列の先頭が O の先頭）
   * @throws IOException 読めないか、形が違うとき
   */
  public static List<long[]> read(Path file) throws IOException {
    try (DataInputStream in =
        new DataInputStream(new BufferedInputStream(Files.newInputStream(file), 1 << 16))) {
      if (in.readInt() != MAGIC) {
        throw new IOException("not a PRISM batch sample: " + file);
      }
      int count = in.readInt();
      List<long[]> rows = new ArrayList<>(count);
      for (int row = 0; row < count; row++) {
        long[] words = new long[in.readInt()];
        for (int word = 0; word < words.length; word++) {
          words[word] = in.readLong();
        }
        rows.add(words);
      }
      if (in.read() != -1) {
        throw new IOException("trailing bytes in PRISM batch sample " + file);
      }
      return rows;
    }
  }
}
