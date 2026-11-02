package com.epsilon.prism.luck;

import com.epsilon.prism.collect.LuckSampleSink;
import com.github.luben.zstd.Zstd;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.zip.CRC32C;

/**
 * 1 コホートの運の網 Ω の標本を、コホートのディレクトリの {@value #DIRECTORY}/ に shard として書く。
 *
 * <p>標本は約 1 MiB ずつ枠にまとめて zstd level 1 で圧縮し、展開後の CRC32C を付ける（{@link LuckSampleFormat}）。shard は
 * {@link #SHARD_BYTES} バイトを超える前に次へ移る。{@link #seal} で全ての shard を同期してから目録 {@value #MANIFEST}（形の指紋つき）
 * を原子的に置き、以後このディレクトリは変えない。封印せずに閉じたら目録を置かないので、読む側は途中のコホートとして拒む。
 *
 * <p>呼ぶ場所: 標本を集める収集の呼び手がコホートの行の書き手と一緒に作り（{@link LuckSampleSink}）、収集の後に封印する。{@link #append} は
 * {@link LuckBatch#settle} が呼ぶ。{@link #append} は排他で、圧縮は呼んだスレッドで行う。書けなかったら閉じ、以後の呼び出しは例外にする。
 */
public final class LuckSampleWriter implements Closeable {

  /** コホートの中の標本のディレクトリ。 */
  public static final String DIRECTORY = "luck";

  /** 目録のファイル。 */
  public static final String MANIFEST = "luck.json";

  /** shard の大きさの上限の目安。 */
  static final long SHARD_BYTES = 256L << 20;

  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  /**
   * shard 1 つ。
   *
   * @param file ファイルの名前（標本のディレクトリの中）
   * @param samples 標本の数
   * @param bytes バイト数
   */
  public record Shard(String file, long samples, long bytes) {}

  /**
   * 封印したコホートの標本の目録。
   *
   * @param fingerprint 形の指紋（O° の指紋・枠・標本の並び・層の定義）
   * @param iteration コホートの番号
   * @param samples 全ての shard の標本の数
   * @param bytes 全ての shard のバイト数
   * @param shards shard（書いた順）
   */
  public record Manifest(
      String fingerprint, int iteration, long samples, long bytes, List<Shard> shards) {}

  private final Path directory;
  private final int iteration;
  private final ByteBuffer block =
      ByteBuffer.allocate(LuckSampleFormat.BLOCK).order(ByteOrder.LITTLE_ENDIAN);
  private final ByteBuffer frameHeader =
      ByteBuffer.allocate(LuckSampleFormat.FRAME_HEADER).order(ByteOrder.LITTLE_ENDIAN);
  private final CRC32C crc = new CRC32C();
  private final List<Shard> shards = new ArrayList<>();
  private FileChannel channel;
  private String file;
  private long shardSamples;
  private long shardBytes;
  private int blockSamples;
  private long samples;
  private long bytes;
  private boolean closed;

  /**
   * コホートの標本の書き手を作る。
   *
   * @param cohort コホートのディレクトリ（標本は {@value #DIRECTORY}/ に書く）
   * @param iteration コホートの番号
   * @throws IOException ディレクトリを作れないか、すでに封印されているとき
   */
  public LuckSampleWriter(Path cohort, int iteration) throws IOException {
    directory = directory(cohort);
    this.iteration = iteration;
    Files.createDirectories(directory);
    if (Files.exists(directory.resolve(MANIFEST))) {
      throw new IOException("Luck samples are already sealed: " + directory);
    }
  }

  /**
   * コホートの標本のディレクトリを返す。
   *
   * @param cohort コホートのディレクトリ
   * @return 標本のディレクトリ
   */
  public static Path directory(Path cohort) {
    return cohort.resolve(DIRECTORY);
  }

  /**
   * 標本を 1 件足す。
   *
   * @param game 半荘 ID（保留の分け方に使う）
   * @param words O° を置いた配列
   * @param base O° の先頭
   * @param y 局末の Φ（O° と同じく行為者から見た相対席の順に 4 席）
   * @param stratum 層の鍵（{@link LuckStratum}）
   * @param weight 重み（1/標本の率）
   * @throws IOException 枠を書けなかったとき
   * @throws IllegalArgumentException Y・層・重みが範囲の外のとき
   * @throws IllegalStateException 閉じた後のとき
   */
  public synchronized void append(
      long game, long[] words, int base, float[] y, int stratum, float weight) throws IOException {
    requireOpen();
    if (block.remaining() < LuckSampleFormat.MAX_SAMPLE) {
      flushBlock();
    }
    LuckSampleFormat.put(block, game, words, base, y, stratum, weight);
    blockSamples++;
  }

  /**
   * 標本を 1 件足す。
   *
   * @param sample 標本
   * @throws IOException 枠を書けなかったとき
   */
  public void append(LuckSample sample) throws IOException {
    append(sample.game(), sample.observation(), 0, sample.y(), sample.stratum(), sample.weight());
  }

  /**
   * 残りを書き、同期して目録を置く。以後は書けない。
   *
   * @return 目録
   * @throws IOException 書けなかったとき
   * @throws IllegalStateException 閉じた後のとき
   */
  public synchronized Manifest seal() throws IOException {
    requireOpen();
    flushBlock();
    try {
      finishShard();
      Manifest manifest =
          new Manifest(
              LuckSampleFormat.fingerprint(), iteration, samples, bytes, List.copyOf(shards));
      writeManifest(manifest);
      closed = true;
      return manifest;
    } catch (IOException | RuntimeException failure) {
      abandon(failure);
      throw failure;
    }
  }

  /** 封印していなければ目録を置かずに閉じる（途中のコホートとして残る）。 */
  @Override
  public synchronized void close() throws IOException {
    if (closed) {
      return;
    }
    closed = true;
    if (channel != null) {
      channel.close();
      channel = null;
    }
  }

  /**
   * 封印した目録を読む。形の指紋は照らさない（照らすのは {@link LuckSampleStore}）。
   *
   * @param directory 標本のディレクトリ
   * @return 目録
   * @throws IOException 封印されていないとき
   */
  static Manifest readManifest(Path directory) throws IOException {
    Path path = directory.resolve(MANIFEST);
    if (!Files.exists(path)) {
      throw new IOException("Luck samples are not sealed: " + directory);
    }
    Manifest manifest = GSON.fromJson(Files.readString(path), Manifest.class);
    if (manifest == null || manifest.fingerprint() == null || manifest.shards() == null) {
      throw new IOException("Corrupt luck sample manifest: " + path);
    }
    return manifest;
  }

  /** まとめた標本を 1 つの枠にして書く。 */
  private void flushBlock() throws IOException {
    if (blockSamples == 0) {
      return;
    }
    try {
      byte[] raw = Arrays.copyOf(block.array(), block.position());
      byte[] compressed = Zstd.compress(raw, LuckSampleFormat.LEVEL);
      crc.reset();
      crc.update(raw);
      int frameLength = LuckSampleFormat.FRAME_HEADER + compressed.length;
      if (channel == null || shardBytes > 0 && shardBytes + frameLength > SHARD_BYTES) {
        finishShard();
        openShard();
      }
      frameHeader.clear();
      frameHeader.putInt(compressed.length).putInt(raw.length).putInt((int) crc.getValue());
      frameHeader.putInt(blockSamples).flip();
      writeFully(frameHeader);
      writeFully(ByteBuffer.wrap(compressed));
      shardSamples += blockSamples;
      shardBytes += frameLength;
      samples += blockSamples;
      bytes += frameLength;
      block.clear();
      blockSamples = 0;
    } catch (IOException | RuntimeException failure) {
      abandon(failure);
      throw failure;
    }
  }

  private void openShard() throws IOException {
    file = String.format("shard-%05d.luck", shards.size());
    channel =
        FileChannel.open(
            directory.resolve(file),
            StandardOpenOption.CREATE,
            StandardOpenOption.TRUNCATE_EXISTING,
            StandardOpenOption.WRITE);
    shardSamples = 0;
    shardBytes = 0;
  }

  private void finishShard() throws IOException {
    if (channel == null) {
      return;
    }
    channel.force(true);
    channel.close();
    channel = null;
    shards.add(new Shard(file, shardSamples, shardBytes));
  }

  private void writeFully(ByteBuffer source) throws IOException {
    while (source.hasRemaining()) {
      channel.write(source);
    }
  }

  /** 目録を同期して原子的に置く。 */
  private void writeManifest(Manifest manifest) throws IOException {
    Path temporary = directory.resolve(MANIFEST + ".tmp");
    try (FileChannel out =
        FileChannel.open(
            temporary,
            StandardOpenOption.CREATE,
            StandardOpenOption.TRUNCATE_EXISTING,
            StandardOpenOption.WRITE)) {
      ByteBuffer json = ByteBuffer.wrap(GSON.toJson(manifest).getBytes(StandardCharsets.UTF_8));
      while (json.hasRemaining()) {
        out.write(json);
      }
      out.force(true);
    }
    Files.move(temporary, directory.resolve(MANIFEST), StandardCopyOption.ATOMIC_MOVE);
  }

  /** 書けなかったので閉じる（目録は置かない）。 */
  private void abandon(Exception failure) {
    closed = true;
    if (channel != null) {
      try {
        channel.close();
      } catch (IOException closing) {
        failure.addSuppressed(closing);
      }
      channel = null;
    }
  }

  private void requireOpen() {
    if (closed) {
      throw new IllegalStateException("luck sample writer is closed: " + directory);
    }
  }
}
