package com.epsilon.prism.learn;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.rows.RowCodec;
import java.io.EOFException;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executor;
import java.util.function.Predicate;

/**
 * 確定済みのコホート（rows/&lt;t&gt;。{@link CohortWriter} のマニフェストと .idx）の行を読む。メモリに持つのはインデックス
 * （位置・長さ・用途・種類・クラス・測った手の数）だけで、行の本体は位置を指定して読む。行の番号は、コホートを読み込んだ順に続けて振る。
 * {@link #select} で、コホートの範囲とインデックスの条件（{@link Selection}）から行を選ぶ。
 *
 * <p>行をどう直列化するかはマニフェストの値で決まる。だから帳簿の特徴量の数とルールは、全てのコホートで同じでなければならない。
 * ルールは、マニフェストにある記録 O のフィンガープリントで決まる。
 */
public final class CohortStore implements AutoCloseable {

  /** スレッドごとの読み込み用バッファ。復号は圧縮部分を新しい配列へ写すので、使い回してよい。 */
  private static final ThreadLocal<ByteBuffer> BUFFERS =
      ThreadLocal.withInitial(() -> ByteBuffer.allocateDirect(1 << 16));

  /** スレッドごとのコーデック（記録 O を読む作業領域を持つ）。帳簿の特徴量の数かルールが違えば作り直す。 */
  private static final ThreadLocal<RowCodec> CODECS = new ThreadLocal<>();

  /** 読み込んだコホートと、その行の範囲 [first, end)。 */
  private record Cohort(int iteration, int first, int end) {}

  private final List<FileChannel> files = new ArrayList<>();
  private final List<Cohort> cohorts = new ArrayList<>();
  private int ledgerFeatures = -1;
  private Variant variant;
  private int size;
  private int[] file = new int[0];
  private long[] offset = new long[0];
  private RowCodec.Index[] index = new RowCodec.Index[0];

  /**
   * 確定済みのコホートの全ての行をインデックスに載せる。
   *
   * @param directory コホートのディレクトリ
   * @throws IOException 確定していないとき、記録のフィンガープリント・ルール・帳簿の特徴量の数・シャードが合わないとき、
   *     または同じコホートを既に読んだとき
   */
  public synchronized void addCohort(Path directory) throws IOException {
    CohortWriter.Manifest manifest = CohortWriter.readManifest(directory);
    for (Cohort cohort : cohorts) {
      if (cohort.iteration() == manifest.iteration()) {
        throw new IOException(
            "Cohort " + manifest.iteration() + " is already loaded: " + directory);
      }
    }
    if (ledgerFeatures >= 0 && manifest.ledgerFeatures() != ledgerFeatures) {
      throw new IOException(
          "Cohort "
              + directory
              + " has "
              + manifest.ledgerFeatures()
              + " ledger features, the store "
              + ledgerFeatures);
    }
    Variant cohortVariant = manifest.variant();
    if (variant != null && cohortVariant != variant) {
      throw new IOException("Cohort " + directory + " holds " + cohortVariant + " rows, the store " + variant);
    }
    int first = size;
    for (CohortWriter.Shard shard : manifest.shards()) {
      addShard(directory.resolve(shard.file()), shard);
    }
    ledgerFeatures = manifest.ledgerFeatures();
    variant = cohortVariant;
    cohorts.add(new Cohort(manifest.iteration(), first, size));
  }

  /** シャードの .idx を先頭から読み、行の位置を載せる。 */
  private void addShard(Path path, CohortWriter.Shard shard) throws IOException {
    FileChannel channel = FileChannel.open(path, StandardOpenOption.READ);
    try {
      if (channel.size() != shard.bytes()) {
        throw new IOException("Shard size mismatch " + path);
      }
    } catch (IOException | RuntimeException failure) {
      channel.close();
      throw failure;
    }
    int fileIndex = files.size();
    files.add(channel);
    long position = 0;
    long counted = 0;
    try (FileChannel entriesFile =
        FileChannel.open(CohortWriter.index(path), StandardOpenOption.READ)) {
      ByteBuffer entries =
          ByteBuffer.allocateDirect(CohortWriter.INDEX_ENTRY << 13)
              .order(ByteOrder.LITTLE_ENDIAN);
      entries.limit(0);
      while (true) {
        if (entries.remaining() < CohortWriter.INDEX_ENTRY) {
          entries.compact();
          while (entries.hasRemaining() && entriesFile.read(entries) >= 0) {}
          entries.flip();
          if (entries.remaining() < CohortWriter.INDEX_ENTRY) {
            break;
          }
        }
        int frameLength = entries.getInt();
        RowCodec.Index entry =
            RowCodec.index(frameLength, entries.get(), entries.get(), entries.get(), entries.get());
        append(fileIndex, position, entry);
        position += frameLength;
        counted++;
      }
    }
    if (position != shard.bytes() || counted != shard.rows()) {
      throw new IOException("Shard index mismatch " + path);
    }
  }

  private void append(int fileIndex, long position, RowCodec.Index entry) {
    if (size == file.length) {
      int capacity = Math.max(1024, size * 2);
      file = Arrays.copyOf(file, capacity);
      offset = Arrays.copyOf(offset, capacity);
      index = Arrays.copyOf(index, capacity);
    }
    file[size] = fileIndex;
    offset[size] = position;
    index[size] = entry;
    size++;
  }

  /**
   * 載せた行の数を返す。
   *
   * @return 行の数
   */
  public synchronized int size() {
    return size;
  }

  /**
   * 読み込んだコホートの番号を、読み込んだ順に返す。
   *
   * @return コホートの番号
   */
  public synchronized List<Integer> cohorts() {
    List<Integer> result = new ArrayList<>(cohorts.size());
    for (Cohort cohort : cohorts) {
      result.add(cohort.iteration());
    }
    return result;
  }

  /**
   * 載せたコホートのルールを返す。
   *
   * @return ルール（記録 O のフィンガープリントで決まる）
   * @throws IllegalStateException コホートを 1 つも載せていないとき
   */
  public synchronized Variant variant() {
    if (variant == null) {
      throw new IllegalStateException("The store holds no cohort");
    }
    return variant;
  }

  public synchronized RowCodec.Index index(int row) {
    return index[row];
  }

  /**
   * 指定した範囲のコホートから、インデックスが条件に合う行を、コホートの番号の順に返す。読み込んでいないコホートは飛ばす。
   * 行の本体は展開しない。
   *
   * @param from 最初のコホート（含む）
   * @param to 最後のコホート（含む）
   * @param filter インデックスの条件
   * @return 行の番号
   */
  public synchronized int[] select(int from, int to, Predicate<RowCodec.Index> filter) {
    int[] rows = new int[size];
    int count = 0;
    for (int iteration = from; iteration <= to; iteration++) {
      for (Cohort cohort : cohorts) {
        if (cohort.iteration() != iteration) {
          continue;
        }
        for (int row = cohort.first(); row < cohort.end(); row++) {
          if (filter.test(index[row])) {
            rows[count++] = row;
          }
        }
      }
    }
    return Arrays.copyOf(rows, count);
  }

  /**
   * 1 行を展開する。位置を指定して読むので、どのスレッドからでも並行に呼べる。
   *
   * @param row 行の番号
   * @return 読んだ行
   * @throws IOException 読めないか、フレームが壊れているとき
   */
  public Row read(int row) throws IOException {
    FileChannel channel;
    long position;
    int bytes;
    int features;
    Variant rowVariant;
    synchronized (this) {
      channel = files.get(file[row]);
      position = offset[row];
      bytes = index[row].frameLength();
      features = ledgerFeatures;
      rowVariant = variant;
    }
    ByteBuffer buffer = BUFFERS.get();
    if (buffer.capacity() < bytes) {
      buffer = ByteBuffer.allocateDirect(Integer.highestOneBit(bytes) << 1);
      BUFFERS.set(buffer);
    }
    buffer.clear().limit(bytes);
    while (buffer.hasRemaining()) {
      if (channel.read(buffer, position + buffer.position()) < 0) {
        throw new EOFException("Row frame past the end of its shard");
      }
    }
    buffer.flip();
    RowCodec codec = CODECS.get();
    if (codec == null || codec.ledgerFeatures() != features || codec.variant() != rowVariant) {
      codec = new RowCodec(features, rowVariant);
      CODECS.set(codec);
    }
    Row decoded = codec.decode(buffer);
    if (buffer.hasRemaining()) {
      throw new IOException("Row frame shorter than its index entry");
    }
    return decoded;
  }

  /**
   * 行をまとめて展開する。行を続いた区間に分けて並列に読む。呼んだスレッドは読み終わるまで待つ。
   *
   * @param rows 行の番号
   * @param executor 読むのに使うスレッド
   * @param parts 区間の数
   * @return 読んだ行（渡した番号の順）
   * @throws IOException どれかの行が読めないとき
   */
  public Row[] read(int[] rows, Executor executor, int parts) throws IOException {
    Row[] read = new Row[rows.length];
    int chunk = Math.max(1, Math.ceilDiv(rows.length, Math.max(1, parts)));
    List<CompletableFuture<Void>> futures = new ArrayList<>();
    for (int from = 0; from < rows.length; from += chunk) {
      int start = from;
      int end = Math.min(rows.length, from + chunk);
      futures.add(
          CompletableFuture.runAsync(
              () -> {
                for (int index = start; index < end; index++) {
                  try {
                    read[index] = read(rows[index]);
                  } catch (IOException failure) {
                    throw new UncheckedIOException(failure);
                  }
                }
              },
              executor));
    }
    try {
      CompletableFuture.allOf(futures.toArray(CompletableFuture[]::new)).join();
    } catch (CompletionException failure) {
      Throwable cause = failure.getCause();
      if (cause instanceof UncheckedIOException unreadable) {
        throw unreadable.getCause();
      }
      if (cause instanceof RuntimeException runtime) {
        throw runtime;
      }
      if (cause instanceof Error error) {
        throw error;
      }
      throw failure;
    }
    return read;
  }

  /**
   * 行を順に読む {@link Iterable} を返す。運の精算の当てはめのように、全ての行を 1 度ずつ見る処理に使う。決まった行数ずつ
   * {@link #read(int[], Executor, int)} で並列に読み、メモリには 1 回に読んだ分だけを持つ。読めない行があると
   * {@link UncheckedIOException} を投げて止まる。
   *
   * @param rows 行の番号
   * @param chunk 1 度に読む行の数
   * @param executor 読むのに使うスレッド
   * @param parts 1 度に読む分を分ける区間の数
   * @return 渡した番号の順に行を返す {@link Iterable}（回すたびに読み直す）
   */
  public Iterable<Row> iterate(int[] rows, int chunk, Executor executor, int parts) {
    if (chunk < 1) {
      throw new IllegalArgumentException("chunk must be positive: " + chunk);
    }
    return () ->
        new Iterator<>() {
          private Row[] block = new Row[0];
          private int at;
          private int from;

          @Override
          public boolean hasNext() {
            return at < block.length || from < rows.length;
          }

          @Override
          public Row next() {
            if (at == block.length) {
              if (from >= rows.length) {
                throw new NoSuchElementException();
              }
              int to = Math.min(rows.length, from + chunk);
              try {
                block = read(Arrays.copyOfRange(rows, from, to), executor, parts);
              } catch (IOException failure) {
                throw new UncheckedIOException(failure);
              }
              from = to;
              at = 0;
            }
            return block[at++];
          }
        };
  }

  @Override
  public synchronized void close() throws IOException {
    IOException failure = null;
    for (FileChannel channel : files) {
      try {
        channel.close();
      } catch (IOException closing) {
        if (failure == null) {
          failure = closing;
        } else {
          failure.addSuppressed(closing);
        }
      }
    }
    files.clear();
    cohorts.clear();
    size = 0;
    variant = null;
    if (failure != null) {
      throw failure;
    }
  }
}
