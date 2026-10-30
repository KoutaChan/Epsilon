package com.epsilon.prism.learn;

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
 * 封印済みのコホート（rows-v2/&lt;t&gt;。{@link CohortWriter} の目録と .idx）の行を読む。索引
 * （位置・長さ・用途・組・種類・類・測った手の数）だけをメモリに持ち、行の本体は位置指定で読む。行はコホートを読み込んだ順に続けて並び、{@link #select} で
 * コホートの範囲と索引の条件（{@link Selection}）から選ぶ。
 *
 * <p>帳簿の欄の数は全てのコホートで同じでなければならない（行の直列化が目録の値で決まるため）。
 */
public final class CohortStore implements AutoCloseable {

  /** スレッドごとの読み込みの領域。復号は圧縮部分を新しい配列へ写すので、使い回してよい。 */
  private static final ThreadLocal<ByteBuffer> BUFFERS =
      ThreadLocal.withInitial(() -> ByteBuffer.allocateDirect(1 << 16));

  /** スレッドごとの直列化（O を読む作業域を持つ）。帳簿の欄の数が違えば作り直す。 */
  private static final ThreadLocal<RowCodec> CODECS = new ThreadLocal<>();

  /** 読み込んだコホートと、その行の範囲 [first, end)。 */
  private record Cohort(int iteration, int first, int end) {}

  private final List<FileChannel> files = new ArrayList<>();
  private final List<Cohort> cohorts = new ArrayList<>();
  private int ledgerFeatures = -1;
  private int size;
  private int[] file = new int[0];
  private long[] offset = new long[0];
  private RowCodec.Index[] index = new RowCodec.Index[0];

  /**
   * 封印済みのコホートの全ての行を索引に載せる。
   *
   * @param directory コホートのディレクトリ
   * @throws IOException 封印されていないか、記録の指紋・帳簿の欄の数・シャードが合わないか、同じコホートを既に読んだとき
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
    int first = size;
    for (CohortWriter.Shard shard : manifest.shards()) {
      addShard(directory.resolve(shard.file()), shard);
    }
    ledgerFeatures = manifest.ledgerFeatures();
    cohorts.add(new Cohort(manifest.iteration(), first, size));
  }

  /** シャードの .idx を頭から読み、行の位置を載せる。 */
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
            RowCodec.index(
                frameLength,
                entries.get(),
                entries.get(),
                entries.get(),
                entries.get(),
                entries.get());
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

  public synchronized RowCodec.Index index(int row) {
    return index[row];
  }

  /**
   * コホート from..to（両端を含む。読み込んでいないものは飛ばす）の、索引が filter に合う行を、コホートの番号の順に返す。本体は展開しない。
   *
   * @param from 最初のコホート
   * @param to 最後のコホート
   * @param filter 索引の条件
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
   * 1 行を展開する。位置指定の読み出しなので、どのスレッドからでも並行に呼べる。
   *
   * @param row 行
   * @return 行
   * @throws IOException 読めないか、フレームが壊れているとき
   */
  public Row read(int row) throws IOException {
    FileChannel channel;
    long position;
    int bytes;
    int features;
    synchronized (this) {
      channel = files.get(file[row]);
      position = offset[row];
      bytes = index[row].frameLength();
      features = ledgerFeatures;
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
    if (codec == null || codec.ledgerFeatures() != features) {
      codec = new RowCodec(features);
      CODECS.set(codec);
    }
    Row decoded = codec.decode(buffer);
    if (buffer.hasRemaining()) {
      throw new IOException("Row frame shorter than its index entry");
    }
    return decoded;
  }

  /**
   * 行をまとめて展開する。行を parts 個の続いた区間に分け、executor で並列に読む（呼んだスレッドは終わるまで待つ）。
   *
   * @param rows 行
   * @param executor 読むスレッド
   * @param parts 区間の数
   * @return 行（rows の順）
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
   * 行を順に読む列を返す（運の精算の当てはめのように、全ての行を 1 度ずつ見る読み手のため）。chunk 行ずつ {@link #read(int[], Executor, int)} で
   * 並列に読み、メモリには 1 塊だけを持つ。読めない行は {@link UncheckedIOException} で止まる。
   *
   * @param rows 行
   * @param chunk 1 度に読む行の数
   * @param executor 読むスレッド
   * @param parts 1 塊を分ける区間の数
   * @return rows の順の行の列（列を回すたびに読み直す）
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
    if (failure != null) {
      throw failure;
    }
  }
}
