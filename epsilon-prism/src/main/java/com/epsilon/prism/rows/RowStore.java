package com.epsilon.prism.rows;

import com.epsilon.prism.core.CrnSeeds;
import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Predicate;

/**
 * 封印済みのコホートの行を読む。索引（位置・長さ・用途・組・種類・類・測った手の数）だけをメモリに持つ（1行は数十バイト）。行はコホートごとに続けて
 * 並び、{@link #select} でコホートと索引の条件から選ぶ。
 *
 * <p>索引は shard ごとの .idx（{@link RowWriter#index}。{@link RowWriter#INDEX_ENTRY} バイトの記録が行の順に並ぶ）を頭から読んで作るので、shard
 * 本体には触らない。行の本体は位置指定で読む（pread）。
 */
public final class RowStore implements RowSource, AutoCloseable {
  /** スレッドごとの読み込みバッファ。復号は圧縮部分を新しい配列へ写すので、使い回してよい。 */
  private static final ThreadLocal<ByteBuffer> BUFFERS =
      ThreadLocal.withInitial(() -> ByteBuffer.allocateDirect(1 << 16));

  /** 読み込んだコホートと、その行の範囲 [first, end)。 */
  private record Cohort(int iteration, int first, int end) {}

  private final List<FileChannel> files = new ArrayList<>();
  private final List<Cohort> cohorts = new ArrayList<>();
  private int size;
  private int[] file = new int[0];
  private long[] offset = new long[0];
  private RowFormat.Index[] index = new RowFormat.Index[0];

  /** 封印済みのコホートの全行を索引に載せる。 */
  public void addCohort(Path directory) throws IOException {
    addCohort(directory, 1, 0);
  }

  /**
   * 封印済みのコホートの行のうち割合 keep を索引に載せる（データの記憶）。コホートの中の行の番号 r（shard の順に数える）ごとに
   * unit(derive(salt, MEMORY, コホート, r)) &lt; keep の行だけを載せるので、同じ salt なら何度読んでも同じ行になる。keep ≥ 1 なら全行。
   *
   * @param keep 載せる行の割合（(0, 1]）
   * @param salt 間引きの種（run の seed）
   */
  public synchronized void addCohort(Path directory, double keep, long salt) throws IOException {
    if (!(keep > 0)) throw new IllegalArgumentException("keep must be positive: " + keep);
    RowWriter.Manifest manifest = RowWriter.readManifest(directory);
    int first = size;
    long ordinal = 0;
    for (RowWriter.Shard shard : manifest.shards()) {
      Path path = directory.resolve(shard.file());
      FileChannel channel = FileChannel.open(path, StandardOpenOption.READ);
      long bytes;
      try {
        bytes = channel.size();
        if (bytes != shard.bytes()) throw new IOException("Shard size mismatch " + path);
      } catch (IOException | RuntimeException failure) {
        channel.close();
        throw failure;
      }
      int fileIndex = files.size();
      files.add(channel);
      long position = 0, counted = 0;
      try (FileChannel entriesFile =
          FileChannel.open(RowWriter.index(path), StandardOpenOption.READ)) {
        ByteBuffer entries =
            ByteBuffer.allocateDirect(RowWriter.INDEX_ENTRY << 13).order(ByteOrder.LITTLE_ENDIAN);
        entries.limit(0);
        while (true) {
          if (entries.remaining() < RowWriter.INDEX_ENTRY) {
            entries.compact();
            while (entries.hasRemaining() && entriesFile.read(entries) >= 0) {}
            entries.flip();
            if (entries.remaining() < RowWriter.INDEX_ENTRY) break;
          }
          int frameLength = entries.getInt();
          RowFormat.Index entry =
              RowFormat.index(
                  frameLength,
                  entries.get(),
                  entries.get(),
                  entries.get(),
                  entries.get(),
                  entries.get());
          if (keep >= 1
              || CrnSeeds.unit(
                      CrnSeeds.derive(salt, CrnSeeds.Domain.MEMORY, manifest.iteration(), ordinal))
                  < keep) append(fileIndex, position, entry);
          ordinal++;
          position += frameLength;
          counted++;
        }
      }
      if (position != bytes || counted != shard.rows())
        throw new IOException("Shard index mismatch " + path);
    }
    cohorts.add(new Cohort(manifest.iteration(), first, size));
  }

  private void append(int fileIndex, long position, RowFormat.Index entry) {
    if (size == file.length) grow(Math.max(1024, size * 2));
    file[size] = fileIndex;
    offset[size] = position;
    index[size] = entry;
    size++;
  }

  private void grow(int capacity) {
    file = Arrays.copyOf(file, capacity);
    offset = Arrays.copyOf(offset, capacity);
    index = Arrays.copyOf(index, capacity);
  }

  public synchronized int size() {
    return size;
  }

  /** 読み込んだコホートの番号（読み込んだ順）。 */
  public synchronized List<Integer> cohorts() {
    List<Integer> result = new ArrayList<>(cohorts.size());
    for (Cohort cohort : cohorts) result.add(cohort.iteration());
    return result;
  }

  /** 行 row の索引。 */
  public synchronized RowFormat.Index index(int row) {
    return index[row];
  }

  @Override
  public synchronized int[] select(int iteration, Predicate<RowFormat.Index> filter) {
    for (Cohort cohort : cohorts) {
      if (cohort.iteration() != iteration) continue;
      int[] rows = new int[cohort.end() - cohort.first()];
      int count = 0;
      for (int row = cohort.first(); row < cohort.end(); row++)
        if (filter.test(index[row])) rows[count++] = row;
      return Arrays.copyOf(rows, count);
    }
    return new int[0];
  }

  /** コホート from..to（両端を含む、読み込んでいないものは飛ばす）の索引が filter に合う行（コホートの順）。 */
  public int[] select(int from, int to, Predicate<RowFormat.Index> filter) {
    int[][] parts = new int[Math.max(0, to - from + 1)][];
    int total = 0;
    for (int c = from; c <= to; c++) total += (parts[c - from] = select(c, filter)).length;
    int[] rows = new int[total];
    int at = 0;
    for (int[] part : parts) {
      System.arraycopy(part, 0, rows, at, part.length);
      at += part.length;
    }
    return rows;
  }

  /** 1行を展開する。位置指定の読み出しは並行にできる。 */
  @Override
  public PrismRow read(int row) throws IOException {
    FileChannel channel;
    long position;
    int bytes;
    synchronized (this) {
      channel = files.get(file[row]);
      position = offset[row];
      bytes = index[row].frameLength();
    }
    ByteBuffer buffer = BUFFERS.get();
    if (buffer.capacity() < bytes) {
      buffer = ByteBuffer.allocateDirect(Integer.highestOneBit(bytes) << 1);
      BUFFERS.set(buffer);
    }
    buffer.clear().limit(bytes);
    while (buffer.hasRemaining())
      if (channel.read(buffer, position + buffer.position()) < 0)
        throw new EOFException("Row frame past the end of its shard");
    buffer.flip();
    return RowFormat.decodeFrame(buffer.order(ByteOrder.LITTLE_ENDIAN));
  }

  @Override
  public synchronized void close() throws IOException {
    IOException failure = null;
    for (FileChannel channel : files) {
      try {
        channel.close();
      } catch (IOException closing) {
        if (failure == null) failure = closing;
        else failure.addSuppressed(closing);
      }
    }
    files.clear();
    cohorts.clear();
    size = 0;
    if (failure != null) throw failure;
  }
}
