package com.epsilon.classic.data;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Stream;

/**
 * 行のディレクトリ。半荘の数で区切って書いた {@link ClassicRows} のファイル（rows-00000.bin から番号の順）を全て読み、行を通し番号で引けるように
 * する。1 つのファイルの O は 1 つの {@code long[]} に入るので、10 万半荘の行（6 千万行、O の long で約 10 億）は、配列の上限を超えないように
 * ファイルを分けて置く。
 *
 * <p>行の通し番号はファイルの順に続き、半荘の通し番号（{@link #gameIndex}）も同じ順に続く。読み込んだ後は変更しないので、どのスレッドから
 * 引いてもよい。ただし、リーダーの {@link Observation} はスレッドごとに持つ。
 */
public final class RowChunks {

  /** 行のファイル名の先頭。 */
  public static final String PREFIX = "rows-";

  /** 行のファイル名の末尾。 */
  public static final String SUFFIX = ".bin";

  /** 並列に読むファイルの数の上限。 */
  private static final int READERS = 8;

  private final Variant variant;
  private final List<ClassicRows> chunks;
  private final List<ClassicRows.Game> games;

  /**
   * 行が 1 つ以上あるファイルごとの、最初の行の通し番号。末尾には行の総数を置く。下の 2 つの配列は同じ並びで、ファイルの番号と、
   * そのファイルの最初の半荘の通し番号を持つ。
   */
  private final int[] rowStarts;

  private final int[] rowChunks;
  private final int[] gameStarts;

  private RowChunks(List<ClassicRows> chunks) {
    if (chunks.isEmpty()) {
      throw new IllegalArgumentException("No rows");
    }
    this.chunks = List.copyOf(chunks);
    variant = chunks.getFirst().variant();
    List<ClassicRows.Game> all = new ArrayList<>();
    int[] starts = new int[chunks.size() + 1];
    int[] owners = new int[chunks.size()];
    int[] gameOffsets = new int[chunks.size()];
    int filled = 0;
    int rows = 0;
    for (int index = 0; index < chunks.size(); index++) {
      ClassicRows chunk = chunks.get(index);
      if (chunk.variant() != variant) {
        throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + chunk.variant());
      }
      if (chunk.size() > 0) {
        starts[filled] = rows;
        owners[filled] = index;
        gameOffsets[filled] = all.size();
        filled++;
        rows = Math.addExact(rows, chunk.size());
      }
      all.addAll(chunk.games());
    }
    starts[filled] = rows;
    rowStarts = Arrays.copyOf(starts, filled + 1);
    rowChunks = Arrays.copyOf(owners, filled);
    gameStarts = Arrays.copyOf(gameOffsets, filled);
    games = List.copyOf(all);
  }

  /** 読み込んだ（または作った）行を、渡した順につなぐ。 */
  public static RowChunks of(List<ClassicRows> chunks) {
    return new RowChunks(chunks);
  }

  /** ディレクトリとファイルの番号から、行のファイルのパスを作る。 */
  public static Path file(Path directory, int chunk) {
    return directory.resolve(String.format(Locale.ROOT, "%s%05d%s", PREFIX, chunk, SUFFIX));
  }

  /** ディレクトリにある行のファイルを、名前の順に返す。 */
  public static List<Path> files(Path directory) throws IOException {
    try (Stream<Path> listed = Files.list(directory)) {
      return listed
          .filter(Files::isRegularFile)
          .filter(file -> {
            String name = file.getFileName().toString();
            return name.startsWith(PREFIX) && name.endsWith(SUFFIX);
          })
          .sorted()
          .toList();
    }
  }

  /**
   * ディレクトリにある行のファイルを全て、並列に読む。
   *
   * @throws IOException 行のファイルが 1 つも無いとき。または読めないファイルがあるとき
   */
  public static RowChunks read(Path directory) throws IOException {
    List<Path> files = files(directory);
    if (files.isEmpty()) {
      throw new IOException("No " + PREFIX + "*" + SUFFIX + " rows in " + directory);
    }
    try (ExecutorService pool = Executors.newFixedThreadPool(Math.min(READERS, files.size()))) {
      List<Future<ClassicRows>> reads = new ArrayList<>(files.size());
      for (Path file : files) {
        reads.add(pool.submit(() -> ClassicRows.read(file)));
      }
      List<ClassicRows> chunks = new ArrayList<>(files.size());
      for (Future<ClassicRows> read : reads) {
        chunks.add(read.get());
      }
      return new RowChunks(chunks);
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while reading the rows of " + directory, interrupted);
    } catch (ExecutionException failed) {
      Throwable cause = failed.getCause();
      if (cause instanceof IOException io) {
        throw io;
      }
      if (cause instanceof UncheckedIOException io) {
        throw io.getCause();
      }
      throw new IOException("Could not read the rows of " + directory, cause);
    }
  }

  public Variant variant() {
    return variant;
  }

  /** 行のファイルの数。 */
  public int chunks() {
    return chunks.size();
  }

  /** 行の総数。 */
  public int size() {
    return rowStarts[rowStarts.length - 1];
  }

  /** 全てのファイルの半荘を、ファイルの順につないだもの。 */
  public List<ClassicRows.Game> games() {
    return games;
  }

  /** リーダーを付け替えて、行の判断の記録を読めるようにする。 */
  public Observation bind(int row, Observation reader) {
    int at = locate(row);
    return chunks.get(rowChunks[at]).bind(row - rowStarts[at], reader);
  }

  /** 行の半荘。 */
  public ClassicRows.Game game(int row) {
    return games.get(gameIndex(row));
  }

  /** 行の半荘の通し番号（{@link #games} の何番目か）。 */
  public int gameIndex(int row) {
    int at = locate(row);
    return gameStarts[at] + chunks.get(rowChunks[at]).gameIndex(row - rowStarts[at]);
  }

  /** 行の局の番号（半荘の何局目か）。 */
  public int round(int row) {
    int at = locate(row);
    return chunks.get(rowChunks[at]).round(row - rowStarts[at]);
  }

  /** 牌譜で選ばれた手（合法手の何番目か）。 */
  public int chosen(int row) {
    int at = locate(row);
    return chunks.get(rowChunks[at]).chosen(row - rowStarts[at]);
  }

  /** 検証用の半荘の行なら true。 */
  public boolean validation(int row) {
    int at = locate(row);
    return chunks.get(rowChunks[at]).validation(row - rowStarts[at]);
  }

  /** 行のあるファイルだけを並べたときに、その行を含むファイルが何番目かを返す。 */
  private int locate(int row) {
    if (row < 0 || row >= size()) {
      throw new IndexOutOfBoundsException("Row " + row + " of " + size());
    }
    int found = Arrays.binarySearch(rowStarts, 0, rowStarts.length - 1, row);
    return found >= 0 ? found : -found - 2;
  }
}
