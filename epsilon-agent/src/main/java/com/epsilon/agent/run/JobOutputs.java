package com.epsilon.agent.run;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.Writer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * ジョブの games.tsv と looks.tsv に行を書き足す。開くときに、job.json に記録した長さまで切り詰める。行を書いた後、job.json を
 * 書く前に落ちたチャンクの行はここで消え、続きを打つときにそのチャンクを打ち直す。
 */
public final class JobOutputs implements AutoCloseable {

  /** 1 行が 1 半荘。 */
  public static final String GAMES = "games.tsv";

  /** 1 行が 1 つの判定の時点。 */
  public static final String LOOKS = "looks.tsv";

  private final Path directory;
  private final Output games;
  private final Output looks;

  private JobOutputs(Path directory, Output games, Output looks) {
    this.directory = directory;
    this.games = games;
    this.looks = looks;
  }

  /**
   * 書き足す先を開く。
   *
   * @param gamesBytes games.tsv の、済んだ部分の長さ（バイト）
   * @param looksBytes looks.tsv の、済んだ部分の長さ（バイト）
   * @throws IOException ファイルが記録した長さより短いとき（記録した後に中身を失った）
   */
  static JobOutputs open(Path directory, long gamesBytes, long looksBytes) throws IOException {
    DurableFiles.createDirectories(directory);
    Output games = Output.open(directory.resolve(GAMES), gamesBytes);
    try {
      return new JobOutputs(directory, games, Output.open(directory.resolve(LOOKS), looksBytes));
    } catch (IOException | RuntimeException failure) {
      games.close();
      throw failure;
    }
  }

  /** games.tsv に 1 行足す（改行はここで付ける）。 */
  public void game(String line) throws IOException {
    games.line(line);
  }

  /** looks.tsv に 1 行足す（改行はここで付ける）。 */
  public void look(String line) throws IOException {
    looks.line(line);
  }

  /** 書いた行を fsync する。 */
  void sync() throws IOException {
    games.sync();
    looks.sync();
    DurableFiles.syncDirectory(directory);
  }

  long gamesBytes() throws IOException {
    return games.length();
  }

  long looksBytes() throws IOException {
    return looks.length();
  }

  @Override
  public void close() throws IOException {
    try {
      games.close();
    } finally {
      looks.close();
    }
  }

  /** 1 つのファイルに行を書き足す。 */
  private static final class Output {

    private final Path file;
    private final FileChannel channel;
    private final Writer writer;

    private Output(Path file, FileChannel channel) {
      this.file = file;
      this.channel = channel;
      this.writer = new BufferedWriter(Channels.newWriter(channel, StandardCharsets.UTF_8));
    }

    static Output open(Path file, long bytes) throws IOException {
      FileChannel channel = FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
      try {
        long size = channel.size();
        if (size < bytes) {
          throw new IOException(file + " holds " + size + " bytes, but the job recorded " + bytes);
        }
        channel.truncate(bytes);
        channel.position(bytes);
        return new Output(file, channel);
      } catch (IOException | RuntimeException failure) {
        channel.close();
        throw failure;
      }
    }

    void line(String line) throws IOException {
      if (line.indexOf('\n') >= 0) {
        throw new IllegalArgumentException("A line of " + file.getFileName() + " must not contain a newline");
      }
      writer.write(line);
      writer.write('\n');
    }

    void sync() throws IOException {
      writer.flush();
      channel.force(true);
    }

    long length() throws IOException {
      writer.flush();
      return channel.position();
    }

    void close() throws IOException {
      try {
        writer.flush();
      } finally {
        channel.close();
      }
    }
  }
}
