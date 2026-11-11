package com.epsilon.agent.run;

import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

/**
 * run のファイルの書き方。どの書き方でも、途中で落ちた後に見えるのは、前の中身か新しい中身のどちらかだけになる。
 *
 * <ul>
 *   <li>ファイル: 同じディレクトリの一時ファイルに書いて fsync し、rename で置き換えてから、ディレクトリを fsync する。
 *   <li>段の出力（{@code .part}）: 中の全てを fsync し、完了後の名前がまだ無いときだけ rename してから、親を fsync する。
 *   <li>jsonl: 1 行ずつ足して fsync する。最後の行が改行で終わっていなければ書きかけとみなし、読むときは飛ばし、次に足すときに切り捨てる。
 * </ul>
 *
 * rename がアトミックでないファイルシステムでは失敗する。ディレクトリの fsync は POSIX のファイルシステムでだけ行う。
 */
public final class DurableFiles {

  private static final boolean POSIX = FileSystems.getDefault().supportedFileAttributeViews().contains("posix");

  private DurableFiles() {}

  /** 中身を書く。渡されたストリームは閉じない。 */
  @FunctionalInterface
  public interface Writer {
    void write(OutputStream stream) throws IOException;
  }

  public static void write(Path target, byte[] bytes) throws IOException {
    write(target, stream -> stream.write(bytes));
  }

  /** UTF-8 の文字列を書く。 */
  public static void write(Path target, String text) throws IOException {
    write(target, text.getBytes(StandardCharsets.UTF_8));
  }

  /** record を 1 行の JSON（末尾に改行）で書く。{@link RecordFile#read} で読める。 */
  public static void write(Path target, Record record) throws IOException {
    write(target, RecordFile.json(record) + "\n");
  }

  public static void write(Path target, Writer writer) throws IOException {
    Path destination = target.toAbsolutePath().normalize();
    Path directory = destination.getParent();
    createDirectories(directory);
    Path temporary = Files.createTempFile(directory, "." + destination.getFileName() + ".", ".tmp");
    try {
      try (FileChannel channel = FileChannel.open(temporary, StandardOpenOption.WRITE)) {
        OutputStream stream = Channels.newOutputStream(channel);
        writer.write(stream);
        stream.flush();
        channel.force(true);
      }
      Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    } finally {
      Files.deleteIfExists(temporary);
    }
    syncDirectory(directory);
  }

  /**
   * 書き終えた段の出力（ディレクトリかファイル）を、完了後の名前に rename して確定する。中の全てを fsync してから rename し、
   * 親のディレクトリを fsync する。
   *
   * @param part 書き終えた出力
   * @param done 完了後の名前
   * @throws FileAlreadyExistsException 完了後の名前がもうあるとき（済んだ段は回さないので、呼び出し側のバグ）
   */
  public static void seal(Path part, Path done) throws IOException {
    if (Files.exists(done, LinkOption.NOFOLLOW_LINKS)) {
      throw new FileAlreadyExistsException(done.toString(), part.toString(), "the output is already sealed");
    }
    syncTree(part);
    Files.move(part, done, StandardCopyOption.ATOMIC_MOVE);
    syncDirectory(done.toAbsolutePath().normalize().getParent());
  }

  /**
   * jsonl に 1 行足して fsync する。最後の行が改行で終わっていなければ（書きかけのまま落ちた行）、先に切り捨てる。
   *
   * @param line 改行を含まない 1 行
   */
  public static void append(Path file, String line) throws IOException {
    if (line.indexOf('\n') >= 0) {
      throw new IllegalArgumentException("A jsonl line must not contain a newline");
    }
    Path absolute = file.toAbsolutePath().normalize();
    createDirectories(absolute.getParent());
    boolean created = !Files.exists(absolute);
    try (FileChannel channel =
        FileChannel.open(absolute, StandardOpenOption.CREATE, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
      long keep = lineStart(channel, channel.size());
      if (keep < channel.size()) {
        channel.truncate(keep);
      }
      ByteBuffer buffer = ByteBuffer.wrap((line + "\n").getBytes(StandardCharsets.UTF_8));
      long position = keep;
      while (buffer.hasRemaining()) {
        position += channel.write(buffer, position);
      }
      channel.force(true);
    }
    if (created) {
      syncDirectory(absolute.getParent());
    }
  }

  /**
   * jsonl に、キーの値が同じ行がまだ無ければ 1 行足す。落ちた後にやり直しても、同じ行を二度書かない。
   *
   * @param key 行を見分けるキー（反復の番号、ジョブの名前）
   * @param value 足す行が持つ、そのキーの値
   * @param line 足す行
   * @return 足したら true
   */
  public static boolean appendOnce(Path file, String key, Node value, Node.Table line) throws IOException {
    List<String> lines = lines(file);
    for (int index = 0; index < lines.size(); index++) {
      if (value.equals(Json.read(lines.get(index), file + ":" + (index + 1)).get(key))) {
        return false;
      }
    }
    append(file, Json.write(line));
    return true;
  }

  /**
   * jsonl に最後の行が無いか、最後の行とキーの値が違えば、1 行足す。行をキーの順に足すファイル（反復ごとの報告）に使う。落ちた後にやり直しても、
   * 同じ行を二度書かない。読むのは最後の行だけ。
   *
   * @return 足したら true
   */
  public static boolean appendAfterLast(Path file, String key, Node value, Node.Table line) throws IOException {
    Optional<String> last = lastLine(file);
    if (last.isPresent() && value.equals(Json.read(last.get(), file.toString()).get(key))) {
      return false;
    }
    append(file, Json.write(line));
    return true;
  }

  /** 改行で終わる行のうち、最後の行。無ければ空。 */
  static Optional<String> lastLine(Path file) throws IOException {
    if (!Files.exists(file)) {
      return Optional.empty();
    }
    try (FileChannel channel = FileChannel.open(file, StandardOpenOption.READ)) {
      long end = lineStart(channel, channel.size());
      if (end == 0) {
        return Optional.empty();
      }
      long start = lineStart(channel, end - 1);
      ByteBuffer buffer = ByteBuffer.allocate((int) (end - 1 - start));
      while (buffer.hasRemaining()) {
        if (channel.read(buffer, start + buffer.position()) < 0) {
          throw new IOException("The file shrank while it was read: " + file);
        }
      }
      return Optional.of(new String(buffer.array(), StandardCharsets.UTF_8));
    }
  }

  /** 改行で終わる行（空の行を除く）。最後の改行より後ろは書きかけとして読まない。ファイルが無ければ空。 */
  public static List<String> lines(Path file) throws IOException {
    if (!Files.exists(file)) {
      return List.of();
    }
    String text = Files.readString(file, StandardCharsets.UTF_8);
    int end = text.lastIndexOf('\n');
    List<String> lines = new ArrayList<>();
    if (end < 0) {
      return lines;
    }
    for (String line : text.substring(0, end).split("\n", -1)) {
      if (!line.isBlank()) {
        lines.add(line);
      }
    }
    return lines;
  }

  /** ディレクトリを作る。新しく作ったディレクトリごとに、その親のディレクトリを fsync する。 */
  public static void createDirectories(Path directory) throws IOException {
    Path absolute = directory.toAbsolutePath().normalize();
    if (Files.isDirectory(absolute)) {
      return;
    }
    Path existing = absolute;
    while (existing != null && !Files.exists(existing)) {
      existing = existing.getParent();
    }
    Files.createDirectories(absolute);
    for (Path created = absolute; created != null && !created.equals(existing); created = created.getParent()) {
      syncDirectory(created.getParent());
    }
  }

  /** ファイルかディレクトリを中身ごと消す。シンボリックリンクはたどらず、リンクを消す。無ければ何もしない。 */
  public static void delete(Path path) throws IOException {
    if (!Files.exists(path, LinkOption.NOFOLLOW_LINKS)) {
      return;
    }
    try (Stream<Path> walk = Files.walk(path)) {
      for (Path entry : walk.sorted(Comparator.reverseOrder()).toList()) {
        Files.delete(entry);
      }
    }
    syncDirectory(path.toAbsolutePath().normalize().getParent());
  }

  /** ファイルかディレクトリの中の全てのファイルとディレクトリを fsync する。 */
  public static void syncTree(Path path) throws IOException {
    if (!Files.isDirectory(path, LinkOption.NOFOLLOW_LINKS)) {
      syncFile(path);
      return;
    }
    try (Stream<Path> walk = Files.walk(path)) {
      for (Path entry : walk.toList()) {
        if (Files.isDirectory(entry, LinkOption.NOFOLLOW_LINKS)) {
          syncDirectory(entry);
        } else if (Files.isRegularFile(entry, LinkOption.NOFOLLOW_LINKS)) {
          syncFile(entry);
        }
      }
    }
  }

  /** ディレクトリを fsync し、その中での作成・rename・削除をディスクに書き出す。POSIX でないファイルシステムでは何もしない。 */
  public static void syncDirectory(Path directory) throws IOException {
    if (!POSIX) {
      return;
    }
    try (FileChannel channel = FileChannel.open(directory, StandardOpenOption.READ)) {
      channel.force(true);
    }
  }

  private static void syncFile(Path file) throws IOException {
    try (FileChannel channel = FileChannel.open(file, StandardOpenOption.WRITE)) {
      channel.force(true);
    }
  }

  /**
   * 指定した位置より前にある最後の改行を探し、その次の位置（行の始め）を返す。改行が無ければ 0。ファイルの長さを渡すと、改行で
   * 終わっていない末尾を除いた長さになる。
   *
   * @param before この位置より前を探す
   */
  private static long lineStart(FileChannel channel, long before) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(4096);
    long end = before;
    while (end > 0) {
      int length = (int) Math.min(buffer.capacity(), end);
      long start = end - length;
      buffer.clear().limit(length);
      while (buffer.hasRemaining()) {
        if (channel.read(buffer, start + buffer.position()) < 0) {
          throw new IOException("The file shrank while it was read");
        }
      }
      for (int index = length - 1; index >= 0; index--) {
        if (buffer.get(index) == '\n') {
          return start + index + 1;
        }
      }
      end = start;
    }
    return 0;
  }
}
