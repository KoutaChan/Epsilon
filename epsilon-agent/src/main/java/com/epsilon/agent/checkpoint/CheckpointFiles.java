package com.epsilon.agent.checkpoint;

import ai.djl.MalformedModelException;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import com.epsilon.agent.run.DurableFiles;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.stream.Stream;

/**
 * ネットワークのチェックポイントを、1 つのディレクトリとして書いて読む。weights.bin は、DJL の {@code Block.saveParameters} が
 * 書く形式のまま。model.json には、モデルの系列ごとに決めたヘッダーの record を、整形した UTF-8 の JSON で書く。
 *
 * <p>書くときは、全てのファイルを &lt;dir&gt;.tmp に書いて fsync してから、&lt;dir&gt; へ rename する。そのため &lt;dir&gt; があれば、
 * 中身は全てそろっている。書くのは新しい &lt;dir&gt; だけで、既にある &lt;dir&gt; には触れない。書き込みに失敗したら &lt;dir&gt;.tmp を消す。
 */
public final class CheckpointFiles {

  private static final String WEIGHTS = "weights.bin";
  private static final String HEADER = "model.json";
  private static final String TEMPORARY = ".tmp";
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  private CheckpointFiles() {}

  /**
   * 初期化したブロックの重みとヘッダーを、新しいディレクトリに書く。ディレクトリが既にあれば FileAlreadyExistsException を、
   * ブロックが初期化されていなければ IllegalArgumentException を投げる。どちらのときも何も書かない。
   * 前に途中で止まった書き込みが &lt;dir&gt;.tmp を残していれば、それを消してから書く。
   */
  public static <H extends Record> void write(Path directory, Block block, H header) throws IOException {
    Objects.requireNonNull(header, "header");
    if (!block.isInitialized()) {
      throw new IllegalArgumentException("Block parameters are not initialized for " + directory);
    }
    if (Files.exists(directory)) {
      throw new FileAlreadyExistsException(directory.toString(), null, "Checkpoint directory already exists");
    }
    byte[] json = GSON.toJson(header).getBytes(StandardCharsets.UTF_8);
    Path temporary = directory.resolveSibling(directory.getFileName() + TEMPORARY);
    if (Files.exists(temporary)) {
      DurableFiles.delete(temporary);
    }
    try {
      Files.createDirectories(temporary);
      writeSynced(temporary.resolve(WEIGHTS), block::saveParameters);
      writeSynced(temporary.resolve(HEADER), out -> out.write(json));
      Files.move(temporary, directory, StandardCopyOption.ATOMIC_MOVE);
    } catch (IOException | RuntimeException failure) {
      try {
        if (Files.exists(temporary)) {
          DurableFiles.delete(temporary);
        }
      } catch (IOException cleanup) {
        failure.addSuppressed(cleanup);
      }
      throw failure;
    }
  }

  /** model.json を、指定した型のヘッダーとして読む。読めないときや、JSON がその型に合わないときは IOException を投げる。 */
  public static <H extends Record> H readHeader(Path directory, Class<H> type) throws IOException {
    Path file = directory.resolve(HEADER);
    H header;
    try {
      header = GSON.fromJson(Files.readString(file), type);
    } catch (RuntimeException invalid) {
      throw new IOException("Invalid checkpoint header " + file, invalid);
    }
    if (header == null) {
      throw new IOException("Empty checkpoint header " + file);
    }
    return header;
  }

  /**
   * weights.bin を、ブロックのパラメーターに読み込む。ブロックは、書いたときと同じ形に組んでおく。
   * 形式が合わないときや、読み終えた後にデータが残るときは IOException を投げる。
   */
  public static void loadParameters(Path directory, Block block, NDManager manager) throws IOException {
    Path file = directory.resolve(WEIGHTS);
    try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file)))) {
      block.loadParameters(manager, in);
      if (in.read() != -1) {
        throw new IOException("Trailing weight data in " + file);
      }
    } catch (MalformedModelException invalid) {
      throw new IOException("Invalid weights " + file, invalid);
    }
  }

  /**
   * ディレクトリ全体の SHA-256 を、小文字の 16 進 64 文字で返す。中の通常ファイルをパスの順に並べ、
   * ファイルごとに次の順でハッシュに足す: '/' で区切った相対パス（UTF-8）、0x00、サイズ（8 バイトのビッグエンディアン）、中身。
   * models.json の sha256 はこの値。
   * ディレクトリでないとき、ディレクトリ自身か中にシンボリックリンクがあるとき、ファイルが 1 つも無いときは IOException を投げる。
   */
  public static String sha256(Path directory) throws IOException {
    if (Files.isSymbolicLink(directory) || !Files.isDirectory(directory)) {
      throw new IOException("Checkpoint must be a regular directory: " + directory);
    }
    List<Path> paths;
    try (Stream<Path> walk = Files.walk(directory)) {
      paths = walk.sorted().toList();
    }
    if (paths.stream().anyMatch(Files::isSymbolicLink)) {
      throw new IOException("Checkpoint holds a symbolic link: " + directory);
    }
    List<Path> files = paths.stream().filter(Files::isRegularFile).toList();
    if (files.isEmpty()) {
      throw new IOException("Checkpoint holds no files: " + directory);
    }
    MessageDigest digest = sha256();
    for (Path file : files) {
      digest.update(directory.relativize(file).toString().replace('\\', '/').getBytes(StandardCharsets.UTF_8));
      digest.update((byte) 0);
      digest.update(ByteBuffer.allocate(Long.BYTES).putLong(Files.size(file)).array());
      try (InputStream in = new DigestInputStream(Files.newInputStream(file), digest)) {
        in.transferTo(OutputStream.nullOutputStream());
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  @FunctionalInterface
  private interface Content {
    void write(DataOutputStream out) throws IOException;
  }

  /** ファイルを新しく作って中身を書き、ディスクへ fsync する。 */
  private static void writeSynced(Path file, Content content) throws IOException {
    try (FileChannel channel = FileChannel.open(file, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
        DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Channels.newOutputStream(channel)))) {
      content.write(out);
      out.flush();
      channel.force(true);
    }
  }

  private static MessageDigest sha256() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException missing) {
      throw new IllegalStateException("SHA-256 is not available", missing);
    }
  }
}
