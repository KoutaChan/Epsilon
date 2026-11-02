package com.epsilon.prism.grp;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/**
 * 裏の作業が動く jar の写し: 本番の jar は反復の境目で上書きされるので、起動のたびに grp/jars/&lt;sha の先頭 16 文字&gt;.jar へ写し（既にあれば
 * 写さない）、その写しから動かす（長い学習の途中の NoClassDefFoundError や古い・新しいクラスの混ざりを防ぐ）。sha は refits.jsonl と
 * phimon.jsonl に書く。走らせ役は {@code GrpWorkerMain --run DIR --copy-jar} で写しのパスを受け取り、そのパスで作業を起動する。
 */
public final class WorkerJar {

  /** 写しの名前に使う sha の文字数。 */
  static final int NAME_LENGTH = 16;

  private WorkerJar() {}

  /** 今動いているクラスの入った jar の sha256（jar でなく展開したクラスのディレクトリから動いていれば空）。 */
  public static String runningSha256() throws IOException {
    Path running = running();
    return Files.isRegularFile(running) ? sha256(running) : "";
  }

  /** 今動いているクラスの入った jar（試験ではクラスのディレクトリ）。 */
  static Path running() throws IOException {
    try {
      return Path.of(WorkerJar.class.getProtectionDomain().getCodeSource().getLocation().toURI());
    } catch (URISyntaxException | RuntimeException unknown) {
      throw new IOException("The running jar is unknown", unknown);
    }
  }

  /**
   * jar を grp/jars へ写す（同じ sha があれば写さない）。
   *
   * @param jar 本番の jar
   * @return 写しのパス
   * @throws IOException 読めないか書けないとき
   */
  public static Path copy(GrpFiles files, Path jar) throws IOException {
    if (!Files.isRegularFile(jar)) throw new IOException("Not a jar file: " + jar);
    String sha = sha256(jar);
    Path target = files.jars().resolve(sha.substring(0, NAME_LENGTH) + ".jar");
    // 写しは中身を確かめてから rename で名前を付けるので、名前があれば中身は sha のとおり
    if (Files.isRegularFile(target)) return target;
    Files.createDirectories(files.jars());
    // 写す途中で本番の jar が差し替わっても、写しの中身を確かめてから名前を付ける
    Path temporary = Files.createTempFile(files.jars(), "copy-", ".tmp");
    try {
      Files.copy(jar, temporary, StandardCopyOption.REPLACE_EXISTING);
      try (FileChannel channel = FileChannel.open(temporary, StandardOpenOption.WRITE)) {
        channel.force(true);
      }
      String copied = sha256(temporary);
      if (!copied.equals(sha)) throw new IOException("The jar changed while it was copied: " + jar);
      Files.move(temporary, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    } finally {
      Files.deleteIfExists(temporary);
    }
    return target;
  }

  /** ファイルの sha256（16 進の小文字）。 */
  static String sha256(Path file) throws IOException {
    MessageDigest digest;
    try {
      digest = MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException("SHA-256 is unavailable", impossible);
    }
    byte[] buffer = new byte[1 << 16];
    try (InputStream in = Files.newInputStream(file)) {
      for (int read; (read = in.read(buffer)) > 0; ) digest.update(buffer, 0, read);
    }
    return HexFormat.of().formatHex(digest.digest());
  }
}
