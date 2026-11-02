package com.epsilon.prism.grp;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/**
 * 裏の作業が動く jar: 本番の jar は反復の境目で上書きされるので、走らせ役が grp/jars/&lt;sha256 の先頭 16 文字&gt;.jar へ写し、その写しから
 * 起動する。sha は refits.jsonl と phimon.jsonl に書く。
 */
public final class WorkerJar {

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
