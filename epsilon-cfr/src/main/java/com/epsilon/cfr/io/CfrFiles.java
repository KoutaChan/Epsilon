package com.epsilon.cfr.io;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/** CFRチェックポイントの内容を識別し、実行ディレクトリ内の参照先を解決する。 */
public final class CfrFiles {
  private CfrFiles() {}

  public static String sha256(Path file) throws IOException {
    MessageDigest digest = digest();
    byte[] buffer = new byte[1 << 16];
    try (var input = Files.newInputStream(file)) {
      int count;
      while ((count = input.read(buffer)) != -1) digest.update(buffer, 0, count);
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  public static String sha256(byte[] bytes) {
    return HexFormat.of().formatHex(digest().digest(bytes));
  }

  public static Path inside(Path root, String relative) throws IOException {
    Path base = root.toAbsolutePath().normalize();
    Path result = base.resolve(relative).normalize();
    if (Path.of(relative).isAbsolute() || !result.startsWith(base) || result.equals(base))
      throw new IOException("Checkpoint path escapes its directory: " + relative);
    return result;
  }

  private static MessageDigest digest() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException impossible) {
      throw new AssertionError(impossible);
    }
  }
}
