package com.epsilon.agent.run;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.CodeSource;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/**
 * jar の ID を求める。jar の ID は jar のファイルの SHA-256 の先頭 8 桁（16 進の小文字）で、run の jars の中で使う名前と同じ。
 */
public final class Jars {

  /** jar ではなくクラスのディレクトリから動いている JVM（テスト）の ID。 */
  public static final String CLASSES = "classes";

  private static final int DIGITS = 8;

  private Jars() {}

  public static String identity(Path jar) throws IOException {
    MessageDigest digest = sha256();
    try (InputStream in = Files.newInputStream(jar)) {
      byte[] buffer = new byte[1 << 16];
      for (int read = in.read(buffer); read > 0; read = in.read(buffer)) {
        digest.update(buffer, 0, read);
      }
    }
    return HexFormat.of().formatHex(digest.digest()).substring(0, DIGITS);
  }

  /** この JVM が run のクラスを読み込んだ jar の ID。クラスのディレクトリから動いていれば {@value #CLASSES}。 */
  public static String running() throws IOException {
    CodeSource source = Jars.class.getProtectionDomain().getCodeSource();
    if (source == null) {
      return CLASSES;
    }
    Path location;
    try {
      location = Path.of(source.getLocation().toURI());
    } catch (URISyntaxException | IllegalArgumentException unusual) {
      throw new IOException("Cannot locate the jar of " + Jars.class.getName(), unusual);
    }
    return Files.isRegularFile(location) ? identity(location) : CLASSES;
  }

  private static MessageDigest sha256() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException missing) {
      throw new IllegalStateException("SHA-256 is missing from this JVM", missing);
    }
  }
}
