package com.epsilon.reviewer;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;

/** 試験の一時のフォルダーと資源。 */
public final class TestFiles {
  private TestFiles() {}

  public static Path directory(String prefix) {
    try {
      return Files.createTempDirectory(prefix);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  /** フォルダーを中身ごと消す（無ければ何もしない）。 */
  public static void delete(Path directory) {
    if (!Files.exists(directory)) return;
    try (var paths = Files.walk(directory)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  /** epsilon-common の試験の天鳳の牌譜（pom の testResources、例 kakan-a）。 */
  public static byte[] golden(String name) {
    String file = "/com/epsilon/engine/golden/" + name + ".xml.gz";
    try (InputStream input = TestFiles.class.getResourceAsStream(file)) {
      if (input == null) throw new IllegalStateException("Missing golden record: " + file);
      return input.readAllBytes();
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }
}
