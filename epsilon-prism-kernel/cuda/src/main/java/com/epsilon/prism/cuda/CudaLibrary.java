package com.epsilon.prism.cuda;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.regex.Pattern;

/** 収集カーネルの共有ライブラリを読み込み、Java 側の宣言と照合する。 */
final class CudaLibrary {
  private CudaLibrary() {}

  private static final boolean WINDOWS = System.getProperty("os.name").startsWith("Windows");
  private static final Pattern WINDOWS_DLL =
      Pattern.compile("prism-cuda-([0-9]{1,18})-[0-9]+\\.dll");
  private static boolean loaded;

  static synchronized void ensureLoaded() {
    if (loaded) return;
    String suffix = Boolean.getBoolean("epsilon.prism.cuda.checked") ? "_check" : "";
    String resource =
        WINDOWS
            ? "/native/windows-x86_64/prism_cuda" + suffix + ".dll"
            : "/native/linux-x86_64/libprism_cuda" + suffix + ".so";
    try (InputStream stream = CudaLibrary.class.getResourceAsStream(resource)) {
      if (stream == null)
        throw new IllegalStateException("Missing CUDA module resource " + resource);
      Path library = WINDOWS ? windowsLibrary() : Files.createTempFile("libprism_cuda-", ".so");
      boolean mapped = false;
      try {
        Files.copy(stream, library, StandardCopyOption.REPLACE_EXISTING);
        System.load(library.toString());
        mapped = true;
        CudaDeclarations.requireCompatible();
        loaded = true;
      } finally {
        if (!WINDOWS || !mapped) Files.deleteIfExists(library);
      }
    } catch (IOException error) {
      throw new UncheckedIOException(error);
    }
  }

  /** Windows は使用中の DLL を削除できないので、終了した JVM の DLL を次のロード時に片付ける。 */
  private static Path windowsLibrary() throws IOException {
    Path directory =
        Files.createDirectories(Path.of(System.getProperty("java.io.tmpdir"), "prism-cuda"));
    try (var files = Files.newDirectoryStream(directory, "prism-cuda-*.dll")) {
      for (Path file : files) {
        var match = WINDOWS_DLL.matcher(file.getFileName().toString());
        if (match.matches() && ProcessHandle.of(Long.parseLong(match.group(1))).isEmpty()) {
          try {
            Files.deleteIfExists(file);
          } catch (java.nio.file.AccessDeniedException ignored) {
            // 別のプロセスがまだファイルを開いている場合は次のロード時に削除する。
          }
        }
      }
    }
    return Files.createTempFile(
        directory, "prism-cuda-" + ProcessHandle.current().pid() + "-", ".dll");
  }
}
