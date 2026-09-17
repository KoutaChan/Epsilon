package com.epsilon.config.settings;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** ローカル設定がない場合だけ、同梱資源から既定ファイルを生成する。 */
public final class SettingsFiles {
  private static final Logger log = LoggerFactory.getLogger(SettingsFiles.class);

  private SettingsFiles() {}

  /** 既存ファイルを保持し、初回だけ同梱資源をそのままコピーする。 */
  public static void createDefault(Path path, String resource) throws IOException {
    if (Files.exists(path)) {
      return;
    }
    try (InputStream stream = SettingsFiles.class.getClassLoader().getResourceAsStream(resource)) {
      if (stream == null) {
        throw new IOException("Bundled settings not found: " + resource);
      }
      Files.createDirectories(path.toAbsolutePath().getParent());
      Files.copy(stream, path);
    }
    log.info("Created settings file: {}", path.toAbsolutePath());
  }
}
