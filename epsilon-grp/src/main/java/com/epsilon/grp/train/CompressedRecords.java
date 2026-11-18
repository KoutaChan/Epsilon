package com.epsilon.grp.train;

import com.epsilon.base.io.AtomicFiles;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Path;
import java.util.zip.GZIPOutputStream;

/** 半荘と局末の記録を gzip で圧縮して保存する。 */
final class CompressedRecords {
  static final int BUFFER = 1 << 16;

  private CompressedRecords() {}

  @FunctionalInterface
  interface Encoder {
    void write(DataOutputStream out) throws IOException;
  }

  static void write(Path file, Encoder encoder) throws IOException {
    AtomicFiles.replace(
        file,
        stream -> {
          try (var out =
              new DataOutputStream(
                  new BufferedOutputStream(new GZIPOutputStream(stream, BUFFER), BUFFER))) {
            encoder.write(out);
          }
        });
  }
}
