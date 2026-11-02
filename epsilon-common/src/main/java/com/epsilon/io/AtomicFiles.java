package com.epsilon.io;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;

/** 同じディレクトリで書き終えたファイルだけを公開する。 */
public final class AtomicFiles {
  private AtomicFiles() {}

  @FunctionalInterface
  public interface Writer {
    /** 呼び出し側がstreamを所有するため、ここではcloseしない。 */
    void write(OutputStream stream) throws IOException;
  }

  /** 再開点など、原子的な公開を保証できないストレージでは失敗させる。 */
  public static void write(Path target, byte[] bytes) throws IOException {
    write(target, stream -> stream.write(bytes));
  }

  public static void write(Path target, Writer writer) throws IOException {
    replace(target, writer, true);
  }

  /** 従来のNanoレポート保存と同様に、非対応FSでは通常の置換を許す。 */
  public static void replace(Path target, byte[] bytes) throws IOException {
    replace(target, stream -> stream.write(bytes), false);
  }

  private static void replace(Path destination, Writer writer, boolean requireAtomic)
      throws IOException {
    Path target = destination.toAbsolutePath().normalize();
    Files.createDirectories(target.getParent());
    Path temporary = Files.createTempFile(target.getParent(), ".atomic-", ".tmp");
    try {
      try (FileChannel channel = FileChannel.open(temporary, StandardOpenOption.WRITE)) {
        writer.write(Channels.newOutputStream(channel));
        channel.force(true);
      }
      try {
        Files.move(
            temporary, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
      } catch (AtomicMoveNotSupportedException unsupported) {
        if (requireAtomic) throw unsupported;
        Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
      }
    } finally {
      Files.deleteIfExists(temporary);
    }
  }
}
