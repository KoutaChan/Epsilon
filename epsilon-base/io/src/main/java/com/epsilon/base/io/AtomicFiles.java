package com.epsilon.base.io;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.channels.FileChannel;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;

/** 完成したファイルを同じディレクトリ内で原子的に公開する。 */
public final class AtomicFiles {
  private static final boolean POSIX =
      FileSystems.getDefault().supportedFileAttributeViews().contains("posix");

  private AtomicFiles() {}

  @FunctionalInterface
  public interface Writer {
    void write(OutputStream stream) throws IOException;
  }

  public static void replace(Path target, Writer writer) throws IOException {
    Path destination = target.toAbsolutePath().normalize();
    Path directory = destination.getParent();
    createDirectories(directory);
    Path temporary = Files.createTempFile(directory, "." + destination.getFileName() + ".", ".tmp");
    try {
      try (OutputStream stream = Files.newOutputStream(temporary)) {
        writer.write(stream);
      }
      syncFile(temporary);
      Files.move(
          temporary,
          destination,
          StandardCopyOption.ATOMIC_MOVE,
          StandardCopyOption.REPLACE_EXISTING);
      syncDirectory(directory);
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  /** ディレクトリを作る。新しく作ったディレクトリごとに、その親のディレクトリを fsync する。 */
  public static void createDirectories(Path directory) throws IOException {
    Path absolute = directory.toAbsolutePath().normalize();
    if (Files.isDirectory(absolute)) {
      return;
    }
    Path existing = absolute;
    while (existing != null && !Files.exists(existing)) {
      existing = existing.getParent();
    }
    Files.createDirectories(absolute);
    for (Path created = absolute;
        created != null && !created.equals(existing);
        created = created.getParent()) {
      syncDirectory(created.getParent());
    }
  }

  /** ディレクトリを fsync し、その中での作成・rename・削除をディスクに書き出す。POSIX でないファイルシステムでは何もしない。 */
  public static void syncDirectory(Path directory) throws IOException {
    if (!POSIX) {
      return;
    }
    try (FileChannel channel = FileChannel.open(directory, StandardOpenOption.READ)) {
      channel.force(true);
    }
  }

  public static void syncFile(Path file) throws IOException {
    try (FileChannel channel = FileChannel.open(file, StandardOpenOption.WRITE)) {
      channel.force(true);
    }
  }
}
