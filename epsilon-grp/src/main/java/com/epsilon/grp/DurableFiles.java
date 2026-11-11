package com.epsilon.grp;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;

/** プロセスが落ちても書きかけのファイルを残さないための書き込み。ファイルの中身と、ディレクトリのエントリーを fsync する。 */
final class DurableFiles {

  private DurableFiles() {}

  /** ファイルを空にして（無ければ作って）書き、fsync する。 */
  static void write(Path file, byte[] bytes) throws IOException {
    try (FileChannel channel = FileChannel.open(
        file, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE)) {
      ByteBuffer buffer = ByteBuffer.wrap(bytes);
      while (buffer.hasRemaining()) {
        channel.write(buffer);
      }
      channel.force(true);
    }
  }

  /** 同じディレクトリの一時ファイルに書いて fsync し、元のファイルと置き換えてから親ディレクトリを fsync する。 */
  static void replace(Path file, byte[] bytes) throws IOException {
    Path target = file.toAbsolutePath();
    Path staging = target.resolveSibling("." + target.getFileName() + ".writing");
    try {
      write(staging, bytes);
      Files.move(staging, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
    } finally {
      Files.deleteIfExists(staging);
    }
    syncDirectory(target.getParent());
  }

  /** ディレクトリのエントリーを fsync する。ディレクトリを開けない OS（Windows）では OS に任せる。 */
  static void syncDirectory(Path directory) {
    try (FileChannel channel = FileChannel.open(directory, StandardOpenOption.READ)) {
      channel.force(true);
    } catch (IOException unsupported) {
      // ディレクトリを開けない OS では、rename の永続化を OS に任せる
    }
  }
}
