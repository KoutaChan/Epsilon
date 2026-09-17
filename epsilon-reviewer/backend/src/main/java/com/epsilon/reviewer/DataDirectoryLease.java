package com.epsilon.reviewer;

import java.io.IOException;
import java.nio.channels.*;
import java.nio.file.*;

/** 同じ保存先を二つのサーバーが整理・更新することを防ぎます。 */
final class DataDirectoryLease implements AutoCloseable {
  private final FileChannel channel;
  private final FileLock lock;

  DataDirectoryLease(Path directory) throws IOException {
    Files.createDirectories(directory);
    channel =
        FileChannel.open(
            directory.resolve(".server.lock"), StandardOpenOption.CREATE, StandardOpenOption.WRITE);
    try {
      lock = channel.tryLock();
      if (lock == null)
        throw new IOException("Another reviewer server is using this data directory.");
    } catch (IOException | OverlappingFileLockException e) {
      channel.close();
      throw new IOException("Another reviewer server is using this data directory.", e);
    }
  }

  @Override
  public void close() throws IOException {
    try {
      lock.release();
    } finally {
      channel.close();
    }
  }
}
