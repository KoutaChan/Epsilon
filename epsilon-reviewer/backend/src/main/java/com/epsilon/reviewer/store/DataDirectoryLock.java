package com.epsilon.reviewer.store;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/** 同じ保存先を 2 つのレビュアーが更新することを防ぐ OS の鍵。 */
public final class DataDirectoryLock implements AutoCloseable {
  private final FileChannel channel;
  private final FileLock lock;

  private DataDirectoryLock(FileChannel channel, FileLock lock) {
    this.channel = channel;
    this.lock = lock;
  }

  /** @throws ReviewerException data_locked（他のレビュアーが鍵を持つ） */
  public static DataDirectoryLock acquire(Path directory) throws IOException {
    FileChannel channel =
        FileChannel.open(
            directory.resolve(StoreLayout.LOCK.fileName()),
            StandardOpenOption.CREATE,
            StandardOpenOption.WRITE);
    try {
      FileLock lock = channel.tryLock();
      if (lock == null) throw new ReviewerException(ErrorCode.DATA_LOCKED);
      return new DataDirectoryLock(channel, lock);
    } catch (OverlappingFileLockException failure) {
      channel.close();
      throw new ReviewerException(ErrorCode.DATA_LOCKED);
    } catch (IOException | RuntimeException failure) {
      channel.close();
      throw failure;
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
