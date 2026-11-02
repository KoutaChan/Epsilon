package com.epsilon.prism.grp;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.Optional;

/**
 * 裏の作業の錠（{@code FileChannel.lock}。プロセスが死ぬとカーネルが外す）。採点と学習は別の錠なので、長い学習の最中も採点は進む。
 */
public final class WorkerLock implements AutoCloseable {

  /** 錠の種類（ファイルは {@link GrpFiles#lock} の名前）。 */
  public enum Kind {
    /** 採点（{@link PhiScorer}）。 */
    SCORE,

    /** 学習と自動の切り替え・巻き戻しの点検。 */
    REFIT;

    /** 錠のファイルの名前（score・refit）。 */
    String fileName() {
      return name().toLowerCase(Locale.ROOT);
    }
  }

  private final FileChannel channel;
  private final FileLock lock;

  private WorkerLock(FileChannel channel, FileLock lock) {
    this.channel = channel;
    this.lock = lock;
  }

  /**
   * 錠を取る（待たない）。
   *
   * @return 取れた錠（ほかの作業が持っていれば空）
   * @throws IOException 錠のファイルが開けないとき
   */
  public static Optional<WorkerLock> tryAcquire(GrpFiles files, Kind kind) throws IOException {
    Files.createDirectories(files.directory());
    FileChannel channel =
        FileChannel.open(files.lock(kind.fileName()), StandardOpenOption.CREATE, StandardOpenOption.WRITE);
    try {
      FileLock lock = channel.tryLock();
      if (lock != null) return Optional.of(new WorkerLock(channel, lock));
    } catch (OverlappingFileLockException heldHere) {
      // 同じ JVM が既に持っている（ほかの作業と同じ扱い）
    } catch (IOException | RuntimeException failure) {
      channel.close();
      throw failure;
    }
    channel.close();
    return Optional.empty();
  }

  /** 錠を外す。 */
  @Override
  public void close() throws IOException {
    try (channel) {
      lock.release();
    }
  }
}
