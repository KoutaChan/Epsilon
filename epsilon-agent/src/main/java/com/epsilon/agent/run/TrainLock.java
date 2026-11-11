package com.epsilon.agent.run;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.channels.OverlappingFileLockException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 学習のロック（run の .train.lock）。段とジョブを回す JVM が、動いている間ずっと 1 つのチャンネルで持つ。Φ の切り替えやジョブの
 * コマンドは、このロックで学習が動いているかを見る。
 *
 * <p>POSIX のロックはプロセスごとなので、同じプロセスが同じファイルを開き直して閉じると外れてしまう。そのため、この JVM が取ったロックは
 * このクラスが一覧で覚えておき、JVM の中ではロックを試さずにその一覧を見る。ほかのプロセスはロックを一瞬だけ取って確かめるので、ロックを
 * 取る側はしばらく待つ。
 */
public final class TrainLock implements AutoCloseable {

  /** ロックを取るまで待つ時間の上限。ほかのプロセスは確かめた後すぐ外すので、これより長く取れなければ別の JVM が持っている。 */
  static final Duration WAIT = Duration.ofSeconds(30);

  private static final Duration POLL = Duration.ofMillis(100);

  private static final Set<Path> HELD = ConcurrentHashMap.newKeySet();

  private final Path file;
  private final FileChannel channel;

  private TrainLock(Path file, FileChannel channel) {
    this.file = file;
    this.channel = channel;
  }

  /**
   * ロックを取る。取れなければ最大 {@link #WAIT} 待つ。
   *
   * @throws IOException 待っても取れない（ほかの JVM が同じ run を回している）か、この JVM が既に持っている
   */
  public static TrainLock hold(Path file) throws IOException {
    return hold(file, WAIT);
  }

  /** @param wait 取れないときに待つ時間の上限 */
  static TrainLock hold(Path file, Duration wait) throws IOException {
    Path path = file.toAbsolutePath().normalize();
    if (!HELD.add(path)) {
      throw new IOException("This JVM already holds " + path);
    }
    try {
      DurableFiles.createDirectories(path.getParent());
      FileChannel channel = FileChannel.open(path, StandardOpenOption.CREATE, StandardOpenOption.WRITE);
      try {
        long deadline = System.nanoTime() + wait.toNanos();
        while (true) {
          FileLock lock;
          try {
            lock = channel.tryLock();
          } catch (OverlappingFileLockException probing) {
            lock = null;
          }
          if (lock != null) {
            return new TrainLock(path, channel);
          }
          if (System.nanoTime() >= deadline) {
            throw new IOException("Another JVM holds " + path);
          }
          Thread.sleep(POLL.toMillis());
        }
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        channel.close();
        throw new IOException("Interrupted while waiting for " + path, interrupted);
      } catch (IOException | RuntimeException failure) {
        channel.close();
        throw failure;
      }
    } catch (IOException | RuntimeException failure) {
      HELD.remove(path);
      throw failure;
    }
  }

  /**
   * 段とジョブを回す JVM が、このロックを持っているか。この JVM が持っていれば、ロックを試さずに true を返す。
   */
  public static boolean running(Path file) throws IOException {
    Path path = file.toAbsolutePath().normalize();
    if (HELD.contains(path)) {
      return true;
    }
    if (!Files.exists(path)) {
      return false;
    }
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.WRITE)) {
      FileLock lock = channel.tryLock();
      if (lock == null) {
        return true;
      }
      lock.release();
      return false;
    } catch (OverlappingFileLockException probing) {
      return true;
    }
  }

  @Override
  public void close() throws IOException {
    try {
      channel.close();
    } finally {
      HELD.remove(file);
    }
  }
}
