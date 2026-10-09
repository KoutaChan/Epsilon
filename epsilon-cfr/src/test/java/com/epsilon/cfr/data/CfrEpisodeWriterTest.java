package com.epsilon.cfr.data;

import static org.testng.Assert.*;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.testng.annotations.Test;

public final class CfrEpisodeWriterTest {
  @Test(timeOut = 10000)
  public void closePublishesAllMatchFrames() throws Exception {
    var directory = Files.createTempDirectory("cfr-episode-writer");
    try {
      try (var writer = new CfrEpisodeWriter(directory)) {
        for (int index = 0; index < 3; index++) {
          for (int i = 0; i < 32; i++)
            writer.accept(new CfrEpisodeFrame(episode(i).episode(), index, index == 2));
        }
      }
      for (int i = 0; i < 32; i++) {
        var path = directory.resolve(String.format(java.util.Locale.ROOT, "%08d.bin", i));
        assertEquals(CfrMatchFile.summary(path), new CfrMatchFile.Summary(i, 3, 0));
        assertEquals(CfrMatchFile.read(path).size(), 3);
        assertEquals(CfrMatchFile.read(path).getFirst().episodeId(), i);
        Files.delete(path);
      }
      try (var files = Files.list(directory)) {
        assertEquals(files.count(), 0L);
      }
    } finally {
      try (var files = Files.list(directory)) {
        for (var file : files.toList()) Files.deleteIfExists(file);
      }
      Files.deleteIfExists(directory);
    }
  }

  @Test(timeOut = 10000)
  public void boundsPendingWritesAndDrainsOnClose() throws Exception {
    var writing = new CountDownLatch(1);
    var release = new CountDownLatch(1);
    var saved = new AtomicInteger();
    var writer =
        new CfrEpisodeWriter(
            1,
            1,
            item -> {
              writing.countDown();
              await(release);
              saved.incrementAndGet();
            });
    try {
      writer.accept(episode(0));
      assertTrue(writing.await(5, TimeUnit.SECONDS));
      var enqueuing = new CountDownLatch(1);
      var second =
          CompletableFuture.runAsync(
              () -> {
                enqueuing.countDown();
                writer.accept(episode(1));
              });
      assertTrue(enqueuing.await(5, TimeUnit.SECONDS));
      assertFalse(second.isDone());
      release.countDown();
      second.get(5, TimeUnit.SECONDS);
      writer.close();
      assertEquals(saved.get(), 2);
      expectThrows(IllegalStateException.class, () -> writer.accept(episode(2)));
    } finally {
      release.countDown();
      writer.close();
    }
  }

  @Test(timeOut = 10000)
  public void failedWriteUnblocksProducerAndFailsDrain() throws Exception {
    var started = new CountDownLatch(1);
    var fail = new CountDownLatch(1);
    var error = new IOException("simulated storage failure");
    var writer =
        new CfrEpisodeWriter(
            1,
            1,
            item -> {
              started.countDown();
              await(fail);
              throw error;
            });
    writer.accept(episode(0));
    assertTrue(started.await(5, TimeUnit.SECONDS));
    var second =
        CompletableFuture.supplyAsync(
            () -> expectThrows(UncheckedIOException.class, () -> writer.accept(episode(1))));
    fail.countDown();
    assertSame(second.get(5, TimeUnit.SECONDS).getCause().getCause(), error);
    assertSame(expectThrows(IOException.class, writer::close).getCause(), error);
  }

  @Test(timeOut = 10000)
  public void nonBlockingAdmissionRejectsWithoutRetainingAndNotifiesWhenCapacityReturns()
      throws Exception {
    var writing = new CountDownLatch(1);
    var release = new CountDownLatch(1);
    var saved = new CopyOnWriteArrayList<Long>();
    var writer =
        new CfrEpisodeWriter(
            1,
            1,
            item -> {
              writing.countDown();
              await(release);
              saved.add(item.episode().episodeId());
            });
    try {
      assertTrue(writer.whenAvailable().toCompletableFuture().isDone());
      assertTrue(writer.tryAccept(episode(0)));
      assertTrue(writing.await(5, TimeUnit.SECONDS));
      assertFalse(writer.tryAccept(episode(1)));
      var available = writer.whenAvailable().toCompletableFuture();
      assertFalse(available.isDone());
      release.countDown();
      available.get(5, TimeUnit.SECONDS);
      assertTrue(writer.whenAvailable().toCompletableFuture().isDone());
      assertTrue(writer.tryAccept(episode(1)));
      writer.close();
      assertEquals(saved, List.of(0L, 1L));
      expectThrows(IllegalStateException.class, () -> writer.tryAccept(episode(2)));
    } finally {
      release.countDown();
      writer.close();
    }
  }

  @Test(timeOut = 10000)
  public void nonBlockingProducerReceivesWriteFailure() throws Exception {
    var started = new CountDownLatch(1);
    var fail = new CountDownLatch(1);
    var error = new IOException("simulated storage failure");
    var writer =
        new CfrEpisodeWriter(
            1,
            1,
            item -> {
              started.countDown();
              await(fail);
              throw error;
            });
    try {
      assertTrue(writer.tryAccept(episode(0)));
      assertTrue(started.await(5, TimeUnit.SECONDS));
      var available = writer.whenAvailable().toCompletableFuture();
      fail.countDown();
      var signaled =
          expectThrows(ExecutionException.class, () -> available.get(5, TimeUnit.SECONDS));
      assertSame(signaled.getCause().getCause().getCause(), error);
      assertSame(
          expectThrows(UncheckedIOException.class, () -> writer.tryAccept(episode(1)))
              .getCause()
              .getCause(),
          error);
    } finally {
      fail.countDown();
      assertSame(expectThrows(IOException.class, writer::close).getCause(), error);
    }
  }

  @Test(timeOut = 10000)
  public void closeWakesAdmissionWaitersBeforeDrainingAcceptedWrites() throws Exception {
    var writing = new CountDownLatch(1);
    var release = new CountDownLatch(1);
    var saved = new AtomicInteger();
    var writer =
        new CfrEpisodeWriter(
            1,
            1,
            item -> {
              writing.countDown();
              await(release);
              saved.incrementAndGet();
            });
    try {
      assertTrue(writer.tryAccept(episode(0)));
      assertTrue(writing.await(5, TimeUnit.SECONDS));
      var available = writer.whenAvailable().toCompletableFuture();
      var closing =
          CompletableFuture.runAsync(
              () -> {
                try {
                  writer.close();
                } catch (IOException failure) {
                  throw new UncheckedIOException(failure);
                }
              });
      var signaled =
          expectThrows(ExecutionException.class, () -> available.get(5, TimeUnit.SECONDS));
      assertTrue(signaled.getCause() instanceof IllegalStateException);
      assertFalse(closing.isDone(), "Accepted writes must finish before close returns");
      release.countDown();
      closing.get(5, TimeUnit.SECONDS);
      assertEquals(saved.get(), 1);
    } finally {
      release.countDown();
      writer.close();
    }
  }

  private static CfrEpisodeFrame episode(long id) {
    return new CfrEpisodeFrame(new CfrEpisode(id, List.of(), new double[] {1, -1, 0, 0}), 0, true);
  }

  @Test(timeOut = 10000)
  public void keepsOneMatchInOrderWhileOtherWorkersContinue() throws Exception {
    var first = new CountDownLatch(1);
    var release = new CountDownLatch(1);
    var other = new CountDownLatch(1);
    var indices = new CopyOnWriteArrayList<Integer>();
    var writer =
        new CfrEpisodeWriter(
            4,
            16,
            frame -> {
              if (frame.episode().episodeId() == 0) {
                indices.add(frame.index());
                if (frame.index() == 0) {
                  first.countDown();
                  await(release);
                }
              } else other.countDown();
            });
    try {
      writer.accept(new CfrEpisodeFrame(episode(0).episode(), 0, false));
      assertTrue(first.await(5, TimeUnit.SECONDS));
      writer.accept(new CfrEpisodeFrame(episode(0).episode(), 1, false));
      writer.accept(new CfrEpisodeFrame(episode(0).episode(), 2, true));
      // game 0と固定laneが衝突するgame 4も、空いているワーカーが保存できることを確かめる。
      writer.accept(episode(4));
      assertTrue(other.await(5, TimeUnit.SECONDS), "An unrelated match must not wait for game 0");
      assertEquals(indices, List.of(0));
      release.countDown();
    } finally {
      release.countDown();
      writer.close();
    }
    assertEquals(indices, List.of(0, 1, 2));
  }

  @Test(timeOut = 10000)
  public void closeDrainsQueuedFramesWhoseMatchIsStillBeingWritten() throws Exception {
    var started = new CountDownLatch(1);
    var release = new CountDownLatch(1);
    var indices = new CopyOnWriteArrayList<Integer>();
    var writer =
        new CfrEpisodeWriter(
            2,
            2,
            frame -> {
              indices.add(frame.index());
              if (frame.index() == 0) {
                started.countDown();
                await(release);
              }
            });
    try {
      writer.accept(new CfrEpisodeFrame(episode(0).episode(), 0, false));
      assertTrue(started.await(5, TimeUnit.SECONDS));
      writer.accept(new CfrEpisodeFrame(episode(0).episode(), 1, true));
      var available = writer.whenAvailable().toCompletableFuture();
      var closing =
          CompletableFuture.runAsync(
              () -> {
                try {
                  writer.close();
                } catch (IOException error) {
                  throw new UncheckedIOException(error);
                }
              });
      expectThrows(ExecutionException.class, () -> available.get(5, TimeUnit.SECONDS));
      assertFalse(closing.isDone());
      assertEquals(indices, List.of(0));
      release.countDown();
      closing.get(5, TimeUnit.SECONDS);
      assertEquals(indices, List.of(0, 1));
    } finally {
      release.countDown();
      writer.close();
    }
  }

  @Test(timeOut = 10000)
  public void unfinishedMatchIsNotPublishedAndCanRestartFromFirstFrame() throws Exception {
    var directory = Files.createTempDirectory("cfr-unfinished-match");
    try {
      var writer = new CfrEpisodeWriter(directory);
      writer.accept(new CfrEpisodeFrame(episode(7).episode(), 0, false));
      expectThrows(IOException.class, writer::close);
      var target = directory.resolve("00000007.bin");
      assertFalse(Files.exists(target));
      assertTrue(Files.exists(directory.resolve("00000007.bin.part")));
      try (var retry = new CfrEpisodeWriter(directory)) {
        retry.accept(episode(7));
      }
      assertEquals(CfrMatchFile.summary(target), new CfrMatchFile.Summary(7, 1, 0));
      assertEquals(CfrMatchFile.read(target).size(), 1);
    } finally {
      try (var files = Files.list(directory)) {
        for (var file : files.toList()) Files.deleteIfExists(file);
      }
      Files.delete(directory);
    }
  }

  private static void await(CountDownLatch latch) throws IOException {
    try {
      if (!latch.await(5, TimeUnit.SECONDS)) throw new IOException("Timed out waiting for test");
    } catch (InterruptedException error) {
      Thread.currentThread().interrupt();
      throw new IOException(error);
    }
  }
}
