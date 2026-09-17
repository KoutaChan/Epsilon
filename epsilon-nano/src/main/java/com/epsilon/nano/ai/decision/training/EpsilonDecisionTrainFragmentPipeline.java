package com.epsilon.nano.ai.decision.training;

import com.epsilon.config.settings.DecisionTrainSpoolSettings;
import com.epsilon.nano.ai.decision.arena.EpsilonDecisionArena;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionCompletedGame;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionFragment;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionFragmentStore;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionTrainingSampleDescriptorReader;
import com.epsilon.util.FormatUtils;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/** 方策勾配学習に使うデータ片を選抜し、非同期の一時保存と読み込みを管理する。 */
final class EpsilonDecisionTrainFragmentPipeline {

  private EpsilonDecisionTrainFragmentPipeline() {}

  static StreamingTrainingPlan buildStreamingTrainingPlan(
      TrainCollection trainCollection, long actorSnapshotId, int maxSamples, long seed)
      throws IOException {
    EpsilonDecisionFragmentStore store = trainCollection.fragmentStore();
    EpsilonDecisionFragmentStore.ExpectedIdentity expectedIdentity =
        trainCollection.expectedIdentity();
    if (expectedIdentity.actorSnapshotId() != actorSnapshotId) {
      throw new IOException(
          "Decision streaming plan actor identity mismatch: expected="
              + expectedIdentity.actorSnapshotId()
              + " requested="
              + actorSnapshotId);
    }
    if (maxSamples < 0) {
      throw new IllegalArgumentException("maxSamples must be non-negative: " + maxSamples);
    }
    HashMap<Path, FragmentRange> rangeByPath = new HashMap<>();
    int total = 0;
    for (Path path : trainCollection.fragmentPaths()) {
      EpsilonDecisionFragmentStore.FragmentMetadata metadata = store.fragmentMetadata(path);
      EpsilonDecisionFragmentStore.requireExactIdentity(path, metadata, expectedIdentity);
      if (rangeByPath.put(path, new FragmentRange(total, metadata.samples())) != null) {
        throw new IOException("Duplicate fragment path in training collection: " + path);
      }
      total = Math.addExact(total, metadata.samples());
    }
    if (total != trainCollection.samples()) {
      throw new IOException(
          "Decision training collection sample count mismatch: expected="
              + trainCollection.samples()
              + " actual="
              + total);
    }
    int selectedCount = Math.min(maxSamples, total);
    BitSet selectedIndexes = selectStreamingIndexes(total, selectedCount, seed);
    EpsilonDecisionTrainingSampleDescriptorReader reader =
        path -> {
          FragmentRange range = rangeByPath.get(path);
          if (range == null) {
            throw new IOException("Fragment is not part of this training plan: " + path);
          }
          return store.loadSelectedSamples(
              path, selectedIndexes, range.globalOffset(), range.samples(), expectedIdentity);
        };
    return new StreamingTrainingPlan(reader, selectedCount);
  }

  static Spool openSpool(
      Path spoolDir,
      EpsilonDecisionFragmentStore.ExpectedIdentity expectedIdentity,
      DecisionTrainSpoolSettings settings) {
    return new Spool(
        spoolDir, expectedIdentity, settings.asyncQueueGames(), settings.asyncWriters());
  }

  private static BitSet selectStreamingIndexes(int total, int selectedCount, long seed) {
    BitSet selected = new BitSet(total);
    if (selectedCount == 0) {
      return selected;
    }
    Random random = new Random(seed);
    // Floyd 無作為抽出でサンプル本体を保持せずにn 件から k 件を一様に選ぶ処理を作る。
    for (int j = total - selectedCount; j < total; j++) {
      int candidate = random.nextInt(j + 1);
      selected.set(selected.get(candidate) ? j : candidate);
    }
    return selected;
  }

  record StreamingTrainingPlan(EpsilonDecisionTrainingSampleDescriptorReader reader, int samples) {}

  private record FragmentRange(int globalOffset, int samples) {}

  record TrainCollection(
      int samples,
      EpsilonDecisionFragmentStore fragmentStore,
      EpsilonDecisionFragmentStore.ExpectedIdentity expectedIdentity,
      List<Path> fragmentPaths,
      List<Path> trajectoryPayloadFiles) {

    TrainCollection {
      fragmentPaths = List.copyOf(fragmentPaths);
      trajectoryPayloadFiles = List.copyOf(trajectoryPayloadFiles);
    }

    int fragmentCount() {
      return fragmentPaths.size();
    }

    TrainCollection withTrajectoryPayloadFiles(List<Path> files) {
      return new TrainCollection(samples, fragmentStore, expectedIdentity, fragmentPaths, files);
    }

    void discardFragments() throws IOException {
      // 学習入力元として読み終わった学習データファイルは反復回数一時保存領域ごと掃除する。
      // 異常終了からの復旧用の永続アーカイブ ではなく、メモリ削減のための一時領域なので、
      // 学習データファイルだけでなく空ディレクトリ・古いファイル・一時ファイル も残さない。
      IOException failureToThrow = null;
      try {
        fragmentStore.discardAll();
      } catch (IOException failure) {
        failureToThrow = failure;
      }
      for (Path payloadFile : trajectoryPayloadFiles) {
        try {
          Files.deleteIfExists(payloadFile);
        } catch (IOException failure) {
          if (failureToThrow == null) {
            failureToThrow = failure;
          } else {
            failureToThrow.addSuppressed(failure);
          }
        }
      }
      if (failureToThrow != null) {
        throw failureToThrow;
      }
    }
  }

  /**
   * 完了ゲームを容量制限付きキューから複数書き込み処理へ渡し、学習データファイル収集の完成まで所有する。
   *
   * <p>{@link #finish()} は全ワーカーの終了待ちと失敗再送出を終えた後にだけ収集を返す。
   */
  static final class Spool implements EpsilonDecisionArena.CompletedGameSink {
    private static final GameWork END = new GameWork(-1, null);

    private final SpoolAggregate aggregate;
    private final ArrayList<BlockingQueue<GameWork>> queues;
    private final Semaphore queueSlots;
    private final AtomicReference<Throwable> failure = new AtomicReference<>();
    private final ArrayList<Thread> workers;
    private boolean finished;

    private Spool(
        Path spoolDir,
        EpsilonDecisionFragmentStore.ExpectedIdentity expectedIdentity,
        int queueGames,
        int writerCount) {
      if (writerCount <= 0) {
        throw new IllegalArgumentException("Decision train spool writerCount must be positive");
      }
      if (queueGames <= 0) {
        throw new IllegalArgumentException(
            "Decision train spool queueGames must be positive: " + queueGames);
      }
      EpsilonDecisionFragmentStore store = new EpsilonDecisionFragmentStore(spoolDir);
      aggregate = new SpoolAggregate(store, expectedIdentity);
      queueSlots = new Semaphore(queueGames);
      queues = new ArrayList<>(writerCount);
      workers = new ArrayList<>(writerCount);
      for (int i = 0; i < writerCount; i++) {
        BlockingQueue<GameWork> queue = new LinkedBlockingQueue<>();
        queues.add(queue);
        Thread worker =
            new Thread(
                () -> runWriter(queue), "decision-train-spool-writer-" + FormatUtils.zeroPad(i, 2));
        worker.setDaemon(true);
        worker.start();
        workers.add(worker);
      }
    }

    @Override
    public void accept(int gameIndex, long seed, EpsilonDecisionCompletedGame game)
        throws Exception {
      rethrowFailure();
      if (finished) {
        throw new IllegalStateException("Decision train spool is already finished");
      }
      if (gameIndex < 0 || game == null) {
        throw new IllegalArgumentException("Decision train spool requires a completed game");
      }
      GameWork work = new GameWork(gameIndex, game);
      BlockingQueue<GameWork> queue = queues.get(spoolWriterIndex(seed, queues.size()));
      try {
        while (!queueSlots.tryAcquire(100L, TimeUnit.MILLISECONDS)) {
          rethrowFailure();
        }
        queue.add(work);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IOException("Interrupted while enqueueing Decision train spool work", e);
      }
      rethrowFailure();
    }

    TrainCollection finish() throws Exception {
      if (finished) {
        throw new IllegalStateException("Decision train spool is already finished");
      }
      finished = true;
      try {
        for (BlockingQueue<GameWork> queue : queues) {
          queue.add(END);
        }
        joinWorkersFailFast();
      } catch (InterruptedException e) {
        interruptWorkers();
        joinWorkersAfterInterrupt();
        Thread.currentThread().interrupt();
        throw new IOException("Interrupted while draining Decision train spool writers", e);
      }
      rethrowFailure();
      return aggregate.toTrainCollection();
    }

    private void joinWorkersFailFast() throws InterruptedException {
      boolean cancelledForFailure = false;
      while (true) {
        if (!cancelledForFailure && failure.get() != null) {
          interruptWorkers();
          cancelledForFailure = true;
        }
        boolean anyAlive = false;
        for (Thread worker : workers) {
          if (worker.isAlive()) {
            anyAlive = true;
            worker.join(100L);
          }
        }
        if (!anyAlive) {
          return;
        }
      }
    }

    private void joinWorkersAfterInterrupt() {
      for (Thread worker : workers) {
        while (worker.isAlive()) {
          try {
            worker.join();
          } catch (InterruptedException ignored) {
            interruptWorkers();
          }
        }
      }
    }

    private void runWriter(BlockingQueue<GameWork> queue) {
      try {
        while (true) {
          GameWork work = queue.take();
          if (work == END) {
            break;
          }
          queueSlots.release();
          publishGame(work.gameIndex(), work.game());
        }
      } catch (Throwable t) {
        failure.compareAndSet(null, t);
      }
    }

    private void publishGame(int gameIndex, EpsilonDecisionCompletedGame game) throws IOException {

      aggregate.addSamples(game.samples().size());

      EpsilonDecisionFragmentStore.ExpectedIdentity identity = aggregate.expectedIdentity();
      EpsilonDecisionFragment fragment =
          new EpsilonDecisionFragment(
              identity.actorSnapshotId(),
              identity.grpTeacherIteration(),
              game,
              identity.trainingTargetIdentity());
      Path path;

      path = aggregate.store().publish(fragment);

      aggregate.addFragmentPath(gameIndex, path);
    }

    private void interruptWorkers() {
      for (Thread worker : workers) {
        worker.interrupt();
      }
    }

    private void rethrowFailure() throws Exception {
      Throwable t = failure.get();
      if (t == null) {
        return;
      }
      if (t instanceof Exception e) {
        throw e;
      }
      if (t instanceof Error e) {
        throw e;
      }
      throw new IOException("Decision train spool writer failed", t);
    }

    private record GameWork(int gameIndex, EpsilonDecisionCompletedGame game) {}
  }

  static int spoolWriterIndex(long seed, int writerCount) {
    if (writerCount <= 0) {
      throw new IllegalArgumentException("Decision train spool writerCount must be positive");
    }
    return (int) Math.floorMod(seed, (long) writerCount);
  }

  private static final class SpoolAggregate {
    private final EpsilonDecisionFragmentStore store;
    private final EpsilonDecisionFragmentStore.ExpectedIdentity expectedIdentity;
    private final ArrayList<PublishedFragment> fragments = new ArrayList<>();
    private int samples;

    private SpoolAggregate(
        EpsilonDecisionFragmentStore store,
        EpsilonDecisionFragmentStore.ExpectedIdentity expectedIdentity) {
      this.store = store;
      this.expectedIdentity = expectedIdentity;
    }

    private EpsilonDecisionFragmentStore store() {
      return store;
    }

    private EpsilonDecisionFragmentStore.ExpectedIdentity expectedIdentity() {
      return expectedIdentity;
    }

    private synchronized void addSamples(int count) {
      samples += count;
    }

    private synchronized void addFragmentPath(long ordinal, Path path) {
      fragments.add(new PublishedFragment(ordinal, path));
    }

    private synchronized TrainCollection toTrainCollection() {
      ArrayList<PublishedFragment> ordered = new ArrayList<>(fragments);
      ordered.sort(Comparator.comparingLong(PublishedFragment::ordinal));
      long previousOrdinal = Long.MIN_VALUE;
      ArrayList<Path> fragmentPaths = new ArrayList<>(ordered.size());
      for (PublishedFragment fragment : ordered) {
        if (fragment.ordinal() == previousOrdinal) {
          throw new IllegalStateException(
              "Duplicate Decision train spool fragment ordinal: " + fragment.ordinal());
        }
        previousOrdinal = fragment.ordinal();
        fragmentPaths.add(fragment.path());
      }
      return new TrainCollection(samples, store, expectedIdentity, fragmentPaths, List.of());
    }

    private record PublishedFragment(long ordinal, Path path) {}
  }
}
