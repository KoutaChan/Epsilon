package com.epsilon.pico.ai.decision.training;

import com.epsilon.ai.decision.EpsilonDecisionHlGauss;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpTeacherIdentity;
import com.epsilon.pico.ai.decision.EpsilonUtilityTargets;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionDataException;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionSample;
import com.epsilon.pico.ai.decision.input.DecisionBucket;
import com.epsilon.pico.ai.decision.input.DecisionHostBatch;
import com.epsilon.pico.ai.decision.input.DecisionInputSchema;
import com.epsilon.pico.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.pico.config.settings.DecisionPretrainSettings;
import com.epsilon.pico.training.EpsilonLogPretrainDataCollector;
import com.epsilon.replay.ReplayRecordReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.BasicFileAttributes;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 牌譜の再生と GRP 教師値の計算を一度だけ行い、事前学習用の分割データとして保存する。 */
final class EpsilonDecisionPretrainDatasetCompiler {

  private static final Logger log =
      LoggerFactory.getLogger(EpsilonDecisionPretrainDatasetCompiler.class);
  private static final String TARGET_PROTOCOL =
      "frozen-grp-immediate-next-boundary-scalar-utility-v4";

  private EpsilonDecisionPretrainDatasetCompiler() {}

  static Compilation compileOrOpen(
      Path datasetRoot,
      List<Path> sourceFiles,
      EpsilonGrpTrainingSession grpSession,
      DecisionPretrainSettings settings)
      throws IOException {
    return compileOrOpen(
        datasetRoot,
        sourceFiles,
        grpSession,
        settings,
        EpsilonUtilityTargets.configuredTrainingProfile());
  }

  static Compilation compileOrOpen(
      Path datasetRoot,
      List<Path> sourceFiles,
      EpsilonGrpTrainingSession grpSession,
      DecisionPretrainSettings settings,
      EpsilonUtilityProfile utilityProfile)
      throws IOException {
    if (sourceFiles.isEmpty()) {
      throw new IllegalArgumentException("Decision pretrain source files must not be empty");
    }
    String sourceDigest = sourceDigest(sourceFiles);
    EpsilonGrpTeacherIdentity teacher = grpSession.identity();
    String teacherIdentity = teacher.iteration() + ":" + teacher.checkpointSha256();
    String compilerIdentity = createCompilerIdentity(settings, utilityProfile);
    String identity =
        sha256(
            DecisionInputSchema.fingerprint()
                + "\n"
                + sourceDigest
                + "\n"
                + teacherIdentity
                + "\n"
                + compilerIdentity);
    Path directory = datasetRoot.toAbsolutePath().normalize().resolve(identity.substring(0, 20));
    if (Files.isRegularFile(directory.resolve("manifest.json"))) {
      EpsilonDecisionCompiledDataset existing = EpsilonDecisionCompiledDataset.open(directory);
      log.info(
          "Reusing compiled Decision warm-start dataset: directory={} rows={} batches={} shards={}",
          directory,
          existing.manifest().rows(),
          existing.manifest().batches(),
          existing.manifest().shards().size());
      return Compilation.reused(existing);
    }

    Files.createDirectories(directory);
    Compilation compilation = Compilation.fresh(directory, identity);
    ArrayList<Path> compileOrder = new ArrayList<>(sourceFiles);
    Collections.shuffle(
        compileOrder,
        new Random(EpsilonDecisionCompiledDataset.epochSeed(1, identity) ^ 0x434f_4d50_494c_45L));
    compilation.start(
        () ->
            compileDataset(
                compilation,
                compileOrder,
                grpSession,
                settings,
                sourceDigest,
                teacherIdentity,
                compilerIdentity,
                utilityProfile));
    log.info(
        "Decision pretrain base compile started asynchronously: directory={} files={} "
            + "rowsPerShard={} warmTrainingCanStartAfterFirstShard=true",
        directory,
        sourceFiles.size(),
        settings.datasetRowsPerShard());
    return compilation;
  }

  private static void compileDataset(
      Compilation compilation,
      List<Path> sourceFiles,
      EpsilonGrpTrainingSession grpSession,
      DecisionPretrainSettings settings,
      String sourceDigest,
      String teacherIdentity,
      String compilerIdentity,
      EpsilonUtilityProfile utilityProfile)
      throws IOException {
    CompilationProgress progress =
        new CompilationProgress(TARGET_PROTOCOL, settings.progressLogIntervalNanos());
    CompilerState state =
        new CompilerState(
            settings.optimizerBatchRows(),
            settings.riichiDecisionWeight(),
            settings.reactionDecisionWeight());
    int processedFiles = 0;
    ArrayList<EpsilonDecisionSample> sampleBuffer =
        new ArrayList<>(settings.compilerSampleBufferRows());
    try (EpsilonDecisionCompiledDataset.Writer writer =
        EpsilonDecisionCompiledDataset.newWriter(
            compilation.directory,
            compilation.identity,
            sourceDigest,
            teacherIdentity,
            compilerIdentity,
            settings.datasetRowsPerShard(),
            compilation::publish)) {
      try (OrderedLogReader sourceReader =
          new OrderedLogReader(
              sourceFiles, settings.compilerReaderWorkers(), settings.compilerPrefetchFiles())) {
        LogRead read;
        while ((read = sourceReader.next()) != null) {
          Path sourceFile = read.sourceFile();
          List<EpsilonDecisionSample> samples = read.samples();
          if (samples.isEmpty()) {
            throw new EpsilonDecisionDataException(
                "Decision pretrain source produced no samples: " + sourceFile);
          }
          if (!sampleBuffer.isEmpty()
              && sampleBuffer.size() + samples.size() > settings.compilerSampleBufferRows()) {
            progress.record(
                compileSamples(sampleBuffer, grpSession, state, writer, utilityProfile));
            sampleBuffer.clear();
          }
          sampleBuffer.addAll(samples);
          if (sampleBuffer.size() >= settings.compilerSampleBufferRows()) {
            progress.record(
                compileSamples(sampleBuffer, grpSession, state, writer, utilityProfile));
            sampleBuffer.clear();
          }
          processedFiles++;
        }
      }
      progress.record(compileSamples(sampleBuffer, grpSession, state, writer, utilityProfile));
      state.flush(writer);
      writer.finish();
    }
    EpsilonDecisionCompiledDataset dataset =
        EpsilonDecisionCompiledDataset.open(compilation.directory);
    if (progress.rows() != dataset.manifest().rows()) {
      throw new IOException(
          "Compiled Decision row count mismatch: prepared="
              + progress.rows()
              + " manifest="
              + dataset.manifest().rows());
    }
    compilation.complete(dataset, processedFiles);
    log.info(
        "Compiled Decision warm-start dataset complete: directory={} files={} rows={} "
            + "batches={} shards={} targetProtocol={} elapsedSeconds={} rowsPerSecond={} "
            + "sourceDigest={} teacher={}",
        compilation.directory,
        processedFiles,
        dataset.manifest().rows(),
        dataset.manifest().batches(),
        dataset.manifest().shards().size(),
        TARGET_PROTOCOL,
        progress.elapsedSeconds(),
        progress.rowsPerSecond(),
        sourceDigest,
        teacherIdentity);
  }

  private static int compileSamples(
      List<EpsilonDecisionSample> samples,
      EpsilonGrpTrainingSession grpSession,
      CompilerState state,
      EpsilonDecisionCompiledDataset.Writer writer,
      EpsilonUtilityProfile utilityProfile)
      throws IOException {
    if (samples.isEmpty()) {
      return 0;
    }
    List<EpsilonDecisionSample> prepared =
        EpsilonDecisionPretrainTargets.prepare(samples, grpSession, utilityProfile);
    state.add(prepared, writer);
    return prepared.size();
  }

  /**
   * パス・サイズ・更新時刻から牌譜キャッシュの識別子を作る。内容の全読みや破損検証は行わない。
   *
   * <p>同じサイズ・更新時刻のまま内容を差し替えた場合は、既存キャッシュを明示的に削除する。
   */
  static String sourceDigest(List<Path> sourceFiles) throws IOException {
    MessageDigest digest = digest();
    digest.update("source-metadata-v1\n".getBytes(StandardCharsets.UTF_8));
    ArrayList<Path> ordered = new ArrayList<>(sourceFiles);
    ordered.sort(Comparator.comparing(path -> path.toAbsolutePath().normalize().toString()));
    for (Path source : ordered) {
      Path normalized = source.toAbsolutePath().normalize();
      BasicFileAttributes attributes = Files.readAttributes(normalized, BasicFileAttributes.class);
      String metadata =
          normalized + "\0" + attributes.size() + "\0" + attributes.lastModifiedTime() + "\0";
      digest.update(metadata.getBytes(StandardCharsets.UTF_8));
    }
    return HexFormat.of().formatHex(digest.digest());
  }

  private static String sha256(String value) {
    return HexFormat.of().formatHex(digest().digest(value.getBytes(StandardCharsets.UTF_8)));
  }

  private static MessageDigest digest() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException(impossible);
    }
  }

  static final class Compilation implements AutoCloseable {
    private static final ShardEvent END = new ShardEvent(null, null, true);

    private final Path directory;
    private final String identity;
    private final boolean reused;
    private final BlockingQueue<ShardEvent> published = new LinkedBlockingQueue<>();
    private final CompletableFuture<EpsilonDecisionCompiledDataset> completion =
        new CompletableFuture<>();
    private final AtomicBoolean streamingEpochClaimed = new AtomicBoolean();
    private final ExecutorService compilerExecutor;
    private volatile EpsilonDecisionCompiledDataset dataset;
    private volatile int processedFiles;
    private volatile boolean closed;

    private Compilation(
        Path directory, String identity, boolean reused, EpsilonDecisionCompiledDataset dataset) {
      this.directory = directory;
      this.identity = identity;
      this.reused = reused;
      this.dataset = dataset;
      compilerExecutor =
          reused
              ? null
              : Executors.newSingleThreadExecutor(
                  runnable -> {
                    Thread thread = new Thread(runnable, "decision-pretrain-dataset-compiler");
                    thread.setDaemon(true);
                    return thread;
                  });
      if (dataset != null) {
        completion.complete(dataset);
      }
    }

    private static Compilation reused(EpsilonDecisionCompiledDataset dataset) {
      return new Compilation(dataset.directory(), dataset.manifest().identity(), true, dataset);
    }

    private static Compilation fresh(Path directory, String identity) {
      return new Compilation(directory, identity, false, null);
    }

    private void start(CompileTask task) {
      compilerExecutor.submit(
          () -> {
            try {
              task.run();
            } catch (Throwable failure) {
              fail(failure);
            }
          });
    }

    EpsilonDecisionPretrainBatchCursor openEpoch(int epoch, int prefetchShards) throws IOException {
      if (!reused && epoch == 1 && streamingEpochClaimed.compareAndSet(false, true)) {
        return new StreamingEpochCursor(prefetchShards);
      }
      return dataset().openEpoch(epoch, prefetchShards);
    }

    EpsilonDecisionCompiledDataset dataset() throws IOException {
      EpsilonDecisionCompiledDataset current = dataset;
      return current == null ? await(completion) : current;
    }

    int processedFiles() {
      return processedFiles;
    }

    boolean reused() {
      return reused;
    }

    @Override
    public void close() {
      if (closed) {
        return;
      }
      closed = true;
      if (compilerExecutor != null) {
        compilerExecutor.shutdownNow();
      }
    }

    private void publish(EpsilonDecisionCompiledDataset.Shard shard) {
      published.add(new ShardEvent(shard, null, false));
    }

    private void complete(EpsilonDecisionCompiledDataset result, int processedFiles) {
      this.dataset = result;
      this.processedFiles = processedFiles;
      completion.complete(result);
      published.add(END);
      compilerExecutor.shutdown();
    }

    private void fail(Throwable failure) {
      completion.completeExceptionally(failure);
      published.add(new ShardEvent(null, failure, true));
      compilerExecutor.shutdown();
    }

    private final class StreamingEpochCursor implements EpsilonDecisionPretrainBatchCursor {
      private final ExecutorService reader;
      private final long epochSeed = EpsilonDecisionCompiledDataset.epochSeed(1, identity);
      private Future<List<DecisionHostBatch>> nextShard;
      private List<DecisionHostBatch> current = List.of();
      private int currentIndex;
      private int shardOrdinal;
      private boolean ended;
      private boolean cursorClosed;

      private StreamingEpochCursor(int prefetchShards) {
        if (prefetchShards < 1) {
          throw new IllegalArgumentException("prefetchShards must be positive");
        }
        reader =
            Executors.newSingleThreadExecutor(
                runnable -> {
                  Thread thread = new Thread(runnable, "decision-pretrain-streaming-shard-reader");
                  thread.setDaemon(true);
                  return thread;
                });
        nextShard = reader.submit(this::readNextShard);
      }

      @Override
      public DecisionHostBatch next() throws IOException {
        if (cursorClosed) {
          throw new IllegalStateException("Decision pretrain streaming cursor is closed");
        }
        while (currentIndex == current.size()) {
          if (ended) {
            return null;
          }
          current = await(nextShard);
          currentIndex = 0;
          if (current == null) {
            ended = true;
            return null;
          }
          nextShard = reader.submit(this::readNextShard);
        }
        return current.get(currentIndex++);
      }

      @Override
      public int totalBatches() {
        EpsilonDecisionCompiledDataset currentDataset = dataset;
        return currentDataset == null ? -1 : currentDataset.manifest().batches();
      }

      @Override
      public long totalRows() {
        EpsilonDecisionCompiledDataset currentDataset = dataset;
        return currentDataset == null ? -1L : currentDataset.manifest().rows();
      }

      @Override
      public void close() {
        if (cursorClosed) {
          return;
        }
        cursorClosed = true;
        if (nextShard != null) {
          nextShard.cancel(true);
        }
        reader.shutdownNow();
        current = List.of();
      }

      private List<DecisionHostBatch> readNextShard() throws IOException {
        ShardEvent event;
        try {
          event = published.take();
        } catch (InterruptedException interrupted) {
          Thread.currentThread().interrupt();
          throw new IOException("Interrupted while awaiting compiled Decision shard", interrupted);
        }
        if (event.failure() != null) {
          throw new IOException("Decision pretrain dataset compile failed", event.failure());
        }
        if (event.end()) {
          return null;
        }
        int ordinal = shardOrdinal++;
        ArrayList<DecisionHostBatch> batches =
            EpsilonDecisionCompiledDataset.readShard(directory, event.shard());
        Collections.shuffle(batches, new Random(epochSeed ^ (long) ordinal << 32));
        return batches;
      }
    }
  }

  private record ShardEvent(
      EpsilonDecisionCompiledDataset.Shard shard, Throwable failure, boolean end) {}

  @FunctionalInterface
  private interface CompileTask {
    void run() throws IOException;
  }

  private record LogRead(Path sourceFile, List<EpsilonDecisionSample> samples) {}

  /** XML/gzip 解析を上限付きに先読みし、main スレッドへ入力元順のまま返す。 */
  private static final class OrderedLogReader implements AutoCloseable {
    private final List<Path> sourceFiles;
    private final int prefetchFiles;
    private final ExecutorService executor;
    private final Deque<PendingRead> pending;
    private int nextSourceIndex;

    private OrderedLogReader(List<Path> sourceFiles, int workers, int prefetchFiles) {
      this.sourceFiles = sourceFiles;
      this.prefetchFiles = Math.min(prefetchFiles, sourceFiles.size());
      int activeWorkers = Math.min(workers, this.prefetchFiles);
      executor =
          Executors.newFixedThreadPool(
              activeWorkers,
              runnable -> {
                Thread thread = new Thread(runnable, "decision-pretrain-log-reader");
                thread.setDaemon(true);
                return thread;
              });
      pending = new ArrayDeque<>(this.prefetchFiles);
      fill();
    }

    private LogRead next() throws IOException {
      if (pending.isEmpty()) {
        return null;
      }
      PendingRead read = pending.removeFirst();
      try {
        LogRead result = read.future().get();
        fill();
        return result;
      } catch (InterruptedException interrupted) {
        Thread.currentThread().interrupt();
        throw new IOException("Interrupted while reading Decision pretrain logs", interrupted);
      } catch (ExecutionException failed) {
        throw new IOException(
            "Decision pretrain log reader failed: " + read.sourceFile(), failed.getCause());
      }
    }

    @Override
    public void close() {
      for (PendingRead read : pending) {
        read.future().cancel(true);
      }
      pending.clear();
      executor.shutdownNow();
    }

    private void fill() {
      while (pending.size() < prefetchFiles && nextSourceIndex < sourceFiles.size()) {
        Path sourceFile = sourceFiles.get(nextSourceIndex++);
        pending.addLast(
            new PendingRead(
                sourceFile,
                executor.submit(
                    () ->
                        new LogRead(
                            sourceFile,
                            EpsilonLogPretrainDataCollector.collectDecisionFile(sourceFile)))));
      }
    }
  }

  private record PendingRead(Path sourceFile, Future<LogRead> future) {}

  private static final class CompilationProgress {
    private final String targetProtocol;
    private final long progressLogIntervalNanos;
    private final long started = System.nanoTime();
    private long nextProgressLog;
    private long rows;

    private CompilationProgress(String targetProtocol, long progressLogIntervalNanos) {
      this.targetProtocol = targetProtocol;
      this.progressLogIntervalNanos = progressLogIntervalNanos;
      nextProgressLog = started + progressLogIntervalNanos;
    }

    private void record(int additionalRows) {
      rows += additionalRows;
      long now = System.nanoTime();
      if (now < nextProgressLog) {
        return;
      }
      double elapsedSeconds = elapsedSeconds(now);
      log.info(
          "Decision pretrain dataset compile progress: targetProtocol={} rows={} "
              + "elapsedSeconds={} rowsPerSecond={}",
          targetProtocol,
          rows,
          elapsedSeconds,
          rows / elapsedSeconds);
      nextProgressLog = now + progressLogIntervalNanos;
    }

    private long rows() {
      return rows;
    }

    private double elapsedSeconds() {
      return elapsedSeconds(System.nanoTime());
    }

    private double rowsPerSecond() {
      return rows / elapsedSeconds();
    }

    private double elapsedSeconds(long now) {
      return Math.max(1.0e-9, (now - started) / 1_000_000_000.0);
    }
  }

  private static final class CompilerState {
    private final int rowsPerBatch;
    private final Map<DecisionBucket, PendingBucket> pending = new HashMap<>();
    private final EpsilonDecisionTrainingBatch.Materializer materializer;
    private long nextSampleId;

    private CompilerState(
        int rowsPerBatch, float riichiDecisionWeight, float reactionDecisionWeight) {
      this.rowsPerBatch = rowsPerBatch;
      materializer =
          new EpsilonDecisionTrainingBatch.Materializer(
              riichiDecisionWeight, reactionDecisionWeight);
    }

    private void add(
        List<EpsilonDecisionSample> samples, EpsilonDecisionCompiledDataset.Writer writer)
        throws IOException {
      for (EpsilonDecisionSample sample : samples) {
        DecisionBucket bucket = sample.input().bucket();
        PendingBucket bucketRows =
            pending.computeIfAbsent(bucket, ignored -> new PendingBucket(rowsPerBatch));
        bucketRows.add(nextSampleId++, sample);
        if (bucketRows.samples.size() == rowsPerBatch) {
          writeBatch(bucketRows, writer);
        }
      }
    }

    private void flush(EpsilonDecisionCompiledDataset.Writer writer) throws IOException {
      ArrayList<Map.Entry<DecisionBucket, PendingBucket>> ordered =
          new ArrayList<>(pending.entrySet());
      ordered.sort(
          Comparator.comparingInt(
                  (Map.Entry<DecisionBucket, PendingBucket> entry) ->
                      entry.getKey().legalActionCapacity())
              .thenComparingInt(entry -> entry.getKey().actionTransitionCapacity()));
      for (Map.Entry<DecisionBucket, PendingBucket> entry : ordered) {
        if (!entry.getValue().samples.isEmpty()) {
          writeBatch(entry.getValue(), writer);
        }
      }
    }

    private void writeBatch(PendingBucket rows, EpsilonDecisionCompiledDataset.Writer writer)
        throws IOException {
      DecisionHostBatch batch =
          materializer.materialize(rows.samples, 0, rows.samples.size()).host();
      writer.add(batch, rows.takeSampleIds());
      rows.samples.clear();
    }
  }

  private static final class PendingBucket {
    private final ArrayList<EpsilonDecisionSample> samples;
    private long[] sampleIds;

    private PendingBucket(int capacity) {
      samples = new ArrayList<>(capacity);
      sampleIds = new long[capacity];
    }

    private void add(long sampleId, EpsilonDecisionSample sample) {
      sampleIds[samples.size()] = sampleId;
      samples.add(sample);
    }

    private long[] takeSampleIds() {
      int rows = samples.size();
      long[] result = rows == sampleIds.length ? sampleIds : Arrays.copyOf(sampleIds, rows);
      sampleIds = new long[sampleIds.length];
      return result;
    }
  }

  private static <T> T await(Future<T> future) throws IOException {
    try {
      return future.get();
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while awaiting Decision pretrain work", interrupted);
    } catch (ExecutionException failed) {
      throw new IOException("Decision pretrain dataset compile failed", failed.getCause());
    }
  }

  public static String createCompilerIdentity(DecisionPretrainSettings settings) {
    return createCompilerIdentity(settings, EpsilonUtilityTargets.configuredTrainingProfile());
  }

  public static String createCompilerIdentity(
      DecisionPretrainSettings settings, EpsilonUtilityProfile utilityProfile) {
    return "optimizerBatchRows="
        + settings.optimizerBatchRows()
        + ";sampleBufferRows="
        + settings.compilerSampleBufferRows()
        + ";datasetRowsPerShard="
        + settings.datasetRowsPerShard()
        + ";riichiWeight="
        + settings.riichiDecisionWeight()
        + ";reactionWeight="
        + settings.reactionDecisionWeight()
        + ";targetProtocol="
        + TARGET_PROTOCOL
        + ";valueDefinition="
        + EpsilonDecisionHlGauss.fingerprint(utilityProfile)
        + ";replayContract="
        + ReplayRecordReader.CONTRACT_REVISION;
  }
}
