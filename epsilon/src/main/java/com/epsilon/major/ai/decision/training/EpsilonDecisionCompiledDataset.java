package com.epsilon.major.ai.decision.training;

import com.epsilon.major.ai.decision.data.EpsilonDecisionBinaryArrayCodec;
import com.epsilon.major.ai.decision.input.DecisionBucket;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.major.ai.decision.input.DecisionInputSchema;
import com.github.luben.zstd.ZstdInputStream;
import com.github.luben.zstd.ZstdOutputStream;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/** 牌譜から生成した、事前学習用の変換済みデータセットを保存・参照する。 */
final class EpsilonDecisionCompiledDataset {

  private static final int DATA_MAGIC = 0xED22_000B;
  private static final int SAMPLE_ID_MAGIC = 0xED22_4901;
  private static final int FORMAT_VERSION = 13;
  private static final String MANIFEST_FILE = "manifest.json";
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  private final Path directory;
  private final Manifest manifest;

  private EpsilonDecisionCompiledDataset(Path directory, Manifest manifest) {
    this.directory = directory;
    this.manifest = manifest;
  }

  static EpsilonDecisionCompiledDataset open(Path directory) throws IOException {
    Path normalized = directory.toAbsolutePath().normalize();
    Path manifestFile = normalized.resolve(MANIFEST_FILE);
    if (!Files.isRegularFile(manifestFile)) {
      throw new IOException("Compiled Decision dataset manifest is missing: " + manifestFile);
    }
    Manifest manifest = GSON.fromJson(Files.readString(manifestFile), Manifest.class);
    if (manifest == null
        || manifest.formatVersion() != FORMAT_VERSION
        || !DecisionInputSchema.fingerprint().equals(manifest.schemaFingerprint())) {
      throw new IOException("Compiled Decision dataset is incompatible: " + manifestFile);
    }
    if (manifest.series() != null && !manifest.series().equals("epsilon")) {
      throw new IOException(
          "Compiled Decision dataset series mismatch: " + manifest.series() + " expected=epsilon");
    }
    long rows = 0L;
    int batches = 0;
    for (Shard shard : manifest.shards()) {
      requireArtifact(normalized, shard.file(), shard.bytes(), "data shard");
      requireArtifact(normalized, shard.sampleIdsFile(), shard.sampleIdsBytes(), "sample-id shard");
      rows += shard.rows();
      batches = Math.addExact(batches, shard.batches());
    }
    if (rows != manifest.rows() || batches != manifest.batches()) {
      throw new IOException("Compiled Decision dataset manifest totals are inconsistent");
    }
    return new EpsilonDecisionCompiledDataset(normalized, manifest);
  }

  Manifest manifest() {
    return manifest;
  }

  Path directory() {
    return directory;
  }

  List<Shard> shardOrder(int epoch) {
    ArrayList<Shard> order = new ArrayList<>(manifest.shards());
    // 新規変換のエポック 1は確定済み分割ファイルをマニフェスト順で消費する。変換前に入力元自体を
    // 識別情報由来乱数シードで並べ替えしているため、再利用時も同じ順序にそろえる。
    if (epoch != 1) {
      Collections.shuffle(order, new Random(epochSeed(epoch, manifest.identity())));
    }
    return order;
  }

  EpochCursor openEpoch(int epoch, int prefetchShards) {
    if (prefetchShards < 1) {
      throw new IllegalArgumentException("prefetchShards must be positive");
    }
    List<Shard> shardOrder = shardOrder(epoch);
    return new EpochCursor(epoch, shardOrder, Math.min(prefetchShards, shardOrder.size()));
  }

  ArrayList<DecisionHostBatch> readShard(Shard shard) throws IOException {
    return readShard(directory, shard);
  }

  static ArrayList<DecisionHostBatch> readShard(Path directory, Shard shard) throws IOException {
    Path file = directory.toAbsolutePath().normalize().resolve(shard.file());
    try (ZstdInputStream zstd =
            new ZstdInputStream(new BufferedInputStream(Files.newInputStream(file), 1 << 20));
        DataInputStream in = new DataInputStream(new BufferedInputStream(zstd, 1 << 20))) {
      if (in.readInt() != DATA_MAGIC || in.readInt() != FORMAT_VERSION) {
        throw new IOException("Invalid compiled Decision shard header: " + file);
      }
      String schema = in.readUTF();
      if (!DecisionInputSchema.fingerprint().equals(schema)) {
        throw new IOException("Compiled Decision shard schema changed: " + file);
      }
      int count = in.readInt();
      if (count != shard.batches()) {
        throw new IOException("Compiled Decision shard batch count changed: " + file);
      }
      EpsilonDecisionBinaryArrayCodec.Scratch scratch =
          new EpsilonDecisionBinaryArrayCodec.Scratch();
      ArrayList<DecisionHostBatch> decoded = new ArrayList<>(count);
      long rows = 0L;
      for (int index = 0; index < count; index++) {
        int batchRows = in.readInt();
        DecisionBucket bucket = new DecisionBucket(in.readInt(), in.readInt());
        decoded.add(
            DecisionHostBatch.takeEncodedTrainingBatch(
                batchRows,
                bucket,
                EpsilonDecisionBinaryArrayCodec.readShortArray(in, scratch),
                EpsilonDecisionBinaryArrayCodec.readFloatArray(in, scratch),
                EpsilonDecisionBinaryArrayCodec.readIntArray(in, scratch),
                EpsilonDecisionBinaryArrayCodec.readFloatArray(in, scratch)));
        rows += batchRows;
      }
      if (rows != shard.rows() || in.read() != -1) {
        throw new IOException("Compiled Decision shard payload changed: " + file);
      }
      return decoded;
    }
  }

  List<long[]> readSampleIds(Shard shard) throws IOException {
    Path file = directory.resolve(shard.sampleIdsFile());
    try (ZstdInputStream zstd =
            new ZstdInputStream(new BufferedInputStream(Files.newInputStream(file), 1 << 20));
        DataInputStream in = new DataInputStream(new BufferedInputStream(zstd, 1 << 20))) {
      if (in.readInt() != SAMPLE_ID_MAGIC || in.readInt() != FORMAT_VERSION) {
        throw new IOException("Invalid compiled Decision sample-id header: " + file);
      }
      int count = in.readInt();
      if (count != shard.batches()) {
        throw new IOException("Compiled Decision sample-id batch count changed: " + file);
      }
      EpsilonDecisionBinaryArrayCodec.Scratch scratch =
          new EpsilonDecisionBinaryArrayCodec.Scratch();
      ArrayList<long[]> decoded = new ArrayList<>(count);
      long rows = 0L;
      for (int index = 0; index < count; index++) {
        long[] sampleIds =
            EpsilonDecisionBinaryArrayCodec.readLongArray(in, Integer.MAX_VALUE, scratch);
        decoded.add(sampleIds);
        rows += sampleIds.length;
      }
      if (rows != shard.rows() || in.read() != -1) {
        throw new IOException("Compiled Decision sample-id payload changed: " + file);
      }
      return decoded;
    }
  }

  List<CompiledBatch> readIndexedShard(Shard shard) throws IOException {
    List<DecisionHostBatch> batches = readShard(shard);
    List<long[]> sampleIds = readSampleIds(shard);
    ArrayList<CompiledBatch> indexed = new ArrayList<>(batches.size());
    for (int index = 0; index < batches.size(); index++) {
      indexed.add(new CompiledBatch(batches.get(index), sampleIds.get(index)));
    }
    return indexed;
  }

  /** 分割ファイル単位でRAM使用量を制限し、次の分割ファイルをCPUで非同期展開するエポック読み出し位置。 */
  final class EpochCursor implements EpsilonDecisionPretrainBatchCursor {
    private final int epoch;
    private final List<Shard> shardOrder;
    private final int prefetchShards;
    private final ExecutorService executor;
    private final Deque<Future<List<DecisionHostBatch>>> pending = new ArrayDeque<>();
    private int nextShardOrdinal;
    private List<DecisionHostBatch> currentShard = List.of();
    private int currentBatchIndex;
    private boolean closed;

    private EpochCursor(int epoch, List<Shard> shardOrder, int prefetchShards) {
      this.epoch = epoch;
      this.shardOrder = shardOrder;
      this.prefetchShards = prefetchShards;
      executor =
          Executors.newFixedThreadPool(
              prefetchShards,
              runnable -> {
                Thread thread = new Thread(runnable, "decision-pretrain-shard-reader");
                thread.setDaemon(true);
                return thread;
              });
      fillPrefetchQueue();
    }

    @Override
    public DecisionHostBatch next() throws IOException {
      if (closed) {
        throw new IllegalStateException("Decision pretrain epoch cursor is closed");
      }
      while (currentBatchIndex == currentShard.size()) {
        if (pending.isEmpty()) {
          return null;
        }
        currentShard = await(pending.removeFirst());
        currentBatchIndex = 0;
        fillPrefetchQueue();
      }
      return currentShard.get(currentBatchIndex++);
    }

    @Override
    public int totalBatches() {
      return manifest.batches();
    }

    @Override
    public long totalRows() {
      return manifest.rows();
    }

    @Override
    public void close() {
      if (closed) {
        return;
      }
      closed = true;
      for (Future<List<DecisionHostBatch>> future : pending) {
        future.cancel(true);
      }
      pending.clear();
      currentShard = List.of();
      executor.shutdownNow();
    }

    private void fillPrefetchQueue() {
      while (pending.size() < prefetchShards && nextShardOrdinal < shardOrder.size()) {
        int shardOrdinal = nextShardOrdinal++;
        Shard shard = shardOrder.get(shardOrdinal);
        pending.addLast(
            executor.submit(
                () -> {
                  ArrayList<DecisionHostBatch> batches = readShard(shard);
                  Collections.shuffle(
                      batches,
                      new Random(
                          epochSeed(epoch, manifest.identity()) ^ (long) shardOrdinal << 32));
                  return batches;
                }));
      }
    }
  }

  static Writer newWriter(
      Path directory,
      String identity,
      String sourceDigest,
      String grpTeacherIdentity,
      String compilerConfiguration,
      int rowsPerShard)
      throws IOException {
    return new Writer(
        directory,
        identity,
        sourceDigest,
        grpTeacherIdentity,
        compilerConfiguration,
        rowsPerShard,
        ignored -> {});
  }

  static Writer newWriter(
      Path directory,
      String identity,
      String sourceDigest,
      String grpTeacherIdentity,
      String compilerIdentity,
      int rowsPerShard,
      ShardListener listener)
      throws IOException {
    return new Writer(
        directory,
        identity,
        sourceDigest,
        grpTeacherIdentity,
        compilerIdentity,
        rowsPerShard,
        listener);
  }

  static final class Writer implements AutoCloseable {
    private final Path directory;
    private final String identity;
    private final String sourceDigest;
    private final String grpTeacherIdentity;
    private final String compilerIdentity;
    private final int rowsPerShard;
    private final ShardListener listener;
    private final ArrayList<CompiledBatch> pending = new ArrayList<>();
    private final ArrayList<Shard> shards = new ArrayList<>();
    private long rows;
    private long nextAutomaticSampleId;
    private int batches;
    private int pendingRows;
    private boolean closed;

    private Writer(
        Path directory,
        String identity,
        String sourceDigest,
        String grpTeacherIdentity,
        String compilerIdentity,
        int rowsPerShard,
        ShardListener listener)
        throws IOException {
      if (rowsPerShard < 1) {
        throw new IllegalArgumentException("rowsPerShard must be positive");
      }
      this.directory = directory.toAbsolutePath().normalize();
      this.identity = identity;
      this.sourceDigest = sourceDigest;
      this.grpTeacherIdentity = grpTeacherIdentity;
      this.compilerIdentity = compilerIdentity;
      this.rowsPerShard = rowsPerShard;
      this.listener = listener;
      Files.createDirectories(this.directory);
      if (Files.exists(this.directory.resolve(MANIFEST_FILE))) {
        throw new IOException("Compiled Decision dataset already exists: " + this.directory);
      }
    }

    void add(DecisionHostBatch batch) throws IOException {
      long[] sampleIds = new long[batch.size()];
      for (int row = 0; row < sampleIds.length; row++) {
        sampleIds[row] = nextAutomaticSampleId++;
      }
      add(batch, sampleIds);
    }

    void add(DecisionHostBatch batch, long[] sampleIds) throws IOException {
      requireOpen();
      if (!batch.hasTrainingTargets() || batch.size() != batch.capacity()) {
        throw new IllegalArgumentException("compiled dataset requires a complete training batch");
      }
      if (sampleIds.length != batch.size()) {
        throw new IllegalArgumentException("sampleIds must match compiled batch rows");
      }
      if (!pending.isEmpty() && pendingRows + batch.size() > rowsPerShard) {
        flushShard();
      }
      pending.add(new CompiledBatch(batch, sampleIds));
      pendingRows = Math.addExact(pendingRows, batch.size());
      rows += batch.size();
      batches++;
      if (pendingRows >= rowsPerShard) {
        flushShard();
      }
    }

    void finish() throws IOException {
      requireOpen();
      flushShard();
      if (batches == 0) {
        throw new IOException("Cannot finalize an empty compiled Decision dataset");
      }
      Manifest manifest =
          new Manifest(
              FORMAT_VERSION,
              DecisionInputSchema.fingerprint(),
              identity,
              sourceDigest,
              grpTeacherIdentity,
              compilerIdentity,
              rows,
              batches,
              shards,
              "epsilon");
      Path temporary = directory.resolve(MANIFEST_FILE + ".tmp");
      Files.writeString(temporary, GSON.toJson(manifest));
      Files.move(temporary, directory.resolve(MANIFEST_FILE), StandardCopyOption.ATOMIC_MOVE);
      closed = true;
    }

    @Override
    public void close() throws IOException {
      if (closed) {
        return;
      }
      closed = true;
      pending.clear();
      pendingRows = 0;
    }

    private void flushShard() throws IOException {
      if (pending.isEmpty()) {
        return;
      }
      int shardIndex = shards.size();
      String ordinal = String.format("%05d", shardIndex);
      String dataFileName = "shard-" + ordinal + ".bin.zst";
      String idsFileName = "sample-ids-" + ordinal + ".bin.zst";
      Path dataFile = directory.resolve(dataFileName);
      Path idsFile = directory.resolve(idsFileName);
      Path dataTemporary = directory.resolve(dataFileName + ".tmp");
      Path idsTemporary = directory.resolve(idsFileName + ".tmp");
      long shardRows = pending.stream().mapToLong(row -> row.batch().size()).sum();
      writeDataShard(dataTemporary);
      writeSampleIdShard(idsTemporary);
      Files.move(
          dataTemporary,
          dataFile,
          StandardCopyOption.ATOMIC_MOVE,
          StandardCopyOption.REPLACE_EXISTING);
      Files.move(
          idsTemporary,
          idsFile,
          StandardCopyOption.ATOMIC_MOVE,
          StandardCopyOption.REPLACE_EXISTING);
      Shard shard =
          new Shard(
              dataFileName,
              pending.size(),
              shardRows,
              Files.size(dataFile),
              idsFileName,
              Files.size(idsFile));
      shards.add(shard);
      pending.clear();
      pendingRows = 0;
      listener.shardPublished(shard);
    }

    private void writeDataShard(Path file) throws IOException {
      try (OutputStream fileOut = Files.newOutputStream(file);
          ZstdOutputStream zstd =
              new ZstdOutputStream(new BufferedOutputStream(fileOut, 1 << 20)).setChecksum(true);
          DataOutputStream out = new DataOutputStream(new BufferedOutputStream(zstd, 1 << 20))) {
        out.writeInt(DATA_MAGIC);
        out.writeInt(FORMAT_VERSION);
        out.writeUTF(DecisionInputSchema.fingerprint());
        out.writeInt(pending.size());
        EpsilonDecisionBinaryArrayCodec.Scratch scratch =
            new EpsilonDecisionBinaryArrayCodec.Scratch();
        for (CompiledBatch indexed : pending) {
          DecisionHostBatch batch = indexed.batch();
          out.writeInt(batch.size());
          out.writeInt(batch.bucket().legalActionCapacity());
          out.writeInt(batch.bucket().actionTransitionCapacity());
          EpsilonDecisionBinaryArrayCodec.writeShortArray(
              out, batch.inputs().denseCategories(), scratch);
          EpsilonDecisionBinaryArrayCodec.writeFloatArray(
              out, batch.inputs().denseNumerics(), scratch);
          EpsilonDecisionBinaryArrayCodec.writeIntArray(
              out, batch.trainingTargets().categoricalSlab(), scratch);
          EpsilonDecisionBinaryArrayCodec.writeFloatArray(
              out, batch.trainingTargets().numericSlab(), scratch);
        }
      }
    }

    private void writeSampleIdShard(Path file) throws IOException {
      try (OutputStream fileOut = Files.newOutputStream(file);
          ZstdOutputStream zstd =
              new ZstdOutputStream(new BufferedOutputStream(fileOut, 1 << 20)).setChecksum(true);
          DataOutputStream out = new DataOutputStream(new BufferedOutputStream(zstd, 1 << 20))) {
        out.writeInt(SAMPLE_ID_MAGIC);
        out.writeInt(FORMAT_VERSION);
        out.writeInt(pending.size());
        EpsilonDecisionBinaryArrayCodec.Scratch scratch =
            new EpsilonDecisionBinaryArrayCodec.Scratch();
        for (CompiledBatch indexed : pending) {
          EpsilonDecisionBinaryArrayCodec.writeLongArray(
              out, indexed.sampleIds(), indexed.sampleIds().length, scratch);
        }
      }
    }

    private void requireOpen() {
      if (closed) {
        throw new IllegalStateException("Compiled Decision dataset writer is closed");
      }
    }
  }

  record Manifest(
      int formatVersion,
      String schemaFingerprint,
      String identity,
      String sourceDigest,
      String grpTeacherIdentity,
      String compilerIdentity,
      long rows,
      int batches,
      List<Shard> shards,
      String series) {}

  record Shard(
      String file, int batches, long rows, long bytes, String sampleIdsFile, long sampleIdsBytes) {}

  record CompiledBatch(DecisionHostBatch batch, long[] sampleIds) {
    CompiledBatch {
      if (sampleIds.length != batch.size()) {
        throw new IllegalArgumentException("sampleIds must match batch rows");
      }
    }
  }

  @FunctionalInterface
  interface ShardListener {
    void shardPublished(Shard shard) throws IOException;
  }

  static long epochSeed(int epoch, String identity) {
    long seed = 0x5052_4554_5241_494eL ^ epoch ^ identity.hashCode();
    seed ^= seed >>> 30;
    seed *= 0xbf58_476d_1ce4_e5b9L;
    seed ^= seed >>> 27;
    seed *= 0x94d0_49bb_1331_11ebL;
    return seed ^ (seed >>> 31);
  }

  private static void requireArtifact(Path directory, String name, long bytes, String label)
      throws IOException {
    Path file = directory.resolve(name).normalize();
    if (!file.getParent().equals(directory)
        || !Files.isRegularFile(file)
        || Files.size(file) != bytes) {
      throw new IOException("Compiled Decision " + label + " changed: " + file);
    }
  }

  private static <T> T await(Future<T> future) throws IOException {
    try {
      return future.get();
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while reading compiled Decision dataset", interrupted);
    } catch (ExecutionException failed) {
      throw new IOException("Compiled Decision shard reader failed", failed.getCause());
    }
  }
}
