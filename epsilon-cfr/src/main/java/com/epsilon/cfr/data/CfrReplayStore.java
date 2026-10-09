package com.epsilon.cfr.data;

import com.github.luben.zstd.Zstd;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestOutputStream;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/** 学習データをディスク上で共有し、後悔学習用と平均方策学習用の索引を永続化する。一つの学習区間につき、一つの書き込み処理が所有する。 */
public final class CfrReplayStore implements AutoCloseable {
  private final Path directory;
  private final String solveId;
  private final Index regret;
  private final Index average;
  private final long maximumRowBytes;
  private final long maximumStoredBytes;
  private final long cacheBytes;
  private final CfrPayloadStore payloads;
  private final Map<EpisodeId, byte[]> episodes = new TreeMap<>();
  private final Set<String> samplerPins = new HashSet<>();
  private long seenRows;
  private boolean poisoned;
  private long committedGeneration = -1;

  public CfrReplayStore(
      Path directory,
      String solveId,
      int regretCapacity,
      int averageCapacity,
      long regretSeed,
      long averageSeed,
      long maximumRowBytes,
      long cacheBytes,
      long maximumStoredBytes)
      throws IOException {
    if (solveId == null
        || solveId.isBlank()
        || maximumRowBytes < 76
        || maximumStoredBytes < maximumRowBytes
        || cacheBytes < 0
        || regretSeed == averageSeed)
      throw new IllegalArgumentException("Invalid CFR replay configuration");
    this.directory = directory.toAbsolutePath().normalize();
    this.solveId = solveId;
    this.regret = new Index(regretCapacity, regretSeed);
    this.average = new Index(averageCapacity, averageSeed);
    this.maximumRowBytes = maximumRowBytes;
    this.maximumStoredBytes = maximumStoredBytes;
    this.cacheBytes = cacheBytes;
    this.payloads = new CfrPayloadStore(this.directory.resolve("payloads"), cacheBytes);
  }

  /** 保存済みの対局と内容が同じなら{@code false}を返す。同じ識別子で内容が異なる再登録は拒否する。 */
  public synchronized boolean appendEpisode(
      long iteration, long episodeId, List<CfrTrainingRow> rows) throws IOException {
    requireHealthy();
    EpisodeId key = new EpisodeId(iteration, episodeId);
    Set<CfrSampleId> unique = new HashSet<>();
    for (CfrTrainingRow row : rows) {
      if (row.id().iteration() != iteration
          || row.id().episodeId() != episodeId
          || !unique.add(row.id())
          || row.serializedBytes() > maximumRowBytes)
        throw new IllegalArgumentException("Invalid episode row batch");
    }
    byte[] fingerprint = fingerprint(rows);
    byte[] previous = episodes.get(key);
    if (previous != null) {
      if (!MessageDigest.isEqual(previous, fingerprint))
        throw new IllegalArgumentException("Conflicting CFR episode retry: " + key);
      return false;
    }
    long required = 0;
    for (CfrTrainingRow row : rows) {
      if (regret.wouldRetain(row.id()) || average.wouldRetain(row.id()))
        required = Math.addExact(required, CfrPayloadStore.maximumCompressedBytes(row));
    }
    if (required > maximumStoredBytes - payloads.storedBytes()) {
      throw new IOException(
          "CFR replay storage budget exhausted; compact or increase budget before retrying");
    }
    try {
      for (CfrTrainingRow row : rows) {
        if (!regret.wouldRetain(row.id()) && !average.wouldRetain(row.id())) continue;
        Entry entry =
            new Entry(
                row.id(), row.logRegretWeight(), row.logAverageWeight(), payloads.append(row));
        regret.offer(entry);
        average.offer(entry);
      }
      // saveで全データをディスクへ同期してから、変更不可の索引を公開する。
      episodes.put(key, fingerprint);
      seenRows = Math.addExact(seenRows, rows.size());
      return true;
    } catch (IOException | RuntimeException error) {
      poisoned = true;
      throw error;
    }
  }

  public synchronized LogWeightedReplay regretSampler() {
    return sampler(regret, true);
  }

  public synchronized LogWeightedReplay averageSampler() {
    return sampler(average, false);
  }

  private LogWeightedReplay sampler(Index index, boolean isRegret) {
    requireHealthy();
    List<Entry> entries = index.entries();
    for (Entry entry : entries) samplerPins.add(entry.ref.file());
    return new LogWeightedReplay(
        entries.size(),
        i -> isRegret ? entries.get(i).logRegret : entries.get(i).logAverage,
        i -> hydrate(entries.get(i)));
  }

  private CfrTrainingRow hydrate(Entry entry) {
    return hydrate(entry, false);
  }

  private CfrTrainingRow hydrate(Entry entry, boolean verifyDisk) {
    try {
      CfrTrainingRow row = verifyDisk ? payloads.readVerified(entry.ref) : payloads.read(entry.ref);
      if (row.serializedBytes() > maximumRowBytes
          || !row.id().equals(entry.id)
          || Double.compare(row.logRegretWeight(), entry.logRegret) != 0
          || Double.compare(row.logAverageWeight(), entry.logAverage) != 0) {
        throw new IOException("CFR payload/index identity mismatch");
      }
      return row;
    } catch (IOException error) {
      throw new UncheckedIOException(error);
    }
  }

  /** サンプラーによるデータ領域の保持を解除する。返却済みのサンプラーがすべて使用を終えてから呼ぶ必要がある。 */
  public synchronized void releaseSamplers() {
    samplerPins.clear();
  }

  public synchronized Path save(long generation) throws IOException {
    requireHealthy();
    if (generation < 0) throw new IllegalArgumentException("Negative replay generation");
    payloads.seal();
    Path snapshot = snapshotPath(this.directory, generation);
    if (Files.exists(snapshot))
      throw new IOException("Immutable CFR replay generation already exists: " + generation);
    CfrReplayCodec.writeEnvelope(
        snapshot, CfrReplayCodec.REPLAY, out -> writeSnapshot(out, generation));
    writePointer(generation);
    committedGeneration = generation;
    return snapshot;
  }

  /** 不要な世代を削除する前に、読み込みまたは保存を済ませた世代へ公開ポインタを戻す。 */
  public synchronized void restoreCurrentPointer() throws IOException {
    requireHealthy();
    if (committedGeneration < 0) throw new IOException("No committed CFR replay generation");
    writePointer(committedGeneration);
  }

  private void writePointer(long generation) throws IOException {
    CfrReplayCodec.writeEnvelope(
        this.directory.resolve("current.bin"),
        CfrReplayCodec.REPLAY_POINTER,
        out -> {
          out.writeUTF(solveId);
          out.writeLong(generation);
        });
  }

  public static CfrReplayStore load(Path directory, String expectedSolveId) throws IOException {
    long generation =
        CfrReplayCodec.readEnvelope(
            directory.resolve("current.bin"),
            CfrReplayCodec.REPLAY_POINTER,
            in -> {
              if (!expectedSolveId.equals(in.readUTF()))
                throw new IOException("CFR solve mismatch");
              return in.readLong();
            });
    return loadGeneration(directory, expectedSolveId, generation);
  }

  public static CfrReplayStore loadGeneration(
      Path directory, String expectedSolveId, long generation) throws IOException {
    return CfrReplayCodec.readEnvelope(
        snapshotPath(directory, generation),
        CfrReplayCodec.REPLAY,
        in -> readSnapshot(in, directory, expectedSolveId, generation));
  }

  public static Path snapshotPath(Path directory, long generation) {
    if (generation < 0) throw new IllegalArgumentException("Negative replay generation");
    return directory.resolve("replay-" + generation + ".bin");
  }

  private void writeSnapshot(DataOutputStream out, long generation) throws IOException {
    out.writeUTF(solveId);
    out.writeLong(generation);
    out.writeInt(regret.capacity);
    out.writeInt(average.capacity);
    out.writeLong(regret.seed);
    out.writeLong(average.seed);
    out.writeLong(maximumRowBytes);
    out.writeLong(cacheBytes);
    out.writeLong(maximumStoredBytes);
    out.writeLong(seenRows);
    out.writeInt(episodes.size());
    for (var entry : episodes.entrySet()) {
      out.writeLong(entry.getKey().iteration);
      out.writeLong(entry.getKey().episodeId);
      out.write(entry.getValue());
    }
    Map<CfrSampleId, Entry> all = union();
    out.writeInt(all.size());
    for (Entry entry : all.values()) {
      CfrReplayCodec.writeId(out, entry.id);
      out.writeDouble(entry.logRegret);
      out.writeDouble(entry.logAverage);
      out.writeUTF(entry.ref.file());
      out.writeLong(entry.ref.offset());
      out.writeInt(entry.ref.length());
      out.writeUTF(entry.ref.sha256());
      out.writeBoolean(regret.byId.containsKey(entry.id));
      out.writeBoolean(average.byId.containsKey(entry.id));
    }
  }

  private static CfrReplayStore readSnapshot(
      DataInputStream in, Path directory, String expectedSolveId, long expectedGeneration)
      throws IOException {
    String solve = in.readUTF();
    if (!solve.equals(expectedSolveId) || in.readLong() != expectedGeneration)
      throw new IOException("CFR replay generation/solve mismatch");
    int regretCapacity = in.readInt();
    int averageCapacity = in.readInt();
    long regretSeed = in.readLong();
    long averageSeed = in.readLong();
    long maximumRowBytes = in.readLong();
    long cacheBytes = in.readLong();
    long maximumStoredBytes = in.readLong();
    CfrReplayStore store =
        new CfrReplayStore(
            directory,
            solve,
            regretCapacity,
            averageCapacity,
            regretSeed,
            averageSeed,
            maximumRowBytes,
            cacheBytes,
            maximumStoredBytes);
    store.seenRows = in.readLong();
    if (store.seenRows < 0) throw new IOException("Invalid CFR seen row count");
    int episodes = CfrBinaryIO.readCount(in, 100_000_000, "episodes");
    for (int i = 0; i < episodes; i++) {
      EpisodeId key = new EpisodeId(in.readLong(), in.readLong());
      byte[] hash = new byte[32];
      in.readFully(hash);
      if (store.episodes.put(key, hash) != null)
        throw new IOException("Invalid CFR episode journal");
    }
    int count =
        CfrBinaryIO.readCount(in, Math.addExact(regretCapacity, averageCapacity), "retained rows");
    Set<CfrSampleId> ids = new HashSet<>();
    for (int i = 0; i < count; i++) {
      CfrSampleId id = CfrReplayCodec.readId(in);
      double logRegret = in.readDouble();
      double logAverage = in.readDouble();
      CfrDataChecks.logWeight(logRegret);
      CfrDataChecks.logWeight(logAverage);
      CfrPayloadStore.Ref ref =
          new CfrPayloadStore.Ref(in.readUTF(), in.readLong(), in.readInt(), in.readUTF());
      if (!ids.add(id)
          || ref.length() > Zstd.compressBound(maximumRowBytes)
          || !store.episodes.containsKey(new EpisodeId(id.iteration(), id.episodeId()))) {
        throw new IOException("Invalid CFR replay row identity");
      }
      Entry entry = new Entry(id, logRegret, logAverage, ref);
      boolean inRegret = in.readBoolean();
      boolean inAverage = in.readBoolean();
      if (!inRegret && !inAverage) throw new IOException("Unreferenced CFR replay payload");
      if (inRegret && store.regret.byId.size() == regretCapacity
          || inAverage && store.average.byId.size() == averageCapacity) {
        throw new IOException("CFR replay exceeds capacity");
      }
      if (inRegret) store.regret.offer(entry);
      if (inAverage) store.average.offer(entry);
      store.payloads.validate(ref);
    }
    store.committedGeneration = expectedGeneration;
    return store;
  }

  /** 保持中の行を重複なく書き直す。保存済みの世代や使用中のサンプラーが参照している古いデータ領域は残す。 */
  public synchronized void compact() throws IOException {
    requireHealthy();
    List<Entry> entries = new ArrayList<>(union().values());
    long rewriteBytes = entries.stream().mapToLong(entry -> entry.ref.length()).sum();
    if (rewriteBytes > maximumStoredBytes - payloads.storedBytes()) {
      throw new IOException("CFR compaction requires temporary disk headroom");
    }
    try {
      payloads.seal();
      Index newRegret = new Index(regret.capacity, regret.seed);
      Index newAverage = new Index(average.capacity, average.seed);
      for (Entry entry : entries) {
        hydrate(entry, true);
        CfrPayloadStore.Ref ref = payloads.copyFrame(entry.ref);
        Entry replacement = new Entry(entry.id, entry.logRegret, entry.logAverage, ref);
        if (regret.byId.containsKey(entry.id)) newRegret.offer(replacement);
        if (average.byId.containsKey(entry.id)) newAverage.offer(replacement);
      }
      payloads.seal();
      regret.replace(newRegret);
      average.replace(newAverage);
    } catch (IOException | RuntimeException failure) {
      // 書き込みに失敗すると、サイズの集計に含まれないデータが末尾に残る場合がある。
      // 書き込みを再開するには、保存済みの世代を読み直して実際の使用量を集計し直す必要がある。
      poisoned = true;
      if (failure instanceof UncheckedIOException io) throw io.getCause();
      throw failure;
    }
  }

  /** 保存済みのどの世代からも、使用中のどのサンプラーからも参照されていないデータ領域だけを削除する。 */
  public synchronized long collectGarbage() throws IOException {
    requireHealthy();
    Set<String> referenced = new HashSet<>(samplerPins);
    for (Entry entry : union().values()) referenced.add(entry.ref.file());
    try (var files = Files.list(directory)) {
      for (Path file :
          files
              .filter(path -> path.getFileName().toString().matches("replay-[0-9]+\\.bin"))
              .toList()) {
        long generation =
            Long.parseLong(
                file.getFileName().toString().replace("replay-", "").replace(".bin", ""));
        referenced.addAll(snapshotReferences(file, solveId, generation));
      }
    }
    return payloads.garbageCollect(referenced);
  }

  public synchronized Stats stats() throws IOException {
    Map<CfrSampleId, Entry> all = union();
    return new Stats(
        episodes.size(),
        seenRows,
        regret.byId.size(),
        average.byId.size(),
        all.size(),
        all.values().stream().mapToLong(entry -> entry.ref.length()).sum(),
        storedBytes(),
        payloads.cachedBytes());
  }

  public long storedBytes() throws IOException {
    return payloads.storedBytes();
  }

  /** 抽選重みが0の行も含め、保持中のすべてのデータを検証する。ファイルは変更しない。 */
  public synchronized int verifyPayloads() throws IOException {
    requireHealthy();
    Map<CfrSampleId, Entry> entries = union();
    try {
      for (Entry entry : entries.values()) hydrate(entry, true);
    } catch (UncheckedIOException error) {
      throw error.getCause();
    }
    return entries.size();
  }

  public String solveId() {
    return solveId;
  }

  public Path directory() {
    return directory;
  }

  public synchronized boolean containsEpisode(long iteration, long episodeId) {
    return episodes.containsKey(new EpisodeId(iteration, episodeId));
  }

  private Map<CfrSampleId, Entry> union() {
    Map<CfrSampleId, Entry> all = new TreeMap<>(regret.byId);
    all.putAll(average.byId);
    return all;
  }

  private static byte[] fingerprint(List<CfrTrainingRow> rows) throws IOException {
    var digest = CfrReplayCodec.sha256();
    try (DataOutputStream out =
        new DataOutputStream(new DigestOutputStream(OutputStream.nullOutputStream(), digest))) {
      out.writeInt(rows.size());
      for (CfrTrainingRow row : rows) CfrReplayCodec.writeRow(out, row);
    }
    return digest.digest();
  }

  private static Set<String> snapshotReferences(Path path, String solveId, long generation)
      throws IOException {
    return CfrReplayCodec.readEnvelope(
        path,
        CfrReplayCodec.REPLAY,
        in -> {
          if (!solveId.equals(in.readUTF()) || generation != in.readLong())
            throw new IOException("Replay reference mismatch");
          int regretCapacity = in.readInt();
          int averageCapacity = in.readInt();
          for (int field = 0; field < 6; field++) in.readLong();
          int episodes = CfrBinaryIO.readCount(in, 100_000_000, "episodes");
          in.skipNBytes(48L * episodes);
          int count =
              CfrBinaryIO.readCount(
                  in, Math.addExact(regretCapacity, averageCapacity), "retained rows");
          Set<String> references = new HashSet<>();
          for (int i = 0; i < count; i++) {
            CfrReplayCodec.readId(in);
            in.readDouble();
            in.readDouble();
            var ref =
                new CfrPayloadStore.Ref(in.readUTF(), in.readLong(), in.readInt(), in.readUTF());
            references.add(ref.file());
            in.readBoolean();
            in.readBoolean();
          }
          return references;
        });
  }

  private void requireHealthy() {
    if (poisoned)
      throw new IllegalStateException("Replay append failed; reload a committed generation");
  }

  @Override
  public void close() throws IOException {
    payloads.close();
  }

  public record Stats(
      long episodes,
      long seenRows,
      int regretRows,
      int averageRows,
      int uniquePayloads,
      long livePayloadBytes,
      long storedBytes,
      long cachedBytes) {}

  private record EpisodeId(long iteration, long episodeId) implements Comparable<EpisodeId> {
    EpisodeId {
      if (iteration <= 0 || episodeId < 0)
        throw new IllegalArgumentException("Invalid episode identity");
    }

    @Override
    public int compareTo(EpisodeId other) {
      int c = Long.compare(iteration, other.iteration);
      return c == 0 ? Long.compare(episodeId, other.episodeId) : c;
    }
  }

  private record Entry(
      CfrSampleId id, double logRegret, double logAverage, CfrPayloadStore.Ref ref) {}

  private static final class Index {
    final int capacity;
    final long seed;
    final Map<CfrSampleId, Entry> byId = new HashMap<>();
    final TreeSet<Entry> ordered;

    Index(int capacity, long seed) {
      if (capacity <= 0 || capacity > 10_000_000)
        throw new IllegalArgumentException("Invalid replay capacity");
      this.capacity = capacity;
      this.seed = seed;
      ordered = new TreeSet<>((a, b) -> compare(a.id, b.id));
    }

    int compare(CfrSampleId first, CfrSampleId second) {
      int c = Long.compareUnsigned(first.retentionPriority(seed), second.retentionPriority(seed));
      return c == 0 ? first.compareTo(second) : c;
    }

    boolean wouldRetain(CfrSampleId id) {
      return byId.size() < capacity || compare(id, ordered.last().id) < 0;
    }

    void offer(Entry entry) {
      if (byId.containsKey(entry.id) || !wouldRetain(entry.id)) return;
      if (byId.size() == capacity) byId.remove(ordered.pollLast().id);
      ordered.add(entry);
      byId.put(entry.id, entry);
    }

    List<Entry> entries() {
      return byId.values().stream().sorted(Comparator.comparing(Entry::id)).toList();
    }

    void replace(Index source) {
      byId.clear();
      ordered.clear();
      byId.putAll(source.byId);
      ordered.addAll(source.ordered);
    }
  }
}
