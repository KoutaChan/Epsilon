package com.epsilon.reviewer;

import com.epsilon.reviewer.RecordFetcher.SourceRecord;
import com.epsilon.reviewer.dto.ModelInfo;
import com.epsilon.reviewer.dto.RecordMetadata;
import com.epsilon.reviewer.dto.ReviewResult;
import com.google.gson.*;
import com.google.gson.stream.JsonWriter;
import java.io.*;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.*;
import java.util.*;
import java.util.zip.GZIPOutputStream;

/** 結果と元牌譜を同時に確定し、小さい一覧データだけを履歴で読みます。 */
public final class ResultStore {
  private static final Gson JSON =
      new GsonBuilder().disableHtmlEscaping().serializeNulls().create();
  private final Path directory;
  private final Map<String, String> sharedOwners = new HashMap<>();
  private final Set<Path> activeTemporaryDirectories = new HashSet<>();
  private final Map<Path, Integer> activeReaders = new HashMap<>();

  public ResultStore(Path directory) throws IOException {
    this.directory = directory.toAbsolutePath().normalize();
    Files.createDirectories(this.directory);
    indexSharedWebResults();
  }

  public void save(String owner, ReviewResult result, SourceRecord source, boolean desktop)
      throws IOException {
    if (result.formatVersion() != ReviewResult.CURRENT_FORMAT_VERSION) throw unsupportedResult();
    String id = result.resultId();
    Path target = resultDirectory(owner, id);
    Path staging;
    synchronized (this) {
      Files.createDirectories(target.getParent());
      staging = Files.createTempDirectory(target.getParent(), ".staging-");
      activeTemporaryDirectories.add(staging);
    }
    try {
      writeCompressedResult(staging.resolve("result.json.gz"), result);
      write(staging.resolve("source.bin"), source.bytes());
      writeJson(staging.resolve("source.json"), new SourceInfo(source.fileName(), source.source()));
      writeJson(
          staging.resolve("summary.json"),
          new StoredSummary(
              result.resultId(),
              result.createdAt(),
              result.metadata(),
              result.model(),
              desktop,
              Instant.now().toString()));
      synchronized (this) {
        Files.move(staging, target, StandardCopyOption.ATOMIC_MOVE);
        if (!desktop) sharedOwners.put(id, owner);
      }
    } finally {
      try {
        if (Files.exists(staging)) removeTree(staging);
      } finally {
        synchronized (this) {
          activeTemporaryDirectories.remove(staging);
        }
      }
    }
  }

  public synchronized List<HistoryEntry> history(String owner) throws IOException {
    Path ownerDirectory = ownerDirectory(owner);
    var summaries = new ArrayList<HistoryEntry>();
    if (!Files.isDirectory(ownerDirectory)) return summaries;
    try (var directories = Files.list(ownerDirectory)) {
      for (Path path :
          directories.filter(p -> isCanonicalResultId(p.getFileName().toString())).toList()) {
        if (!Files.isRegularFile(path.resolve("result.json.gz"))
            || Files.exists(path.resolve(".deleted"))) continue;
        StoredSummary summary = readJson(path.resolve("summary.json"), StoredSummary.class);
        if (!summary.desktop())
          summaries.add(
              new HistoryEntry(
                  summary.resultId(), summary.createdAt(), summary.metadata(), summary.model()));
      }
    }
    summaries.sort(Comparator.comparing(HistoryEntry::createdAt).reversed());
    return summaries;
  }

  /** 所有者を確認してファイルを開き、転送中はストアのロックを保持しない。 */
  public synchronized ResultContent openResult(String owner, String id) throws IOException {
    return openCompressedResult(requireCurrentResultFile(requireOwnedResultDirectory(owner, id)));
  }

  /** 所有者の結果だけに共有 URL を発行し、デスクトップ保存は対象外にする。 */
  public synchronized String requireShareableResultId(String owner, String id) throws IOException {
    Path result = requireOwnedResultDirectory(owner, id);
    requireCurrentResultFile(result);
    if (readJson(result.resolve("summary.json"), StoredSummary.class).desktop())
      throw new ApiException(
          409, "share_unavailable", "Only server-stored web results can be shared.");
    return id;
  }

  /** 共有 ID は閲覧だけに使い、所有者の Cookie や元牌譜への権限には変換しない。 */
  public synchronized ResultContent openSharedResult(String id) throws IOException {
    String owner = isCanonicalResultId(id) ? sharedOwners.get(id) : null;
    if (owner == null) throw sharedResultNotFound();
    try {
      return openResult(owner, id);
    } catch (ApiException failure) {
      if (failure.status() == 404 || failure.status() == 422) throw sharedResultNotFound();
      throw failure;
    } catch (NoSuchFileException failure) {
      throw sharedResultNotFound();
    }
  }

  public synchronized SourceRecord source(String owner, String id) throws IOException {
    Path result = requireOwnedResultDirectory(owner, id);
    SourceInfo info = readJson(result.resolve("source.json"), SourceInfo.class);
    return new SourceRecord(
        Files.readAllBytes(result.resolve("source.bin")), info.fileName(), info.source());
  }

  public void delete(String owner, String id) throws IOException {
    Path result;
    boolean remove;
    synchronized (this) {
      result = requireOwnedResultDirectory(owner, id);
      // Windows の転送中ファイルは移動できないため、先に永続的な削除状態を確定する。
      write(result.resolve(".deleted"), new byte[0]);
      sharedOwners.remove(id, owner);
      remove = !activeReaders.containsKey(result);
      if (remove) activeTemporaryDirectories.add(result);
    }
    if (remove) removeDeletedResult(result);
  }

  private void releaseResultReader(Path result) throws IOException {
    boolean remove = false;
    synchronized (this) {
      int remaining = activeReaders.get(result) - 1;
      if (remaining == 0) {
        activeReaders.remove(result);
        if (Files.exists(result.resolve(".deleted"))) {
          activeTemporaryDirectories.add(result);
          remove = true;
        }
      } else activeReaders.put(result, remaining);
    }
    if (remove) removeDeletedResult(result);
  }

  private void removeDeletedResult(Path result) throws IOException {
    try {
      removeTree(result);
    } finally {
      synchronized (this) {
        activeTemporaryDirectories.remove(result);
      }
    }
  }

  public synchronized void cleanTemporary(Instant now) throws IOException {
    try (var owners = Files.list(directory)) {
      for (Path owner : owners.filter(Files::isDirectory).toList()) {
        try (var results = Files.list(owner)) {
          for (Path result : results.filter(Files::isDirectory).toList()) {
            String name = result.getFileName().toString();
            if (name.startsWith(".staging-") || name.startsWith(".deleting-")) {
              if (!activeTemporaryDirectories.contains(result)) removeTree(result);
            } else if (isCanonicalResultId(name) && Files.exists(result.resolve(".deleted"))) {
              if (!activeReaders.containsKey(result)
                  && !activeTemporaryDirectories.contains(result)) removeTree(result);
            } else if (isCanonicalResultId(name)
                && Files.isRegularFile(result.resolve("result.json.gz"))) {
              StoredSummary summary = readJson(result.resolve("summary.json"), StoredSummary.class);
              if (summary.desktop()
                  && Instant.parse(summary.storedAt()).plus(Duration.ofHours(24)).isBefore(now))
                delete(owner.getFileName().toString(), name);
            }
          }
        }
      }
    }
  }

  /** 起動時に既存の履歴情報から索引を復元し、公開リクエストでは全結果を走査しない。 */
  private void indexSharedWebResults() throws IOException {
    try (var owners = Files.list(directory)) {
      for (Path owner : owners.filter(Files::isDirectory).toList()) {
        String ownerId = owner.getFileName().toString();
        if (!ownerId.matches("[0-9a-f]{64}")) continue;
        try (var results = Files.list(owner)) {
          for (Path result : results.filter(Files::isDirectory).toList()) {
            String id = result.getFileName().toString();
            if (!isCanonicalResultId(id)
                || !Files.isRegularFile(result.resolve("result.json.gz"))
                || Files.exists(result.resolve(".deleted"))) continue;
            if (!readJson(result.resolve("summary.json"), StoredSummary.class).desktop()
                && sharedOwners.put(id, ownerId) != null)
              throw new IOException("Duplicate persisted web result ID.");
          }
        }
      }
    }
  }

  private static Path requireCurrentResultFile(Path result) {
    Path file = result.resolve("result.json.gz");
    if (!Files.isRegularFile(file)) throw unsupportedResult();
    return file;
  }

  private ResultContent openCompressedResult(Path file) throws IOException {
    FileChannel channel = FileChannel.open(file, StandardOpenOption.READ);
    try {
      long size = channel.size();
      Path result = file.getParent();
      activeReaders.merge(result, 1, Integer::sum);
      return new ResultContent(Channels.newInputStream(channel), size, result);
    } catch (IOException | RuntimeException failure) {
      channel.close();
      throw failure;
    }
  }

  private static ApiException unsupportedResult() {
    return new ApiException(
        422,
        "invalid_result",
        "Unsupported review result format. Analyze the source record again.");
  }

  private static ApiException sharedResultNotFound() {
    return new ApiException(404, "share_not_found", "Shared analysis not found.");
  }

  private Path requireOwnedResultDirectory(String owner, String id) {
    Path path = resultDirectory(owner, id);
    if (!Files.isRegularFile(path.resolve("summary.json"))
        || Files.exists(path.resolve(".deleted")))
      throw new ApiException(404, "result_not_found", "Analysis result not found.");
    return path;
  }

  private Path resultDirectory(String owner, String id) {
    if (!isCanonicalResultId(id))
      throw new ApiException(404, "result_not_found", "Analysis result not found.");
    return ownerDirectory(owner).resolve(id);
  }

  private Path ownerDirectory(String owner) {
    if (owner == null || !owner.matches("[0-9a-f]{64}"))
      throw new IllegalArgumentException("invalid owner");
    return directory.resolve(owner);
  }

  private static boolean isCanonicalResultId(String id) {
    return id != null && id.matches("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}");
  }

  private static <T> T readJson(Path path, Class<T> type) throws IOException {
    try (Reader reader = Files.newBufferedReader(path)) {
      return JSON.fromJson(reader, type);
    } catch (RuntimeException e) {
      throw new IOException("Could not read persisted reviewer data.", e);
    }
  }

  /** 呼び出し元が閉じる、圧縮済み結果の入力とファイルサイズ。最後の読取終了時に削除を確定する。 */
  public final class ResultContent implements Closeable {
    private final InputStream input;
    private final long size;
    private final Path result;
    private boolean closed;

    private ResultContent(InputStream input, long size, Path result) {
      this.input = input;
      this.size = size;
      this.result = result;
    }

    public InputStream input() {
      return input;
    }

    public long size() {
      return size;
    }

    @Override
    public void close() throws IOException {
      if (closed) return;
      closed = true;
      try {
        input.close();
      } finally {
        releaseResultReader(result);
      }
    }
  }

  public record HistoryEntry(
      String resultId, String createdAt, RecordMetadata metadata, ModelInfo model) {}

  private record SourceInfo(String fileName, String source) {}

  private record StoredSummary(
      String resultId,
      String createdAt,
      RecordMetadata metadata,
      ModelInfo model,
      boolean desktop,
      String storedAt) {}

  private static void writeCompressedResult(Path path, ReviewResult result) throws IOException {
    try (FileChannel channel =
            FileChannel.open(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
        var gzip = new GZIPOutputStream(Channels.newOutputStream(channel));
        JsonWriter writer =
            JSON.newJsonWriter(new OutputStreamWriter(gzip, StandardCharsets.UTF_8))) {
      JSON.toJson(result, ReviewResult.class, writer);
      writer.flush();
      gzip.finish();
      channel.force(true);
    }
  }

  private static void writeJson(Path path, Object value) throws IOException {
    try (FileChannel channel =
            FileChannel.open(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
        JsonWriter writer =
            JSON.newJsonWriter(
                new OutputStreamWriter(
                    Channels.newOutputStream(channel), StandardCharsets.UTF_8))) {
      JSON.toJson(value, value.getClass(), writer);
      writer.flush();
      channel.force(true);
    }
  }

  private static void write(Path path, byte[] data) throws IOException {
    try (FileChannel channel =
        FileChannel.open(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      var buffer = java.nio.ByteBuffer.wrap(data);
      while (buffer.hasRemaining()) channel.write(buffer);
      channel.force(true);
    }
  }

  private void removeTree(Path root) throws IOException {
    if (!root.toAbsolutePath().normalize().startsWith(directory) || root.equals(directory))
      throw new IOException("Refusing to delete a path outside the result directory.");
    try (var paths = Files.walk(root)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.deleteIfExists(path);
    }
  }
}
