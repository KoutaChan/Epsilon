package com.epsilon.reviewer.store;

import com.epsilon.reviewer.engine.format.ResultWriter;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.json.Json;
import com.epsilon.reviewer.json.StrictRecords;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.zip.GZIPInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 解析結果の保存（results/&lt;resultId&gt;/ に文書・牌譜・meta.json）。索引は起動時に meta.json を 1 回ずつ読んで記憶に作り、保存と削除で
 * 更新する。読みと削除・確定は読み書きの鍵で分ける。
 */
public final class ResultStore implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(ResultStore.class);

  /** meta.json の大きさの上限（バイト）。 */
  private static final int META_LIMIT = 64 * 1024;

  /** 1 件のフォルダーに保存が書くファイル。消す時もこれだけを消す。 */
  private static final List<StoreLayout> ENTRY_FILES =
      List.of(StoreLayout.DOCUMENT, StoreLayout.RECORD, StoreLayout.META);

  /** ディレクトリの同期は Linux だけが持つ（Windows はディレクトリを開けない）。 */
  private static final boolean SYNC_DIRECTORIES =
      System.getProperty("os.name", "").toLowerCase(Locale.ROOT).contains("linux");

  private static final Comparator<Indexed> NEWEST_FIRST =
      Comparator.comparing(Indexed::created)
          .reversed()
          .thenComparing(entry -> entry.meta().resultId());

  private final DataDirectoryLock lock;
  private final Path results;
  private final Path staging;
  private final Path trash;
  private final ReentrantReadWriteLock files = new ReentrantReadWriteLock();

  /** resultId → 索引の 1 件。files の鍵で守る。 */
  private final Map<String, Indexed> byId = new HashMap<>();

  /** 持ち主 → 新しい順の索引。files の鍵で守る。 */
  private final Map<String, List<Indexed>> byOwner = new HashMap<>();

  private ResultStore(Path directory, DataDirectoryLock lock) {
    this.lock = lock;
    results = directory.resolve(StoreLayout.RESULTS.fileName());
    staging = directory.resolve(StoreLayout.STAGING.fileName());
    trash = directory.resolve(StoreLayout.TRASH.fileName());
  }

  /**
   * 鍵を取り、staging と trash に残った 1 件を消し、索引を作る。読めない・形が違う項目は記録に 1 行残して飛ばす。どのフォルダーでも UUID でない
   * 名前は読まず消さない。
   *
   * @throws ReviewerException data_locked
   */
  public static ResultStore open(Path directory) throws IOException {
    Files.createDirectories(directory);
    DataDirectoryLock lock = DataDirectoryLock.acquire(directory);
    try {
      var store = new ResultStore(directory, lock);
      for (Path path : List.of(store.results, store.staging, store.trash))
        Files.createDirectories(path);
      discardEntries(store.staging);
      discardEntries(store.trash);
      store.index();
      return store;
    } catch (IOException | RuntimeException failure) {
      lock.close();
      throw failure;
    }
  }

  /** 持ち主の結果（createdAt の新しい順）。 */
  public List<ResultMeta> history(OwnerId owner) {
    files.readLock().lock();
    try {
      List<Indexed> entries = byOwner.getOrDefault(owner.value(), List.of());
      List<ResultMeta> metas = new ArrayList<>(entries.size());
      for (Indexed entry : entries) metas.add(entry.meta());
      return metas;
    } finally {
      files.readLock().unlock();
    }
  }

  /** @throws ReviewerException result_not_found（無い・他の持ち主） */
  public ResultMeta find(OwnerId owner, ResultId id) {
    files.readLock().lock();
    try {
      Indexed entry = byId.get(id.value());
      if (entry == null || !entry.meta().owner().equals(owner.value()))
        throw new ReviewerException(ErrorCode.RESULT_NOT_FOUND);
      return entry.meta();
    } finally {
      files.readLock().unlock();
    }
  }

  /** @throws ReviewerException share_not_found */
  public ResultMeta findShared(ResultId id) {
    files.readLock().lock();
    try {
      Indexed entry = byId.get(id.value());
      if (entry == null) throw new ReviewerException(ErrorCode.SHARE_NOT_FOUND);
      return entry.meta();
    } finally {
      files.readLock().unlock();
    }
  }

  /**
   * 保存した gzip の文書の全体。
   *
   * @throws NoSuchFileException 探した後に消された
   */
  public byte[] readDocument(ResultMeta meta) throws IOException {
    files.readLock().lock();
    try {
      requireIndexed(meta);
      return Files.readAllBytes(file(meta, StoreLayout.DOCUMENT));
    } finally {
      files.readLock().unlock();
    }
  }

  /**
   * 元の牌譜（展開したバイト列）。
   *
   * @throws NoSuchFileException 探した後に消された
   */
  public byte[] readRecord(ResultMeta meta) throws IOException {
    files.readLock().lock();
    try {
      requireIndexed(meta);
      try (var in = new GZIPInputStream(Files.newInputStream(file(meta, StoreLayout.RECORD)))) {
        return in.readAllBytes();
      }
    } finally {
      files.readLock().unlock();
    }
  }

  /** 保存を始める（staging/&lt;id&gt;/）。 */
  public ResultDraft draft(ResultId id) throws IOException {
    return new ResultDraft(id, Files.createDirectory(staging.resolve(id.value())));
  }

  /** 文書・牌譜の後に meta.json を書いて同期し、results/ へ原子的に移して索引に足す。 */
  public ResultMeta commit(ResultDraft draft, ResultMeta meta) throws IOException {
    Path directory = draft.seal();
    writeFile(directory.resolve(StoreLayout.META.fileName()), Json.write(meta));
    var entry = new Indexed(meta, Instant.parse(meta.createdAt()));
    files.writeLock().lock();
    try {
      Files.move(directory, results.resolve(meta.resultId()), StandardCopyOption.ATOMIC_MOVE);
      draft.committed();
      add(entry);
    } finally {
      files.writeLock().unlock();
    }
    syncDirectory(results);
    return meta;
  }

  /** 索引から外し（すぐ見えなくなり共有も止まる）、trash へ移して消す。消せなかった物は掃除が続ける。 */
  public void delete(OwnerId owner, ResultId id) throws IOException {
    Path removed = trash.resolve(UUID.randomUUID().toString());
    files.writeLock().lock();
    try {
      Indexed entry = byId.get(id.value());
      if (entry == null || !entry.meta().owner().equals(owner.value()))
        throw new ReviewerException(ErrorCode.RESULT_NOT_FOUND);
      Files.move(results.resolve(id.value()), removed, StandardCopyOption.ATOMIC_MOVE);
      byId.remove(id.value());
      List<Indexed> entries = byOwner.get(owner.value());
      entries.remove(entry);
      if (entries.isEmpty()) byOwner.remove(owner.value());
    } finally {
      files.writeLock().unlock();
    }
    discard(removed);
  }

  /**
   * 文書を新しい一時ファイル（&lt;target&gt;.&lt;UUID&gt;.tmp）へ写して &lt;target&gt; と置き換える（デスクトップの書き出し）。
   * 選ばれた target の他に既にあるファイルは書き換えない。
   *
   * @param target 既にあるフォルダーの中の絶対パス
   * @throws ReviewerException result_not_found・invalid_request
   */
  public void export(OwnerId owner, ResultId id, Path target) throws IOException {
    ResultMeta meta = find(owner, id);
    Path folder = target.getParent();
    if (!target.isAbsolute() || folder == null || !Files.isDirectory(folder))
      throw new ReviewerException(
          ErrorCode.INVALID_REQUEST, "path must be an absolute file path in an existing folder.");
    Path temporary =
        target.resolveSibling(target.getFileName() + "." + UUID.randomUUID() + ".tmp");
    try {
      files.readLock().lock();
      try {
        requireIndexed(meta);
        Files.copy(file(meta, StoreLayout.DOCUMENT), temporary);
      } finally {
        files.readLock().unlock();
      }
      Files.move(
          temporary, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    } catch (IOException failure) {
      Files.deleteIfExists(temporary);
      throw failure;
    }
  }

  /** 消しきれなかった trash の 1 件を消す。 */
  public void sweep() {
    discardEntries(trash);
  }

  @Override
  public void close() {
    try {
      lock.close();
    } catch (IOException failure) {
      LOG.error("Could not release the data directory lock.", failure);
    }
  }

  private void index() throws IOException {
    try (DirectoryStream<Path> entries = Files.newDirectoryStream(results)) {
      for (Path entry : entries) {
        String name = entry.getFileName().toString();
        if (!ResultId.isValid(name)) continue;
        try {
          ResultMeta meta = readMeta(entry.resolve(StoreLayout.META.fileName()), name);
          add(new Indexed(meta, Instant.parse(meta.createdAt())));
        } catch (IOException | RuntimeException failure) {
          LOG.warn("Skipped the stored result {}: {}", name, failure.toString());
        }
      }
    }
  }

  private static ResultMeta readMeta(Path path, String name) throws IOException {
    if (Files.size(path) > META_LIMIT) throw new IOException("meta.json is too large.");
    ResultMeta meta =
        StrictRecords.read(
            Json.readObject(Files.readAllBytes(path), META_LIMIT), ResultMeta.class);
    if (meta.format() != ResultWriter.FORMAT)
      throw new IllegalArgumentException("Unsupported format " + meta.format() + ".");
    if (!meta.resultId().equals(name))
      throw new IllegalArgumentException("meta.json names another result.");
    if (meta.model() == null || meta.record() == null)
      throw new IllegalArgumentException("meta.json has no model or record.");
    if (!OwnerId.isValid(meta.owner())) throw new IllegalArgumentException("Invalid owner ID.");
    return meta;
  }

  /** files の書きの鍵の中か、起動の索引作りで呼ぶ。 */
  private void add(Indexed entry) {
    byId.put(entry.meta().resultId(), entry);
    List<Indexed> entries =
        byOwner.computeIfAbsent(entry.meta().owner(), owner -> new ArrayList<>());
    int position = Collections.binarySearch(entries, entry, NEWEST_FIRST);
    entries.add(position < 0 ? -position - 1 : position, entry);
  }

  private void requireIndexed(ResultMeta meta) throws NoSuchFileException {
    if (!byId.containsKey(meta.resultId())) throw new NoSuchFileException(meta.resultId());
  }

  private Path file(ResultMeta meta, StoreLayout file) {
    return results.resolve(meta.resultId()).resolve(file.fileName());
  }

  /**
   * staging か trash に残った 1 件（UUID の名前の、リンクでないフォルダー）を消す。保存先は利用者が選んだフォルダーのこともあるので、他の名前の物は
   * 読まず消さない。
   */
  private static void discardEntries(Path directory) {
    try (DirectoryStream<Path> entries =
        Files.newDirectoryStream(
            directory,
            entry ->
                ResultId.isValid(entry.getFileName().toString())
                    && Files.isDirectory(entry, LinkOption.NOFOLLOW_LINKS))) {
      for (Path entry : entries) discard(entry);
    } catch (IOException failure) {
      LOG.warn("Could not list {}: {}", directory, failure.toString());
    }
  }

  /** 新しいファイルへ書いてディスクへ同期する。 */
  static void writeFile(Path path, byte[] bytes) throws IOException {
    try (FileChannel channel =
        FileChannel.open(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      var buffer = ByteBuffer.wrap(bytes);
      while (buffer.hasRemaining()) channel.write(buffer);
      channel.force(true);
    }
  }

  /** 1 件の保存のファイルを消し、空になったフォルダーを消す。他のファイルがあればフォルダーは残す（保存が書いていない物は消さない）。 */
  static void discard(Path entry) {
    try {
      for (StoreLayout file : ENTRY_FILES) Files.deleteIfExists(entry.resolve(file.fileName()));
      Files.deleteIfExists(entry);
    } catch (IOException failure) {
      LOG.warn("Could not delete {}: {}", entry, failure.toString());
    }
  }

  private static void syncDirectory(Path directory) throws IOException {
    if (!SYNC_DIRECTORIES) return;
    try (FileChannel channel = FileChannel.open(directory, StandardOpenOption.READ)) {
      channel.force(true);
    }
  }

  /** 索引の 1 件と、並べ替えのために読んだ時刻。 */
  private record Indexed(ResultMeta meta, Instant created) {}
}
