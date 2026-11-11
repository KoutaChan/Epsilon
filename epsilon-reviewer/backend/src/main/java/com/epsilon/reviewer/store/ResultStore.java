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
 * 解析結果を保存する。1 件ごとに results/&lt;resultId&gt;/ に文書・牌譜・meta.json を置く。インデックスはメモリに持ち、起動時に
 * 各結果の meta.json を 1 回ずつ読んで作り、保存と削除のたびに更新する。ファイルの読み込みと、削除・確定は、読み書きのロックで分ける。
 */
public final class ResultStore implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(ResultStore.class);

  /** meta.json のサイズの上限（バイト）。 */
  private static final int META_LIMIT = 64 * 1024;

  /** 保存で 1 件のフォルダーに書くファイル。消すときもこれだけを消す。 */
  private static final List<StoreLayout> ENTRY_FILES =
      List.of(StoreLayout.DOCUMENT, StoreLayout.RECORD, StoreLayout.META);

  /** ディレクトリを同期するのは Linux だけ（Windows ではディレクトリを開けない）。 */
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

  /** resultId からインデックスの 1 件を引く表。{@code files} のロックで守る。 */
  private final Map<String, Indexed> byId = new HashMap<>();

  /** オーナーから、新しい順に並べたインデックスを引く表。{@code files} のロックで守る。 */
  private final Map<String, List<Indexed>> byOwner = new HashMap<>();

  private ResultStore(Path directory, DataDirectoryLock lock) {
    this.lock = lock;
    results = directory.resolve(StoreLayout.RESULTS.fileName());
    staging = directory.resolve(StoreLayout.STAGING.fileName());
    trash = directory.resolve(StoreLayout.TRASH.fileName());
  }

  /**
   * 保存先を開く。ロックを取り、staging と trash に残っている結果を消してから、インデックスを作る。読めない結果や形の違う結果は、
   * 警告のログを 1 行書いて飛ばす。どのフォルダーでも、名前が UUID でないものは読まず、消さない。
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

  /** オーナーの結果を、createdAt の新しい順に返す。時刻が同じなら resultId の昇順に並べる。 */
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

  /** @throws ReviewerException result_not_found（結果が無い、またはオーナーが違う） */
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
   * 保存した文書を、gzip のまま全て読む。
   *
   * @throws NoSuchFileException 結果を見つけた後に消された
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
   * 元の牌譜を、展開したバイト列で読む。
   *
   * @throws NoSuchFileException 結果を見つけた後に消された
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

  /** 保存を始める。書いている途中のファイルは staging/&lt;id&gt;/ に置く。 */
  public ResultDraft draft(ResultId id) throws IOException {
    return new ResultDraft(id, Files.createDirectory(staging.resolve(id.value())));
  }

  /**
   * 1 件を確定する。文書を閉じ、meta.json を書いて同期してから、フォルダーを results/ へアトミックに移してインデックスに足す。文書と牌譜を
   * 書き終えていなければ IllegalStateException を投げる。meta.json が起動時にインデックスへ読める大きさを超えるなら、何も確定せずに
   * IOException を投げる。
   */
  public ResultMeta commit(ResultDraft draft, ResultMeta meta) throws IOException {
    Path directory = draft.seal();
    byte[] json = Json.write(meta);
    if (json.length > META_LIMIT) throw new IOException("meta.json is too large.");
    writeFile(directory.resolve(StoreLayout.META.fileName()), json);
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

  /**
   * 結果をインデックスから外し、trash へ移して消す。外した時点で履歴から見えなくなり、共有も止まる。消しきれなかったものは
   * {@link #sweep} が消す。結果が無いか、オーナーが違えば result_not_found で断る。
   */
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
   * デスクトップ版で、文書をファイルに書き出す。新しい一時ファイル（&lt;target&gt;.&lt;UUID&gt;.tmp）へコピーしてから
   * &lt;target&gt; と置き換える。書き換えるのは選ばれたファイルだけで、ほかの既にあるファイルには触れない。
   *
   * @param target 書き出すファイルの絶対パス。フォルダーは既にあること
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

  /** trash に残った、消しきれなかった結果を消す。 */
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

  /** {@code files} の書き込みのロックを取った中か、起動時にインデックスを作るときに呼ぶ。 */
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
   * staging か trash に残った結果（名前が UUID で、リンクでないフォルダー）を消す。保存先はユーザーが選んだフォルダーのこともあるので、
   * ほかの名前のものは読まず、消さない。
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

  /** 新しいファイルを作って書き、ディスクに同期する。 */
  static void writeFile(Path path, byte[] bytes) throws IOException {
    try (FileChannel channel =
        FileChannel.open(path, StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE)) {
      var buffer = ByteBuffer.wrap(bytes);
      while (buffer.hasRemaining()) channel.write(buffer);
      channel.force(true);
    }
  }

  /**
   * 1 件のフォルダーから保存で書いたファイルを消し、空になったフォルダーも消す。保存で書いていないファイルは消さないので、それが残っていれば
   * フォルダーも残る。消せなくても警告のログを書くだけで、例外は投げない。
   */
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

  /** インデックスの 1 件。並べ替えには、createdAt を読んで Instant にした時刻を使う。 */
  private record Indexed(ResultMeta meta, Instant created) {}
}
