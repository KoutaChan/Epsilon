package com.epsilon.reviewer.store;

import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.engine.format.ModelInfo;
import com.epsilon.reviewer.engine.load.RecordSummary;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/** 原子の保存、再起動の後の索引、持ち主の分け、削除ですぐ消える、壊れた meta を飛ばす、staging と trash の掃除（保存が書いた物だけ）、書き出し、鍵。 */
public class ResultStoreTest {
  private static final OwnerId OWNER = OwnerId.fromCookieKey("A".repeat(43));
  private static final OwnerId OTHER = OwnerId.fromCookieKey("B".repeat(43));
  private Path directory;

  @BeforeMethod
  public void create() {
    directory = TestFiles.directory("reviewer-store-");
  }

  @AfterMethod(alwaysRun = true)
  public void delete() {
    TestFiles.delete(directory);
  }

  @Test
  public void committedResultsAreIndexedNewestFirstAndSurviveARestart() throws IOException {
    ResultId older;
    ResultId newer;
    try (ResultStore store = ResultStore.open(directory)) {
      older = save(store, OWNER, "2026-11-04T00:00:00Z", bytes("plain record"));
      newer = save(store, OWNER, "2026-11-04T00:00:00.500Z", gzip("gzip record"));
      save(store, OTHER, "2026-11-04T00:00:01Z", bytes("other"));
      assertHistory(store, newer, older);
    }
    try (ResultStore store = ResultStore.open(directory)) {
      assertHistory(store, newer, older);
      ResultMeta meta = store.find(OWNER, older);
      Assert.assertEquals(text(store.readRecord(meta)), "plain record");
      Assert.assertEquals(text(store.readRecord(store.find(OWNER, newer))), "gzip record");
      Path record = directory.resolve(Path.of("results", older.value(), "record.gz"));
      byte[] stored = Files.readAllBytes(record);
      Assert.assertEquals(stored[0] & 0xff, 0x1f);
      Assert.assertEquals(stored[1] & 0xff, 0x8b);
      Assert.assertEquals(
          Files.readAllBytes(directory.resolve(Path.of("results", newer.value(), "record.gz"))),
          gzip("gzip record"));
      Assert.assertEquals(
          new String(gunzip(store.readDocument(meta)), StandardCharsets.UTF_8), "{\"format\":6}");
      Assert.assertEquals(store.findShared(older), meta);
      refused(() -> store.find(OTHER, older), ErrorCode.RESULT_NOT_FOUND);
      refused(() -> store.findShared(ResultId.random()), ErrorCode.SHARE_NOT_FOUND);
    }
  }

  @Test
  public void deletionHidesTheResultAtOnceAndLeavesNoFiles() throws IOException {
    try (ResultStore store = ResultStore.open(directory)) {
      ResultId id = save(store, OWNER, "2026-11-04T00:00:00Z", bytes("record"));
      ResultMeta meta = store.find(OWNER, id);
      refused(() -> store.delete(OTHER, id), ErrorCode.RESULT_NOT_FOUND);
      store.delete(OWNER, id);
      refused(() -> store.find(OWNER, id), ErrorCode.RESULT_NOT_FOUND);
      refused(() -> store.findShared(id), ErrorCode.SHARE_NOT_FOUND);
      Assert.expectThrows(NoSuchFileException.class, () -> store.readDocument(meta));
      Assert.assertTrue(store.history(OWNER).isEmpty());
      Assert.assertFalse(Files.exists(directory.resolve(Path.of("results", id.value()))));
      try (var trash = Files.list(directory.resolve("trash"))) {
        Assert.assertEquals(trash.count(), 0L);
      }
    }
  }

  @Test
  public void unreadableEntriesAreSkippedAndCleaningKeepsWhatTheStoreDidNotWrite()
      throws IOException {
    ResultId good;
    try (ResultStore store = ResultStore.open(directory)) {
      good = save(store, OWNER, "2026-11-04T00:00:00Z", bytes("record"));
    }
    Path broken =
        Files.createDirectories(directory.resolve(Path.of("results", ResultId.random().value())));
    Files.writeString(broken.resolve("meta.json"), "{not json");
    Path foreign = Files.createDirectories(directory.resolve(Path.of("results", "notes")));
    Path staged = entry("staging", "result.json.gz");
    Path trashed = entry("trash", "meta.json");
    Path mixed = entry("trash", "record.gz");
    Files.writeString(mixed.resolve("notes.txt"), "mine");
    Path userFolder = Files.createDirectories(directory.resolve(Path.of("staging", "partial")));
    Files.writeString(userFolder.resolve("result.json.gz"), "mine");
    Path userFile = directory.resolve(Path.of("trash", ResultId.random().value()));
    Files.writeString(userFile, "mine");
    try (ResultStore store = ResultStore.open(directory)) {
      Assert.assertEquals(store.history(OWNER).size(), 1);
      Assert.assertEquals(store.find(OWNER, good).resultId(), good.value());
      Assert.assertTrue(Files.exists(broken), "unreadable results stay on disk");
      Assert.assertTrue(Files.exists(foreign));
      Assert.assertFalse(Files.exists(staged));
      Assert.assertFalse(Files.exists(trashed));
      Assert.assertEquals(names(mixed), List.of("notes.txt"));
      Assert.assertEquals(names(userFolder), List.of("result.json.gz"));
      Assert.assertTrue(Files.isRegularFile(userFile));
      Path late = entry("trash", "result.json.gz");
      store.sweep();
      Assert.assertFalse(Files.exists(late));
      Assert.assertEquals(names(mixed), List.of("notes.txt"));
    }
  }

  @Test
  public void exportReplacesTheTargetAndDraftsAreDiscarded() throws IOException {
    try (ResultStore store = ResultStore.open(directory)) {
      ResultId id = save(store, OWNER, "2026-11-04T00:00:00Z", bytes("record"));
      Path target = directory.resolve("export.epsilon-reviewer.json.gz");
      Files.writeString(target, "old");
      Path userTemporary = directory.resolve("export.epsilon-reviewer.json.gz.tmp");
      Files.writeString(userTemporary, "mine");
      store.export(OWNER, id, target);
      Assert.assertEquals(
          Files.readAllBytes(target), store.readDocument(store.find(OWNER, id)));
      Assert.assertEquals(Files.readString(userTemporary), "mine");
      Assert.assertEquals(
          names(directory),
          List.of(
              "export.epsilon-reviewer.json.gz",
              "export.epsilon-reviewer.json.gz.tmp",
              "results",
              "reviewer.lock",
              "staging",
              "trash"));
      Path relative = Path.of("relative.json.gz");
      refused(() -> store.export(OWNER, id, relative), ErrorCode.INVALID_REQUEST);
      refused(() -> store.export(OTHER, id, target), ErrorCode.RESULT_NOT_FOUND);
      ResultId discarded = ResultId.random();
      try (ResultDraft draft = store.draft(discarded)) {
        draft.document().write(1);
      }
      Assert.assertFalse(Files.exists(directory.resolve(Path.of("staging", discarded.value()))));
      refused(() -> store.find(OWNER, discarded), ErrorCode.RESULT_NOT_FOUND);
    }
  }

  @Test
  public void aSecondReviewerCannotOpenTheSameDirectory() throws IOException {
    try (ResultStore store = ResultStore.open(directory)) {
      refused(() -> ResultStore.open(directory), ErrorCode.DATA_LOCKED);
    }
    ResultStore.open(directory).close();
  }

  private static ResultId save(ResultStore store, OwnerId owner, String createdAt, byte[] record)
      throws IOException {
    ResultId id = ResultId.random();
    try (ResultDraft draft = store.draft(id)) {
      try (var document = draft.document()) {
        document.write("{\"format\":6}".getBytes(StandardCharsets.UTF_8));
      }
      draft.record(record);
      store.commit(
          draft,
          new ResultMeta(
              6,
              id.value(),
              owner.value(),
              createdAt,
              new ModelInfo("fixture", "r1", "Fixture", "1"),
              new RecordSummary("tenhou", List.of("A", "B", "C", "D"), null, 1)));
    }
    return id;
  }

  /** 保存が作る形の 1 件（UUID の名前のフォルダーと保存のファイル 1 つ）。 */
  private Path entry(String folder, String file) throws IOException {
    Path entry =
        Files.createDirectories(directory.resolve(Path.of(folder, ResultId.random().value())));
    Files.writeString(entry.resolve(file), "partial");
    return entry;
  }

  private static List<String> names(Path folder) throws IOException {
    try (var entries = Files.list(folder)) {
      return entries.map(path -> path.getFileName().toString()).sorted().toList();
    }
  }

  private static void assertHistory(ResultStore store, ResultId... expected) {
    Assert.assertEquals(
        store.history(OWNER).stream().map(ResultMeta::resultId).toList(),
        List.of(expected).stream().map(ResultId::value).toList());
  }

  private static void refused(Assert.ThrowingRunnable action, ErrorCode code) {
    ReviewerException failure = Assert.expectThrows(ReviewerException.class, action);
    Assert.assertEquals(failure.code(), code);
  }

  private static byte[] gzip(String text) throws IOException {
    var bytes = new ByteArrayOutputStream();
    try (var out = new GZIPOutputStream(bytes)) {
      out.write(text.getBytes(StandardCharsets.UTF_8));
    }
    return bytes.toByteArray();
  }

  private static byte[] bytes(String text) {
    return text.getBytes(StandardCharsets.UTF_8);
  }

  private static String text(byte[] bytes) {
    return new String(bytes, StandardCharsets.UTF_8);
  }

  private static byte[] gunzip(byte[] gzip) throws IOException {
    try (var in = new GZIPInputStream(new ByteArrayInputStream(gzip))) {
      return in.readAllBytes();
    }
  }
}
