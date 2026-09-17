package com.epsilon.reviewer;

import com.epsilon.reviewer.dto.*;
import com.google.gson.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.zip.GZIPInputStream;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** 解析結果の保存・復元・共有・削除、所有者ごとのアクセス制限、保存失敗時の一貫性を検証する。 */
public final class ResultStoreTest {
  @Test
  public void completedResultSurvivesRestartAndCannotBeReadOrDeletedByAnotherOwner()
      throws Exception {
    Path data = Files.createTempDirectory("reviewer-store-test-");
    try {
      var store = new ResultStore(data);
      String owner = "a".repeat(64), other = "b".repeat(64);
      String id = UUID.randomUUID().toString();
      ReviewResult result = result(id);
      var source =
          new RecordFetcher.SourceRecord(
              "original record".getBytes(StandardCharsets.UTF_8), "game.mjson", "mjai");
      store.save(owner, result, source, false);
      var restarted = new ResultStore(data);
      Assert.assertEquals(restarted.history(owner).size(), 1);
      Assert.assertEquals(restarted.history(other).size(), 0);
      Assert.assertEquals(
          new Gson().fromJson(readJsonResult(restarted, owner, id), ReviewResult.class), result);
      Assert.assertEquals(restarted.source(owner, id).bytes(), source.bytes());
      byte[] stored = Files.readAllBytes(data.resolve(owner).resolve(id).resolve("result.json.gz"));
      Files.delete(data.resolve(owner).resolve(id).resolve("source.bin"));
      Assert.assertEquals(readCompressedResult(new ResultStore(data), owner, id), stored);
      Assert.expectThrows(ApiException.class, () -> restarted.openResult(other, id));
      Assert.expectThrows(ApiException.class, () -> restarted.delete(other, id));
      Assert.expectThrows(ApiException.class, () -> restarted.openResult(owner, "../" + id));
      restarted.delete(owner, id);
      Assert.assertEquals(restarted.history(owner).size(), 0);
      Assert.expectThrows(ApiException.class, () -> restarted.source(owner, id));
    } finally {
      delete(data);
    }
  }

  @Test
  public void desktopResultsStayOutOfWebHistoryAndExpireWithoutDeletingWebResults()
      throws Exception {
    Path data = Files.createTempDirectory("reviewer-expiry-test-");
    try {
      var store = new ResultStore(data);
      String owner = "a".repeat(64),
          desktop = UUID.randomUUID().toString(),
          web = UUID.randomUUID().toString();
      var source = new RecordFetcher.SourceRecord(new byte[] {1}, "game.mjson", "mjai");
      store.save(owner, result(desktop), source, true);
      store.save(owner, result(web), source, false);
      Assert.assertEquals(store.history(owner).size(), 1);
      Assert.assertEquals(store.history(owner).get(0).resultId(), web);
      store.cleanTemporary(java.time.Instant.now().plus(java.time.Duration.ofHours(25)));
      Assert.expectThrows(ApiException.class, () -> store.openResult(owner, desktop));
      Assert.assertTrue(readCompressedResult(store, owner, web).length > 0);
    } finally {
      delete(data);
    }
  }

  @Test
  public void webResultsAreSharedOnSaveAndRestartWithoutAdditionalStoredFiles() throws Exception {
    Path data = Files.createTempDirectory("reviewer-sharing-store-test-");
    try {
      var store = new ResultStore(data);
      String owner = "a".repeat(64), other = "b".repeat(64);
      String id = UUID.randomUUID().toString();
      store.save(
          owner,
          result(id),
          new RecordFetcher.SourceRecord(new byte[] {1}, "game.mjson", "mjai"),
          false);
      byte[] stored = readCompressedResult(store, owner, id);
      Assert.assertEquals(store.requireShareableResultId(owner, id), id);
      Assert.assertEquals(readSharedResult(store, id), stored);
      var denied =
          Assert.expectThrows(ApiException.class, () -> store.requireShareableResultId(other, id));
      Assert.assertEquals(denied.code(), "result_not_found");
      try (var files = Files.list(data.resolve(owner).resolve(id))) {
        Assert.assertEquals(
            files.map(path -> path.getFileName().toString()).sorted().toList(),
            List.of("result.json.gz", "source.bin", "source.json", "summary.json"));
      }
      var restarted = new ResultStore(data);
      Assert.assertEquals(readSharedResult(restarted, id), stored);
      restarted.delete(owner, id);
      Assert.assertEquals(
          Assert.expectThrows(ApiException.class, () -> restarted.openSharedResult(id)).code(),
          "share_not_found");
      Assert.assertEquals(
          Assert.expectThrows(ApiException.class, () -> new ResultStore(data).openSharedResult(id))
              .code(),
          "share_not_found");
    } finally {
      delete(data);
    }
  }

  @Test
  public void desktopResultsAndInvalidSharedIdsNeverResolve() throws Exception {
    Path data = Files.createTempDirectory("reviewer-sharing-scope-test-");
    try {
      var store = new ResultStore(data);
      String owner = "a".repeat(64), id = UUID.randomUUID().toString();
      store.save(
          owner,
          result(id),
          new RecordFetcher.SourceRecord(new byte[] {1}, "game.mjson", "mjai"),
          true);
      var unavailable =
          Assert.expectThrows(ApiException.class, () -> store.requireShareableResultId(owner, id));
      Assert.assertEquals(unavailable.status(), 409);
      Assert.assertEquals(unavailable.code(), "share_unavailable");
      for (String sharedId : List.of(id, "../" + id, "", UUID.randomUUID().toString())) {
        var missing =
            Assert.expectThrows(ApiException.class, () -> store.openSharedResult(sharedId));
        Assert.assertEquals(missing.status(), 404);
        Assert.assertEquals(missing.code(), "share_not_found");
      }
      Assert.assertEquals(
          Assert.expectThrows(ApiException.class, () -> new ResultStore(data).openSharedResult(id))
              .code(),
          "share_not_found");
    } finally {
      delete(data);
    }
  }

  @Test
  public void failedCommitDoesNotReplaceAnExistingResultOrRegisterPartialHistory()
      throws Exception {
    Path data = Files.createTempDirectory("reviewer-atomic-test-");
    try {
      var store = new ResultStore(data);
      String owner = "a".repeat(64), id = UUID.randomUUID().toString();
      var original = result(id);
      var source = new RecordFetcher.SourceRecord(new byte[] {1}, "game.mjson", "mjai");
      store.save(owner, original, source, false);
      var duplicate =
          new ReviewResult(
              ReviewResult.CURRENT_FORMAT_VERSION,
              id,
              "2030-01-01T00:00:00Z",
              original.metadata(),
              original.model(),
              original.rounds());
      Assert.expectThrows(
          java.io.IOException.class, () -> store.save(owner, duplicate, source, false));
      Assert.assertEquals(store.history(owner).size(), 1);
      Assert.assertEquals(
          new Gson().fromJson(readJsonResult(store, owner, id), ReviewResult.class), original);
      try (var entries = Files.list(data.resolve(owner))) {
        Assert.assertEquals(entries.count(), 1L);
      }
    } finally {
      delete(data);
    }
  }

  @DataProvider(name = "unsupportedResultVersions")
  public Object[][] unsupportedResultVersions() {
    return new Object[][] {{1}, {2}, {3}, {ReviewResult.CURRENT_FORMAT_VERSION + 1}};
  }

  @Test(dataProvider = "unsupportedResultVersions")
  public void unsupportedVersionsAreRejectedBeforeWritingAnyResult(int formatVersion)
      throws Exception {
    Path data = Files.createTempDirectory("reviewer-format-test-");
    try {
      var store = new ResultStore(data);
      String owner = "a".repeat(64), id = UUID.randomUUID().toString();
      var current = result(id);
      var unsupported =
          new ReviewResult(
              formatVersion,
              id,
              current.createdAt(),
              current.metadata(),
              current.model(),
              current.rounds());
      var source = new RecordFetcher.SourceRecord(new byte[] {1}, "game.mjson", "mjai");
      ApiException error =
          Assert.expectThrows(
              ApiException.class, () -> store.save(owner, unsupported, source, false));
      Assert.assertEquals(error.status(), 422);
      Assert.assertEquals(error.code(), "invalid_result");
      Assert.assertFalse(Files.exists(data.resolve(owner).resolve(id)));
      Assert.assertTrue(store.history(owner).isEmpty());
    } finally {
      delete(data);
    }
  }

  @Test
  public void legacyWebAndDesktopResultsAreHiddenButTheirSourcesAndFilesRemainUntouched()
      throws Exception {
    Path data = Files.createTempDirectory("reviewer-legacy-result-");
    try {
      var store = new ResultStore(data);
      String owner = "a".repeat(64);
      List<String> ids = new ArrayList<>();
      var source = new RecordFetcher.SourceRecord(new byte[] {1, 2}, "game.mjson", "mjai");
      for (boolean desktop : new boolean[] {false, true}) {
        String id = UUID.randomUUID().toString();
        ids.add(id);
        store.save(owner, result(id), source, desktop);
        JsonObject legacy =
            JsonParser.parseString(readJsonResult(store, owner, id)).getAsJsonObject();
        legacy.addProperty("formatVersion", 3);
        Path directory = data.resolve(owner).resolve(id);
        Files.writeString(directory.resolve("result.json"), legacy.toString());
        Files.delete(directory.resolve("result.json.gz"));
      }
      var restarted = new ResultStore(data);
      Assert.assertTrue(restarted.history(owner).isEmpty());
      restarted.cleanTemporary(java.time.Instant.now().plus(java.time.Duration.ofHours(25)));
      for (String id : ids) {
        Path directory = data.resolve(owner).resolve(id);
        Assert.assertTrue(Files.isRegularFile(directory.resolve("result.json")));
        Assert.assertFalse(Files.exists(directory.resolve("result.json.gz")));
        Assert.assertEquals(restarted.source(owner, id).bytes(), source.bytes());
        ApiException owned =
            Assert.expectThrows(ApiException.class, () -> restarted.openResult(owner, id));
        Assert.assertEquals(owned.status(), 422);
        Assert.assertEquals(owned.code(), "invalid_result");
        Assert.assertEquals(
            Assert.expectThrows(
                    ApiException.class, () -> restarted.requireShareableResultId(owner, id))
                .code(),
            "invalid_result");
        Assert.assertEquals(
            Assert.expectThrows(ApiException.class, () -> restarted.openSharedResult(id)).code(),
            "share_not_found");
        Assert.assertEquals(
            JsonParser.parseString(Files.readString(directory.resolve("result.json")))
                .getAsJsonObject()
                .get("formatVersion")
                .getAsInt(),
            3);
      }
    } finally {
      delete(data);
    }
  }

  @Test
  public void openedCompressedResultRemainsReadableAfterDeletionWithoutKeepingTheStoreLocked()
      throws Exception {
    Path data = Files.createTempDirectory("reviewer-open-result-");
    try {
      var store = new ResultStore(data);
      String owner = "a".repeat(64), id = UUID.randomUUID().toString();
      store.save(
          owner,
          result(id),
          new RecordFetcher.SourceRecord(new byte[] {1}, "game.mjson", "mjai"),
          false);
      byte[] expected = readCompressedResult(store, owner, id);
      try (var opened = store.openResult(owner, id)) {
        Assert.assertEquals(opened.size(), expected.length);
        store.delete(owner, id);
        Assert.assertTrue(store.history(owner).isEmpty());
        Assert.assertEquals(
            Assert.expectThrows(ApiException.class, () -> store.openResult(owner, id)).code(),
            "result_not_found");
        Assert.assertEquals(
            Assert.expectThrows(ApiException.class, () -> store.openSharedResult(id)).code(),
            "share_not_found");
        store.cleanTemporary(java.time.Instant.now());
        Assert.assertTrue(Files.exists(data.resolve(owner).resolve(id).resolve(".deleted")));
        Assert.assertEquals(opened.input().readAllBytes(), expected);
        opened.input().close();
        opened.close();
        opened.close();
      }
      Assert.assertFalse(Files.exists(data.resolve(owner).resolve(id)));
      Assert.assertEquals(
          Assert.expectThrows(ApiException.class, () -> store.openSharedResult(id)).code(),
          "share_not_found");
    } finally {
      delete(data);
    }
  }

  @Test
  public void aPersistedDeletionMarkerStaysPrivateAcrossRestartAndIsCleanedUp() throws Exception {
    Path data = Files.createTempDirectory("reviewer-deletion-restart-");
    try {
      var store = new ResultStore(data);
      String owner = "a".repeat(64), id = UUID.randomUUID().toString();
      store.save(
          owner,
          result(id),
          new RecordFetcher.SourceRecord(new byte[] {1}, "game.mjson", "mjai"),
          false);
      Path directory = data.resolve(owner).resolve(id);
      Files.write(directory.resolve(".deleted"), new byte[0]);
      var restarted = new ResultStore(data);
      Assert.assertTrue(restarted.history(owner).isEmpty());
      Assert.assertEquals(
          Assert.expectThrows(ApiException.class, () -> restarted.openResult(owner, id)).code(),
          "result_not_found");
      Assert.assertEquals(
          Assert.expectThrows(ApiException.class, () -> restarted.source(owner, id)).code(),
          "result_not_found");
      Assert.assertEquals(
          Assert.expectThrows(ApiException.class, () -> restarted.openSharedResult(id)).code(),
          "share_not_found");
      restarted.cleanTemporary(java.time.Instant.now());
      Assert.assertFalse(Files.exists(directory));
    } finally {
      delete(data);
    }
  }

  @Test
  public void cleanupRemovesOrphanedStagingButPreservesAnActivelyStreamingSave() throws Exception {
    Path data = Files.createTempDirectory("reviewer-active-save-");
    var writing = new CountDownLatch(1);
    var resume = new CountDownLatch(1);
    try (var executor = Executors.newSingleThreadExecutor()) {
      var store = new ResultStore(data);
      String owner = "a".repeat(64), id = UUID.randomUUID().toString();
      List<String> names =
          new AbstractList<>() {
            @Override
            public String get(int index) {
              if (index == 0) {
                writing.countDown();
                try {
                  if (!resume.await(5, TimeUnit.SECONDS))
                    throw new AssertionError("The streaming save was not resumed.");
                } catch (InterruptedException failure) {
                  Thread.currentThread().interrupt();
                  throw new AssertionError("The streaming save was interrupted.", failure);
                }
              }
              return List.of("A", "B", "C", "D").get(index);
            }

            @Override
            public int size() {
              return 4;
            }
          };
      var current = result(id);
      var metadata = current.metadata();
      var result =
          new ReviewResult(
              current.formatVersion(),
              id,
              current.createdAt(),
              new RecordMetadata(
                  metadata.source(),
                  metadata.fileName(),
                  names,
                  metadata.finalScores(),
                  metadata.finalScoresSource(),
                  metadata.roundCount()),
              current.model(),
              current.rounds());
      Path orphan = Files.createDirectories(data.resolve(owner).resolve(".staging-orphan"));
      Files.writeString(orphan.resolve("partial.json.gz"), "orphaned write");
      var saved =
          executor.submit(
              () -> {
                store.save(
                    owner,
                    result,
                    new RecordFetcher.SourceRecord(new byte[] {1}, "game.mjson", "mjai"),
                    false);
                return null;
              });
      try {
        Assert.assertTrue(writing.await(5, TimeUnit.SECONDS), "The JSON writer did not start.");
        store.cleanTemporary(java.time.Instant.now());
        Assert.assertFalse(Files.exists(orphan));
      } finally {
        resume.countDown();
      }
      saved.get(5, TimeUnit.SECONDS);
      Assert.assertEquals(store.history(owner).size(), 1);
      Assert.assertEquals(
          new Gson()
              .fromJson(readJsonResult(store, owner, id), ReviewResult.class)
              .metadata()
              .names(),
          List.of("A", "B", "C", "D"));
      Assert.assertEquals(readSharedResult(store, id), readCompressedResult(store, owner, id));
    } finally {
      resume.countDown();
      delete(data);
    }
  }

  private static byte[] readCompressedResult(ResultStore store, String owner, String id)
      throws Exception {
    try (var content = store.openResult(owner, id)) {
      byte[] bytes = content.input().readAllBytes();
      Assert.assertEquals(bytes.length, content.size());
      Assert.assertEquals(bytes[0] & 255, 0x1f);
      Assert.assertEquals(bytes[1] & 255, 0x8b);
      return bytes;
    }
  }

  private static byte[] readSharedResult(ResultStore store, String id) throws Exception {
    try (var content = store.openSharedResult(id)) {
      return content.input().readAllBytes();
    }
  }

  private static String readJsonResult(ResultStore store, String owner, String id)
      throws Exception {
    try (var content = store.openResult(owner, id);
        var gzip = new GZIPInputStream(content.input())) {
      return new String(gzip.readAllBytes(), StandardCharsets.UTF_8);
    }
  }

  private static ReviewResult result(String id) {
    return new ReviewResult(
        ReviewResult.CURRENT_FORMAT_VERSION,
        id,
        "2026-09-15T01:00:00Z",
        new RecordMetadata(
            "mjai",
            "record.mjai",
            List.of("A", "B", "C", "D"),
            List.of(30000, 27000, 23000, 20000),
            "mjai.end_game",
            1),
        new ModelInfo("epsilon", "fixed", "Epsilon", "1", "epsilon"),
        List.of());
  }

  private static void delete(Path directory) throws Exception {
    try (var paths = Files.walk(directory)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    }
  }
}
