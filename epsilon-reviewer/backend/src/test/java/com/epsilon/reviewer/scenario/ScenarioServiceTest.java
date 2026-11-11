package com.epsilon.reviewer.scenario;

import com.epsilon.reviewer.FixtureFamily;
import com.epsilon.reviewer.Reviewer;
import com.epsilon.reviewer.ReviewerConfig;
import com.epsilon.reviewer.StoredResults;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.TestReviewer;
import com.epsilon.reviewer.engine.Format6;
import com.epsilon.reviewer.engine.TestEdits;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.model.ModelCatalog;
import com.epsilon.reviewer.model.ModelFamilies;
import com.epsilon.reviewer.model.ModelRegistry;
import com.epsilon.reviewer.store.ResultId;
import com.epsilon.reviewer.store.ResultMeta;
import com.epsilon.reviewer.store.ResultStore;
import com.epsilon.reviewer.store.StoreLayout;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.zip.GZIPInputStream;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * もしもの受付を確かめる。評価は 1 つずつで、待てるのは 4 つまで、待つのは 3 秒まで。ほかに、結果に書いた revision と今の
 * revision のどちらで答えるか、削除した結果のキャッシュを捨てること、読めなくなった牌譜と文書、文書のインデックスに無い判断と
 * 候補の違う判断、カタログに無いモデルの結果も確かめる。
 */
public class ScenarioServiceTest {
  private Path directory;
  private ResultStore store;
  private ModelCatalog catalog;
  private ModelRegistry models;
  private ResultId fixture;
  private ResultId held;
  private ResultId deletable;
  private ResultId unreplayable;
  private ResultId unreadable;
  private ResultId changedCandidates;
  private ModelFamilies families;
  private ScenarioQuery query;

  @BeforeClass
  public void prepare() throws IOException {
    directory = TestFiles.directory("reviewer-scenario-");
    ReviewerConfig config = TestReviewer.config(directory);
    FixtureFamily.release = new CountDownLatch(0);
    try (Reviewer reviewer = Reviewer.open(config)) {
      fixture = TestReviewer.analyze(reviewer, TestReviewer.OWNER, TestReviewer.FIXTURE);
      held = TestReviewer.analyze(reviewer, TestReviewer.OWNER, TestReviewer.HELD);
    }
    Format6.Located decision =
        firstChoice(StoredResults.read(config.data(), fixture, StoreLayout.DOCUMENT));
    query =
        new ScenarioQuery(
            decision.round(),
            decision.step().eventIndex(),
            decision.step().actor(),
            TestEdits.NONE,
            null);
    deletable = StoredResults.copy(config.data(), fixture, record -> record, document -> document);
    unreplayable =
        StoredResults.copy(
            config.data(), fixture, ScenarioServiceTest::wrongDealer, document -> document);
    unreadable =
        StoredResults.copy(
            config.data(),
            fixture,
            record -> record,
            ignored -> "not a gzip document".getBytes(StandardCharsets.UTF_8));
    changedCandidates =
        StoredResults.copy(
            config.data(), fixture, record -> record, document -> dropCandidate(document, query));
    families = config.families();
    catalog = ModelCatalog.load(config.models(), families);
    store = ResultStore.open(config.data());
    models = new ModelRegistry(catalog, families, 2, Duration.ofMinutes(10));
  }

  @AfterClass(alwaysRun = true)
  public void close() {
    FixtureFamily.release.countDown();
    if (models != null) models.close();
    if (store != null) store.close();
    TestFiles.delete(directory);
  }

  @Test
  public void oneEvaluationRunsFourWaitAndTheRestAreBusy() throws Exception {
    var scenarios = new ScenarioService(catalog, models, store);
    ResultMeta meta = store.find(TestReviewer.OWNER, held);
    FixtureFamily.entered = new CountDownLatch(1);
    FixtureFamily.release = new CountDownLatch(1);
    ExecutorService threads = Executors.newFixedThreadPool(ScenarioService.ADMITTED);
    try {
      List<Future<?>> admitted = new ArrayList<>();
      admitted.add(threads.submit(() -> scenarios.evaluate(meta, query)));
      Assert.assertTrue(FixtureFamily.entered.await(30, TimeUnit.SECONDS));
      for (int i = 1; i < ScenarioService.ADMITTED; i++)
        admitted.add(threads.submit(() -> scenarios.evaluate(meta, query)));
      Thread.sleep(200);
      long started = System.nanoTime();
      busy(() -> scenarios.evaluate(meta, query));
      Assert.assertTrue(
          System.nanoTime() - started < Duration.ofSeconds(1).toNanos(),
          "the sixth is refused at once");
      FixtureFamily.release.countDown();
      for (Future<?> future : admitted) future.get(30, TimeUnit.SECONDS);
      FixtureFamily.release = new CountDownLatch(1);
      FixtureFamily.entered = new CountDownLatch(1);
      Future<?> running = threads.submit(() -> scenarios.evaluate(meta, query));
      Assert.assertTrue(FixtureFamily.entered.await(30, TimeUnit.SECONDS));
      started = System.nanoTime();
      busy(() -> scenarios.evaluate(meta, query));
      Assert.assertTrue(
          System.nanoTime() - started >= ScenarioService.WAIT.toNanos(), "the second waits 3 s");
      FixtureFamily.release.countDown();
      running.get(30, TimeUnit.SECONDS);
    } finally {
      FixtureFamily.release.countDown();
      threads.shutdownNow();
    }
  }

  @Test
  public void answersTheStoredDecisionAndForgetsDeletedResults() throws Exception {
    var scenarios = new ScenarioService(catalog, models, store);
    ResultMeta meta = store.find(TestReviewer.OWNER, deletable);
    Format6.Located decision = firstChoice(store.readDocument(meta));
    Assert.assertEquals(
        Format6.candidateSet(scenarios.evaluate(meta, query)), decision.step().candidateSet());
    store.delete(TestReviewer.OWNER, deletable);
    Assert.assertNotNull(scenarios.evaluate(meta, query), "the cached record still answers");
    scenarios.forget(deletable);
    Assert.assertEquals(
        Assert.expectThrows(ReviewerException.class, () -> scenarios.evaluate(meta, query)).code(),
        ErrorCode.RESULT_NOT_FOUND);
  }

  @Test
  public void anUnavailableRevisionIsEvaluatedWithTheModelsCurrentRevision() throws IOException {
    ResultMeta meta = store.find(TestReviewer.OWNER, fixture);
    // 結果に書いた r1 はチェックポイントが無く使えないので、今の r2 で答える。
    try (var renewed = new Scenarios(catalog("renewed", "missing", "checkpoint"))) {
      Assert.assertNotNull(renewed.service.evaluate(meta, query));
    }
    // 結果に書いた r1 が使えれば、今の r2 が使えなくても r1 で答える。
    try (var listed = new Scenarios(catalog("listed", "checkpoint", "missing"))) {
      Assert.assertNotNull(listed.service.evaluate(meta, query));
    }
    // どちらも使えなければ断る。
    try (var withdrawn = new Scenarios(catalog("withdrawn", "missing", "missing"))) {
      Assert.assertEquals(
          Assert.expectThrows(ReviewerException.class, () -> withdrawn.service.evaluate(meta, query))
              .code(),
          ErrorCode.MODEL_UNAVAILABLE);
    }
  }

  @Test
  public void aStoredRecordThatNoLongerReplaysIsAnInvalidRecord() throws IOException {
    var scenarios = new ScenarioService(catalog, models, store);
    ResultMeta meta = store.find(TestReviewer.OWNER, unreplayable);
    ReviewerException failure =
        Assert.expectThrows(ReviewerException.class, () -> scenarios.evaluate(meta, query));
    Assert.assertEquals(failure.code(), ErrorCode.INVALID_RECORD);
    Assert.assertTrue(
        failure.getMessage().contains("The stored record can no longer be read"),
        failure.getMessage());
  }

  @Test
  public void anUnreadableDocumentIsAnInvalidRecord() throws IOException {
    var scenarios = new ScenarioService(catalog, models, store);
    ResultMeta meta = store.find(TestReviewer.OWNER, unreadable);
    ReviewerException failure =
        Assert.expectThrows(ReviewerException.class, () -> scenarios.evaluate(meta, query));
    Assert.assertEquals(failure.code(), ErrorCode.INVALID_RECORD);
    Assert.assertTrue(
        failure.getMessage().contains("The stored result can no longer be read"),
        failure.getMessage());
  }

  @Test
  public void decisionsOutsideTheDocumentIndexOrWithOtherCandidatesAreNotFound()
      throws IOException {
    var scenarios = new ScenarioService(catalog, models, store);
    ResultMeta meta = store.find(TestReviewer.OWNER, fixture);
    var elsewhere =
        new ScenarioQuery(
            query.round(), query.eventIndex() + 10_000, query.seat(), TestEdits.NONE, null);
    Assert.assertEquals(
        Assert.expectThrows(ReviewerException.class, () -> scenarios.evaluate(meta, elsewhere))
            .code(),
        ErrorCode.DECISION_NOT_FOUND);
    ResultMeta changed = store.find(TestReviewer.OWNER, changedCandidates);
    Assert.assertEquals(
        Assert.expectThrows(ReviewerException.class, () -> scenarios.evaluate(changed, query))
            .code(),
        ErrorCode.DECISION_NOT_FOUND);
  }

  @Test
  public void resultsOfModelsMissingFromTheCatalogAreRefused() throws IOException {
    var scenarios = new ScenarioService(ModelCatalog.empty(), models, store);
    ResultMeta meta = store.find(TestReviewer.OWNER, held);
    Assert.assertEquals(
        Assert.expectThrows(ReviewerException.class, () -> scenarios.evaluate(meta, query)).code(),
        ErrorCode.MODEL_UNAVAILABLE);
  }

  /** fixture の r1 と r2 を載せたカタログを作る（今の revision は r2）。checkpoint に missing を渡した revision は使えない。 */
  private ModelCatalog catalog(String name, String first, String second) throws IOException {
    Path file = directory.resolve(name + ".json");
    Files.writeString(
        file,
        """
        {"models":[
        {"modelId":"fixture","revision":"r1","displayName":"Fixture","version":"1",
         "series":"%1$s","variant":"TENHOU_4P","checkpoint":"%2$s"},
        {"modelId":"fixture","revision":"r2","displayName":"Fixture","version":"2",
         "series":"%1$s","variant":"TENHOU_4P","checkpoint":"%3$s"}]}
        """
            .formatted(FixtureFamily.SERIES.id(), first, second));
    return ModelCatalog.load(file, families);
  }

  /** 別のカタログを使うもしものサービス。 */
  private final class Scenarios implements AutoCloseable {
    final ModelRegistry registry;
    final ScenarioService service;

    Scenarios(ModelCatalog catalog) {
      registry = new ModelRegistry(catalog, families, 1, Duration.ofMinutes(10));
      service = new ScenarioService(catalog, registry, store);
    }

    @Override
    public void close() {
      registry.close();
    }
  }

  /** 保存した天鳳の牌譜（gzip）の、1 局目の INIT の親を替えたもの。読み直せない。 */
  private static byte[] wrongDealer(byte[] gzip) {
    String record = new String(StoredResults.gunzip(gzip), StandardCharsets.UTF_8);
    String dealer = " oya=\"0\" hai0=";
    int at = record.indexOf(dealer);
    Assert.assertTrue(at >= 0);
    String changed = record.substring(0, at) + " oya=\"1\" hai0=" + record.substring(at + dealer.length());
    return StoredResults.gzip(changed.getBytes(StandardCharsets.UTF_8));
  }

  /** {@code query} が指す判断から、最後の候補を 1 つ消した文書を返す。 */
  private static byte[] dropCandidate(byte[] gzip, ScenarioQuery query) {
    JsonObject document =
        JsonParser.parseString(new String(StoredResults.gunzip(gzip), StandardCharsets.UTF_8))
            .getAsJsonObject();
    JsonArray candidates =
        Format6.decode(document).decisions().stream()
            .filter(
                located ->
                    located.round() == query.round()
                        && located.step().actor() == query.seat()
                        && located.step().eventIndex() == query.eventIndex())
            .findFirst()
            .orElseThrow()
            .step()
            .candidates();
    candidates.remove(candidates.size() - 1);
    return StoredResults.gzip(document.toString().getBytes(StandardCharsets.UTF_8));
  }

  private static void busy(Assert.ThrowingRunnable action) {
    Assert.assertEquals(
        Assert.expectThrows(ReviewerException.class, action).code(), ErrorCode.BUSY);
  }

  private static Format6.Located firstChoice(byte[] gzip) throws IOException {
    try (var in = new GZIPInputStream(new ByteArrayInputStream(gzip))) {
      var document =
          Format6.decode(
              JsonParser.parseString(new String(in.readAllBytes(), StandardCharsets.UTF_8))
                  .getAsJsonObject());
      return document.decisions().stream()
          .filter(located -> located.step().candidates().size() > 1)
          .findFirst()
          .orElseThrow();
    }
  }
}
