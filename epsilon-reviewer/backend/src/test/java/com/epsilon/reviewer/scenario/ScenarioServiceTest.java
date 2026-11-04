package com.epsilon.reviewer.scenario;

import com.epsilon.reviewer.FixturePolicyProvider;
import com.epsilon.reviewer.Reviewer;
import com.epsilon.reviewer.ReviewerConfig;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.TestReviewer;
import com.epsilon.reviewer.engine.Format6;
import com.epsilon.reviewer.engine.TestEdits;
import com.epsilon.reviewer.engine.scenario.ScenarioQuery;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.model.ModelCatalog;
import com.epsilon.reviewer.model.ModelRegistry;
import com.epsilon.reviewer.model.PolicyProviders;
import com.epsilon.reviewer.store.ResultId;
import com.epsilon.reviewer.store.ResultMeta;
import com.epsilon.reviewer.store.ResultStore;
import com.google.gson.JsonParser;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
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

/** もしもの受付（走る 1 つと待ち 4 つ、3 秒の待ち）、モデルが使えない時、削除で写しを外すこと。 */
public class ScenarioServiceTest {
  private Path directory;
  private ResultStore store;
  private ModelCatalog catalog;
  private ModelRegistry models;
  private ResultId fixture;
  private ResultId held;
  private ScenarioQuery query;

  @BeforeClass
  public void prepare() throws IOException {
    directory = TestFiles.directory("reviewer-scenario-");
    ReviewerConfig config = TestReviewer.config(directory);
    FixturePolicyProvider.release = new CountDownLatch(0);
    try (Reviewer reviewer = Reviewer.open(config)) {
      fixture = TestReviewer.analyze(reviewer, TestReviewer.OWNER, "kakan-a", TestReviewer.FIXTURE);
      held = TestReviewer.analyze(reviewer, TestReviewer.OWNER, "kakan-a", TestReviewer.HELD);
    }
    PolicyProviders providers = PolicyProviders.load();
    catalog = ModelCatalog.load(config.models(), providers);
    store = ResultStore.open(config.data());
    models = new ModelRegistry(catalog, providers, 2, Duration.ofMinutes(10));
    Format6.Located decision =
        firstChoice(store.readDocument(store.find(TestReviewer.OWNER, fixture)));
    query =
        new ScenarioQuery(
            decision.round(),
            decision.step().eventIndex(),
            decision.step().actor(),
            TestEdits.NONE,
            null);
  }

  @AfterClass(alwaysRun = true)
  public void close() {
    FixturePolicyProvider.release.countDown();
    models.close();
    store.close();
    TestFiles.delete(directory);
  }

  @Test
  public void oneEvaluationRunsFourWaitAndTheRestAreBusy() throws Exception {
    var scenarios = new ScenarioService(catalog, models, store);
    ResultMeta meta = store.find(TestReviewer.OWNER, held);
    FixturePolicyProvider.entered = new CountDownLatch(1);
    FixturePolicyProvider.release = new CountDownLatch(1);
    ExecutorService threads = Executors.newFixedThreadPool(ScenarioService.ADMITTED);
    try {
      List<Future<?>> admitted = new ArrayList<>();
      admitted.add(threads.submit(() -> scenarios.evaluate(meta, query)));
      Assert.assertTrue(FixturePolicyProvider.entered.await(30, TimeUnit.SECONDS));
      for (int i = 1; i < ScenarioService.ADMITTED; i++)
        admitted.add(threads.submit(() -> scenarios.evaluate(meta, query)));
      Thread.sleep(200);
      long started = System.nanoTime();
      busy(() -> scenarios.evaluate(meta, query));
      Assert.assertTrue(
          System.nanoTime() - started < Duration.ofSeconds(1).toNanos(),
          "the sixth is refused at once");
      FixturePolicyProvider.release.countDown();
      for (Future<?> future : admitted) future.get(30, TimeUnit.SECONDS);
      FixturePolicyProvider.release = new CountDownLatch(1);
      FixturePolicyProvider.entered = new CountDownLatch(1);
      Future<?> running = threads.submit(() -> scenarios.evaluate(meta, query));
      Assert.assertTrue(FixturePolicyProvider.entered.await(30, TimeUnit.SECONDS));
      started = System.nanoTime();
      busy(() -> scenarios.evaluate(meta, query));
      Assert.assertTrue(
          System.nanoTime() - started >= ScenarioService.WAIT.toNanos(), "the second waits 3 s");
      FixturePolicyProvider.release.countDown();
      running.get(30, TimeUnit.SECONDS);
    } finally {
      FixturePolicyProvider.release.countDown();
      threads.shutdownNow();
    }
  }

  @Test
  public void answersTheStoredDecisionAndForgetsDeletedResults() throws Exception {
    var scenarios = new ScenarioService(catalog, models, store);
    ResultMeta meta = store.find(TestReviewer.OWNER, fixture);
    Format6.Located decision = firstChoice(store.readDocument(meta));
    Assert.assertEquals(
        Format6.candidateSet(scenarios.evaluate(meta, query)), decision.step().candidateSet());
    store.delete(TestReviewer.OWNER, fixture);
    Assert.assertNotNull(scenarios.evaluate(meta, query), "the cached record still answers");
    scenarios.forget(fixture);
    Assert.assertEquals(
        Assert.expectThrows(ReviewerException.class, () -> scenarios.evaluate(meta, query)).code(),
        ErrorCode.RESULT_NOT_FOUND);
  }

  @Test
  public void resultsOfModelsMissingFromTheCatalogAreRefused() throws IOException {
    var scenarios = new ScenarioService(ModelCatalog.empty(), models, store);
    ResultMeta meta = store.find(TestReviewer.OWNER, held);
    Assert.assertEquals(
        Assert.expectThrows(ReviewerException.class, () -> scenarios.evaluate(meta, query)).code(),
        ErrorCode.MODEL_UNAVAILABLE);
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
