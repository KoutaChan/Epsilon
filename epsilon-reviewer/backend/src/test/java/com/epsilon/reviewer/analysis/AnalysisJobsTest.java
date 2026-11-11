package com.epsilon.reviewer.analysis;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.FixtureFamily;
import com.epsilon.reviewer.ReviewerConfig;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.TestReviewer;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.load.RecordLoader;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.model.ModelCatalog;
import com.epsilon.reviewer.model.ModelFamilies;
import com.epsilon.reviewer.model.ModelKey;
import com.epsilon.reviewer.model.ModelRegistry;
import com.epsilon.reviewer.record.PreparedRecord;
import com.epsilon.reviewer.record.PreparedRecords;
import com.epsilon.reviewer.store.OwnerId;
import com.epsilon.reviewer.store.ResultId;
import com.epsilon.reviewer.store.ResultStore;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

/**
 * 解析のジョブを確かめる。確かめるのは、状態の遷移、待っているジョブと実行中のジョブの取り消し、saving に入った後は取り消しが
 * 効かないこと、ジョブはオーナーごとに 1 つまでであること、待ちの上限、終わったジョブの牌譜は取り込める牌譜の上限に数えなくなること。
 */
public class AnalysisJobsTest {
  private static final OwnerId OWNER = TestReviewer.OWNER;
  private Path directory;
  private ModelCatalog catalog;
  private ResultStore store;
  private ModelRegistry models;
  private PreparedRecords records;
  private AnalysisJobs jobs;
  private byte[] bytes;
  private LoadedRecord loaded;

  @BeforeMethod
  public void open() throws IOException {
    FixtureFamily.entered = new CountDownLatch(0);
    FixtureFamily.release = new CountDownLatch(0);
    directory = TestFiles.directory("reviewer-jobs-");
    ReviewerConfig config = TestReviewer.config(directory);
    ModelFamilies providers = config.families();
    catalog = ModelCatalog.load(config.models(), providers);
    store = ResultStore.open(config.data());
    models = new ModelRegistry(catalog, providers, 2, Duration.ofMinutes(10));
    records = new PreparedRecords();
    jobs = new AnalysisJobs(records, models, store, config.settings().analysis());
    bytes = TestFiles.played(Variant.TENHOU_4P, 0);
    loaded = RecordLoader.load(bytes);
  }

  @AfterMethod(alwaysRun = true)
  public void close() {
    FixtureFamily.release.countDown();
    jobs.close();
    models.close();
    store.close();
    TestFiles.delete(directory);
  }

  @Test
  public void completedJobsSaveTheResultAndReleaseTheirRecord() {
    JobSnapshot done = await(OWNER, submit(OWNER, TestReviewer.FIXTURE));
    Assert.assertEquals(done.status(), JobStatus.COMPLETED, done.toString());
    Assert.assertEquals(done.progress(), 100);
    Assert.assertNull(done.error());
    var meta = store.find(OWNER, new ResultId(done.resultId()));
    Assert.assertEquals(meta.record(), loaded.summary());
    Assert.assertEquals(meta.model().modelId(), "fixture");
    for (int i = 0; i < 4; i++) records.add(OWNER, loaded, bytes);
    refused(() -> records.add(OWNER, loaded, bytes), ErrorCode.TOO_MANY_RECORDS);
    JobId id = JobId.parse(done.jobId());
    refused(() -> jobs.snapshot(TestReviewer.OTHER, id), ErrorCode.JOB_NOT_FOUND);
  }

  @Test
  public void cancellingARunningJobDiscardsTheDocument() throws Exception {
    FixtureFamily.entered = new CountDownLatch(1);
    FixtureFamily.release = new CountDownLatch(1);
    JobId job = submit(OWNER, TestReviewer.HELD);
    Assert.assertTrue(FixtureFamily.entered.await(30, TimeUnit.SECONDS));
    Assert.assertEquals(jobs.cancel(OWNER, job).status(), JobStatus.RUNNING);
    FixtureFamily.release.countDown();
    Assert.assertEquals(await(OWNER, job).status(), JobStatus.CANCELLED);
    Assert.assertTrue(store.history(OWNER).isEmpty());
    try (var staging = Files.list(directory.resolve(Path.of("data", "staging")))) {
      Assert.assertEquals(staging.count(), 0L);
    }
  }

  @Test
  public void oneJobPerOwnerSixteenWaitingAndQueuedJobsCancelWhenPickedUp() throws Exception {
    FixtureFamily.entered = new CountDownLatch(1);
    FixtureFamily.release = new CountDownLatch(1);
    JobId running = submit(OWNER, TestReviewer.HELD);
    Assert.assertTrue(FixtureFamily.entered.await(30, TimeUnit.SECONDS));
    refused(() -> submit(OWNER, TestReviewer.FIXTURE), ErrorCode.ANALYSIS_ACTIVE);
    List<OwnerId> owners = new ArrayList<>();
    List<JobId> queued = new ArrayList<>();
    for (int i = 0; i < TestReviewer.SETTINGS.analysis().queue(); i++) {
      OwnerId owner = OwnerId.fromCookieKey(String.format("%043d", i));
      owners.add(owner);
      queued.add(submit(owner, TestReviewer.FIXTURE));
    }
    refused(
        () -> submit(OwnerId.fromCookieKey("C".repeat(43)), TestReviewer.FIXTURE),
        ErrorCode.QUEUE_FULL);
    for (int i = 0; i < queued.size(); i++)
      Assert.assertEquals(jobs.cancel(owners.get(i), queued.get(i)).status(), JobStatus.QUEUED);
    FixtureFamily.release.countDown();
    Assert.assertEquals(await(OWNER, running).status(), JobStatus.COMPLETED);
    for (int i = 0; i < queued.size(); i++)
      Assert.assertEquals(await(owners.get(i), queued.get(i)).status(), JobStatus.CANCELLED);
  }

  /**
   * worker の数と待てる数は設定に従う。worker が 2 つなら、待たせたモデルのジョブが動いている間に、別のモデルのジョブが終わる（同じ
   * モデルは 1 つずつ評価する）。worker が 1 つで待てる数が 2 なら、3 つ目の待ちを queue_full で断る。
   */
  @Test
  public void workersAndTheQueueFollowTheSettings() throws Exception {
    List<OwnerId> owners = new ArrayList<>();
    for (int i = 0; i < 4; i++) owners.add(OwnerId.fromCookieKey(String.format("%043d", 100 + i)));
    jobs.close();
    jobs = new AnalysisJobs(records, models, store, new AnalysisSettings(2, 16, 24));
    FixtureFamily.entered = new CountDownLatch(1);
    FixtureFamily.release = new CountDownLatch(1);
    JobId held = submit(owners.get(0), TestReviewer.HELD);
    Assert.assertTrue(FixtureFamily.entered.await(30, TimeUnit.SECONDS));
    JobId beside = submit(owners.get(1), TestReviewer.FIXTURE);
    Assert.assertEquals(await(owners.get(1), beside).status(), JobStatus.COMPLETED);
    Assert.assertEquals(jobs.snapshot(owners.get(0), held).status(), JobStatus.RUNNING);
    FixtureFamily.release.countDown();
    Assert.assertEquals(await(owners.get(0), held).status(), JobStatus.COMPLETED);

    jobs.close();
    jobs = new AnalysisJobs(records, models, store, new AnalysisSettings(1, 2, 24));
    FixtureFamily.entered = new CountDownLatch(1);
    FixtureFamily.release = new CountDownLatch(1);
    JobId running = submit(owners.get(0), TestReviewer.HELD);
    Assert.assertTrue(FixtureFamily.entered.await(30, TimeUnit.SECONDS));
    JobId first = submit(owners.get(1), TestReviewer.FIXTURE);
    JobId second = submit(owners.get(2), TestReviewer.FIXTURE);
    refused(() -> submit(owners.get(3), TestReviewer.FIXTURE), ErrorCode.QUEUE_FULL);
    FixtureFamily.release.countDown();
    Assert.assertEquals(await(owners.get(0), running).status(), JobStatus.COMPLETED);
    Assert.assertEquals(await(owners.get(1), first).status(), JobStatus.COMPLETED);
    Assert.assertEquals(await(owners.get(2), second).status(), JobStatus.COMPLETED);
  }

  @Test
  public void modelsThatCannotOpenFailTheJob() {
    JobSnapshot failed = await(OWNER, submit(OWNER, TestReviewer.BROKEN));
    Assert.assertEquals(failed.status(), JobStatus.FAILED);
    Assert.assertEquals(failed.error().code(), "model_unavailable");
    Assert.assertNull(failed.resultId());
    Assert.assertTrue(store.history(OWNER).isEmpty());
  }

  @Test
  public void cancellingAfterSavingHasNoEffect() {
    var saving = new Job(OWNER);
    Assert.assertTrue(saving.begin());
    Assert.assertTrue(saving.save());
    saving.requestCancel();
    saving.complete("result");
    Assert.assertEquals(saving.snapshot().status(), JobStatus.COMPLETED);
    var waiting = new Job(OWNER);
    waiting.requestCancel();
    Assert.assertFalse(waiting.begin());
    Assert.assertEquals(waiting.snapshot().status(), JobStatus.CANCELLED);
  }

  private JobId submit(OwnerId owner, ModelKey model) {
    PreparedRecord added = records.add(owner, loaded, bytes);
    PreparedRecord taken = records.take(owner, added.id());
    return JobId.parse(jobs.submit(owner, taken, catalog.require(model)).jobId());
  }

  private JobSnapshot await(OwnerId owner, JobId id) {
    long deadline = System.nanoTime() + Duration.ofSeconds(60).toNanos();
    while (System.nanoTime() < deadline) {
      JobSnapshot job = jobs.snapshot(owner, id);
      if (job.status().finished()) return job;
      try {
        Thread.sleep(10);
      } catch (InterruptedException failure) {
        Thread.currentThread().interrupt();
        throw new AssertionError(failure);
      }
    }
    throw new AssertionError("The job did not finish.");
  }

  private static void refused(Assert.ThrowingRunnable action, ErrorCode code) {
    Assert.assertEquals(Assert.expectThrows(ReviewerException.class, action).code(), code);
  }
}
