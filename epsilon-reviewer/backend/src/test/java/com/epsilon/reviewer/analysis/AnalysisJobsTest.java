package com.epsilon.reviewer.analysis;

import com.epsilon.reviewer.FixturePolicyProvider;
import com.epsilon.reviewer.ReviewerConfig;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.TestReviewer;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.load.RecordLoader;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.model.ModelCatalog;
import com.epsilon.reviewer.model.ModelKey;
import com.epsilon.reviewer.model.ModelRegistry;
import com.epsilon.reviewer.model.PolicyProviders;
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

/** 仕事の状態の遷移、待ちと走りの取り消し、saving の後の取り消し、持ち主ごとに 1 つ、待ちの上限、牌譜を放すこと。 */
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
    FixturePolicyProvider.entered = new CountDownLatch(0);
    FixturePolicyProvider.release = new CountDownLatch(0);
    directory = TestFiles.directory("reviewer-jobs-");
    ReviewerConfig config = TestReviewer.config(directory);
    PolicyProviders providers = PolicyProviders.load();
    catalog = ModelCatalog.load(config.models(), providers);
    store = ResultStore.open(config.data());
    models = new ModelRegistry(catalog, providers, 2, Duration.ofMinutes(10));
    records = new PreparedRecords();
    jobs = new AnalysisJobs(records, models, store);
    bytes = TestFiles.golden("kakan-a");
    loaded = RecordLoader.load(bytes);
  }

  @AfterMethod(alwaysRun = true)
  public void close() {
    FixturePolicyProvider.release.countDown();
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
    FixturePolicyProvider.entered = new CountDownLatch(1);
    FixturePolicyProvider.release = new CountDownLatch(1);
    JobId job = submit(OWNER, TestReviewer.HELD);
    Assert.assertTrue(FixturePolicyProvider.entered.await(30, TimeUnit.SECONDS));
    Assert.assertEquals(jobs.cancel(OWNER, job).status(), JobStatus.RUNNING);
    FixturePolicyProvider.release.countDown();
    Assert.assertEquals(await(OWNER, job).status(), JobStatus.CANCELLED);
    Assert.assertTrue(store.history(OWNER).isEmpty());
    try (var staging = Files.list(directory.resolve(Path.of("data", "staging")))) {
      Assert.assertEquals(staging.count(), 0L);
    }
  }

  @Test
  public void oneJobPerOwnerSixteenWaitingAndQueuedJobsCancelWhenPickedUp() throws Exception {
    FixturePolicyProvider.entered = new CountDownLatch(1);
    FixturePolicyProvider.release = new CountDownLatch(1);
    JobId running = submit(OWNER, TestReviewer.HELD);
    Assert.assertTrue(FixturePolicyProvider.entered.await(30, TimeUnit.SECONDS));
    refused(() -> submit(OWNER, TestReviewer.FIXTURE), ErrorCode.ANALYSIS_ACTIVE);
    List<OwnerId> owners = new ArrayList<>();
    List<JobId> queued = new ArrayList<>();
    for (int i = 0; i < AnalysisJobs.QUEUE; i++) {
      OwnerId owner = OwnerId.fromCookieKey(String.format("%043d", i));
      owners.add(owner);
      queued.add(submit(owner, TestReviewer.FIXTURE));
    }
    refused(
        () -> submit(OwnerId.fromCookieKey("C".repeat(43)), TestReviewer.FIXTURE),
        ErrorCode.QUEUE_FULL);
    for (int i = 0; i < queued.size(); i++)
      Assert.assertEquals(jobs.cancel(owners.get(i), queued.get(i)).status(), JobStatus.QUEUED);
    FixturePolicyProvider.release.countDown();
    Assert.assertEquals(await(OWNER, running).status(), JobStatus.COMPLETED);
    for (int i = 0; i < queued.size(); i++)
      Assert.assertEquals(await(owners.get(i), queued.get(i)).status(), JobStatus.CANCELLED);
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
