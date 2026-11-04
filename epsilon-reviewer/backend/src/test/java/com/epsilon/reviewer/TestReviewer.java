package com.epsilon.reviewer;

import com.epsilon.reviewer.analysis.AnalysisStarted;
import com.epsilon.reviewer.analysis.JobId;
import com.epsilon.reviewer.analysis.JobSnapshot;
import com.epsilon.reviewer.analysis.JobStatus;
import com.epsilon.reviewer.model.ModelKey;
import com.epsilon.reviewer.record.ImportedRecord;
import com.epsilon.reviewer.record.RecordId;
import com.epsilon.reviewer.store.OwnerId;
import com.epsilon.reviewer.store.ResultId;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Set;

/** 試験のモデル（FixturePolicyProvider）のカタログと一時の保存先で開いた Reviewer。 */
public final class TestReviewer implements AutoCloseable {
  public static final OwnerId OWNER = OwnerId.fromCookieKey("A".repeat(43));
  public static final OwnerId OTHER = OwnerId.fromCookieKey("B".repeat(43));

  public static final ModelKey FIXTURE = new ModelKey("fixture", "r1");

  /** 評価の途中で FixturePolicyProvider.release を待つモデル。 */
  public static final ModelKey HELD = new ModelKey("fixture-hold", "r1");

  /** 開けないモデル。 */
  public static final ModelKey BROKEN = new ModelKey("fixture-broken", "r1");

  private static final Set<JobStatus> FINISHED =
      Set.of(JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED);

  public final Path directory;
  public final Reviewer reviewer;

  private TestReviewer(Path directory, Reviewer reviewer) {
    this.directory = directory;
    this.reviewer = reviewer;
  }

  public static TestReviewer open() {
    Path directory = TestFiles.directory("reviewer-test-");
    try {
      return new TestReviewer(directory, Reviewer.open(config(directory)));
    } catch (IOException failure) {
      TestFiles.delete(directory);
      throw new UncheckedIOException(failure);
    }
  }

  /** 保存先 data と試験のカタログ（models.json と 1 つのファイルのチェックポイント）。 */
  public static ReviewerConfig config(Path directory) throws IOException {
    Path checkpoint = Files.createDirectories(directory.resolve("checkpoint"));
    Files.writeString(checkpoint.resolve("weights.txt"), "fixture");
    Path models = directory.resolve("models.json");
    Files.writeString(
        models,
        """
        {"models":[
        {"modelId":"fixture","revision":"r1","displayName":"Fixture","version":"1",
         "series":"fixture-scenario","checkpoint":"checkpoint"},
        {"modelId":"fixture-hold","revision":"r1","displayName":"Fixture","version":"1",
         "series":"fixture-scenario","checkpoint":"checkpoint","options":{"hold":"true"}},
        {"modelId":"fixture-broken","revision":"r1","displayName":"Fixture","version":"1",
         "series":"fixture-scenario","checkpoint":"checkpoint","options":{"broken":"true"}}],
         "defaultModel":{"modelId":"fixture","revision":"r1"}}
        """);
    return new ReviewerConfig(directory.resolve("data"), models, 2);
  }

  /** 天鳳の牌譜を取り込んで解析し、保存した結果の id。 */
  public ResultId analyze(OwnerId owner, String record, ModelKey model) {
    return analyze(reviewer, owner, record, model);
  }

  /** 天鳳の牌譜を取り込んで解析し、保存した結果の id。 */
  public static ResultId analyze(Reviewer reviewer, OwnerId owner, String record, ModelKey model) {
    ImportedRecord imported = reviewer.importBytes(owner, TestFiles.golden(record));
    AnalysisStarted started =
        reviewer.startAnalysis(owner, RecordId.parse(imported.recordId()), model);
    JobSnapshot job = await(reviewer, owner, JobId.parse(started.jobId()));
    if (job.status() != JobStatus.COMPLETED) throw new AssertionError("Analysis failed: " + job);
    return new ResultId(job.resultId());
  }

  /** 仕事が終わるまで待つ（30 秒まで）。 */
  public static JobSnapshot await(Reviewer reviewer, OwnerId owner, JobId id) {
    long deadline = System.nanoTime() + Duration.ofSeconds(30).toNanos();
    while (System.nanoTime() < deadline) {
      JobSnapshot job = reviewer.job(owner, id);
      if (FINISHED.contains(job.status())) return job;
      try {
        Thread.sleep(10);
      } catch (InterruptedException failure) {
        Thread.currentThread().interrupt();
        throw new AssertionError(failure);
      }
    }
    throw new AssertionError("The analysis job did not finish before the deadline.");
  }

  @Override
  public void close() {
    reviewer.close();
    TestFiles.delete(directory);
  }
}
