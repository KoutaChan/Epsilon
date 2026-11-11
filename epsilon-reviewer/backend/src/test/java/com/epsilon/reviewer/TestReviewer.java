package com.epsilon.reviewer;

import com.epsilon.agent.settings.Settings;
import com.epsilon.mahjong.rules.Variant;
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
import java.util.List;
import java.util.Set;

/** テスト用のモデル（FixtureFamily）のカタログと、一時的な保存先で開いた Reviewer。 */
public final class TestReviewer implements AutoCloseable {
  public static final OwnerId OWNER = OwnerId.fromCookieKey("A".repeat(43));
  public static final OwnerId OTHER = OwnerId.fromCookieKey("B".repeat(43));

  public static final ModelKey FIXTURE = new ModelKey("fixture", "r1");

  /** {@link FixtureFamily#release} が開くまで、評価の途中で待つモデル。 */
  public static final ModelKey HELD = new ModelKey("fixture-hold", "r1");

  /** 開けないモデル。 */
  public static final ModelKey BROKEN = new ModelKey("fixture-broken", "r1");

  /** 全体の設定の既定値（同時に開くモデルは 2 つ、解析の worker は 1 つで、待てるのは 16 件）。 */
  public static final ReviewerSettings SETTINGS = Settings.bind(ReviewerSettings.class, List.of()).value();

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

  /**
   * 保存先を {@code directory} の下の data にし、テスト用のカタログを書いて設定を作る。カタログは models.json と 3 つのチェックポイントで、
   * チェックポイントの振る舞いは印のファイルで決める。
   */
  public static ReviewerConfig config(Path directory) throws IOException {
    checkpoint(directory, "checkpoint", null);
    checkpoint(directory, "checkpoint-hold", FixtureFamily.HOLD);
    checkpoint(directory, "checkpoint-broken", FixtureFamily.BROKEN);
    Path models = directory.resolve("models.json");
    Files.writeString(
        models,
        """
        {"models":[
        {"modelId":"fixture","revision":"r1","displayName":"Fixture","version":"1",
         "series":"%1$s","variant":"TENHOU_4P","checkpoint":"checkpoint"},
        {"modelId":"fixture-hold","revision":"r1","displayName":"Fixture","version":"1",
         "series":"%1$s","variant":"TENHOU_4P","checkpoint":"checkpoint-hold"},
        {"modelId":"fixture-broken","revision":"r1","displayName":"Fixture","version":"1",
         "series":"%1$s","variant":"TENHOU_4P","checkpoint":"checkpoint-broken"}],
         "defaultModel":{"modelId":"fixture","revision":"r1"}}
        """
            .formatted(FixtureFamily.SERIES.id()));
    return new ReviewerConfig(directory.resolve("data"), models, SETTINGS, FixtureFamily.families());
  }

  private static void checkpoint(Path directory, String name, String marker) throws IOException {
    Path checkpoint = Files.createDirectories(directory.resolve(name));
    Files.writeString(checkpoint.resolve("weights.bin"), "fixture");
    if (marker != null) Files.writeString(checkpoint.resolve(marker), "");
  }

  /** エンジンで打った四人打ちの半荘（{@link TestFiles#played} の種 0）を取り込んで解析し、保存した結果の id を返す。 */
  public ResultId analyze(OwnerId owner, ModelKey model) {
    return analyze(reviewer, owner, model);
  }

  /**
   * エンジンで打った四人打ちの半荘（{@link TestFiles#played} の種 0）を取り込んで解析し、終わるまで待って、保存した結果の id を返す。
   * 解析が完了しなければ AssertionError を投げる。
   */
  public static ResultId analyze(Reviewer reviewer, OwnerId owner, ModelKey model) {
    ImportedRecord imported = reviewer.importBytes(owner, TestFiles.played(Variant.TENHOU_4P, 0));
    AnalysisStarted started =
        reviewer.startAnalysis(owner, RecordId.parse(imported.recordId()), model);
    JobSnapshot job = await(reviewer, owner, JobId.parse(started.jobId()));
    if (job.status() != JobStatus.COMPLETED) throw new AssertionError("Analysis failed: " + job);
    return new ResultId(job.resultId());
  }

  /** ジョブが終わるまで待つ。30 秒で終わらなければ AssertionError を投げる。 */
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
