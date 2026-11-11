package com.epsilon.reviewer.analysis;

import com.epsilon.reviewer.engine.format.ModelInfo;
import com.epsilon.reviewer.engine.format.ResultWriter;
import com.epsilon.reviewer.engine.review.Analysis;
import com.epsilon.reviewer.error.ErrorBody;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.model.ModelEntry;
import com.epsilon.reviewer.model.ModelLease;
import com.epsilon.reviewer.model.ModelRegistry;
import com.epsilon.reviewer.record.PreparedRecord;
import com.epsilon.reviewer.record.PreparedRecords;
import com.epsilon.reviewer.store.OwnerId;
import com.epsilon.reviewer.store.ResultDraft;
import com.epsilon.reviewer.store.ResultId;
import com.epsilon.reviewer.store.ResultMeta;
import com.epsilon.reviewer.store.ResultStore;
import java.time.Duration;
import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.Map;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 解析ジョブを管理する。worker のスレッド（reviewer-analysis-&lt;n&gt;）の数、待たせておけるジョブの数、終わったジョブを覚えておく時間は設定で決める。
 * 終わっていないジョブはオーナーごとに 1 つまで。
 *
 * <p>worker では次の順に進む。running → モデルを借りる → 文書をストリームで書く（イベントと判断ごとに取り消しを確かめる）→ saving → 牌譜と meta
 * を書いて確定する → completed。読み込んだ牌譜を持つのは worker に渡すタスクだけで、ジョブが終われば必ず解放する。
 */
public final class AnalysisJobs implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(AnalysisJobs.class);

  private final PreparedRecords records;
  private final ModelRegistry models;
  private final ResultStore store;
  private final Duration retention;
  private final Map<JobId, Job> jobs = new ConcurrentHashMap<>();
  private final ThreadPoolExecutor worker;

  public AnalysisJobs(
      PreparedRecords records, ModelRegistry models, ResultStore store, AnalysisSettings settings) {
    this.records = records;
    this.models = models;
    this.store = store;
    retention = Duration.ofHours(settings.keepHours());
    AtomicInteger threads = new AtomicInteger();
    worker =
        new ThreadPoolExecutor(
            settings.workers(),
            settings.workers(),
            0,
            TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(settings.queue()),
            task -> new Thread(task, "reviewer-analysis-" + threads.incrementAndGet()));
  }

  /**
   * @param record 取り出した牌譜。受け付けられなければ、呼び出し元が元に戻す
   * @throws ReviewerException オーナーの終わっていないジョブがある（analysis_active）、待ちがいっぱい（queue_full）、または閉じている途中
   *     （shutting_down）
   */
  public synchronized AnalysisStarted submit(
      OwnerId owner, PreparedRecord record, ModelEntry model) {
    if (worker.isShutdown()) throw new ReviewerException(ErrorCode.SHUTTING_DOWN);
    for (Job job : jobs.values())
      if (job.owner.equals(owner) && !job.finished())
        throw new ReviewerException(ErrorCode.ANALYSIS_ACTIVE);
    var job = new Job(owner);
    jobs.put(job.id, job);
    try {
      worker.execute(() -> run(job, record, model));
    } catch (RejectedExecutionException failure) {
      jobs.remove(job.id);
      throw new ReviewerException(ErrorCode.QUEUE_FULL);
    }
    return new AnalysisStarted(job.id.value(), model.info());
  }

  /** @throws ReviewerException ジョブが無い、オーナーが違う、または覚えておく時間を過ぎた（job_not_found） */
  public JobSnapshot snapshot(OwnerId owner, JobId id) {
    return owned(owner, id).snapshot();
  }

  /** 待ち中か実行中のジョブなら、取り消しを要求する。状態は worker が変える。 */
  public JobSnapshot cancel(OwnerId owner, JobId id) {
    Job job = owned(owner, id);
    job.requestCancel();
    return job.snapshot();
  }

  /** 覚えておく時間を過ぎたジョブを捨てる。 */
  public void sweep() {
    Instant limit = Instant.now().minus(retention);
    jobs.values()
        .removeIf(job -> job.finishedAt() != null && job.finishedAt().isBefore(limit));
  }

  /** 全てのジョブを取り消し、実行中のジョブが止まるのを待つ。 */
  @Override
  public void close() {
    synchronized (this) {
      jobs.values().forEach(Job::requestCancel);
      worker.shutdown();
    }
    boolean interrupted = false;
    while (!worker.isTerminated()) {
      try {
        worker.awaitTermination(1, TimeUnit.SECONDS);
      } catch (InterruptedException failure) {
        interrupted = true;
      }
    }
    if (interrupted) Thread.currentThread().interrupt();
  }

  private Job owned(OwnerId owner, JobId id) {
    Job job = jobs.get(id);
    if (job == null || !job.owner.equals(owner))
      throw new ReviewerException(ErrorCode.JOB_NOT_FOUND);
    return job;
  }

  private void run(Job job, PreparedRecord record, ModelEntry model) {
    try {
      if (!job.begin()) return;
      ResultId id = ResultId.random();
      String createdAt = Instant.now().truncatedTo(ChronoUnit.MILLIS).toString();
      ModelInfo info = model.info();
      try (ModelLease lease = models.acquire(model.key());
          ResultDraft draft = store.draft(id)) {
        try (var writer =
            new ResultWriter(
                draft.document(), id.value(), createdAt, info, record.record().summary())) {
          Analysis.run(record.record(), lease, job, writer);
        }
        if (!job.save()) return;
        draft.record(record.source());
        store.commit(
            draft,
            new ResultMeta(
                ResultWriter.FORMAT,
                id.value(),
                job.owner.value(),
                createdAt,
                info,
                record.record().summary()));
        job.complete(id.value());
      }
    } catch (CancellationException failure) {
      job.cancel();
    } catch (ReviewerException failure) {
      LOG.warn("Analysis job {} failed: {}", job.id.value(), failure.getMessage());
      job.fail(ErrorBody.of(failure));
    } catch (Exception | Error failure) {
      LOG.error("Analysis job {} failed.", job.id.value(), failure);
      job.fail(ErrorBody.of(ErrorCode.ANALYSIS_FAILED));
    } finally {
      records.release(record);
    }
  }
}
