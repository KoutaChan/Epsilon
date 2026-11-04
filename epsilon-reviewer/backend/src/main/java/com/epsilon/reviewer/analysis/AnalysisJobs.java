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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 解析の仕事。1 本の worker（reviewer-analysis）と 16 の待ちで、持ち主ごとに終わっていない仕事は 1 つ。終わった仕事は 24 時間覚える。
 *
 * <p>worker: running → モデルを借りる → 文書を流して書く（出来事と判断ごとに取り消しを見る）→ saving → 牌譜と meta を書いて確定 →
 * completed。読み込んだ牌譜は worker に渡す手だけが持ち、仕事の終わりに必ず放す。
 */
public final class AnalysisJobs implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(AnalysisJobs.class);

  /** 走りの外の待ちの数。 */
  static final int QUEUE = 16;

  /** 終わった仕事を覚える時間。 */
  static final Duration RETENTION = Duration.ofHours(24);

  private final PreparedRecords records;
  private final ModelRegistry models;
  private final ResultStore store;
  private final Map<JobId, Job> jobs = new ConcurrentHashMap<>();
  private final ThreadPoolExecutor worker =
      new ThreadPoolExecutor(
          1,
          1,
          0,
          TimeUnit.SECONDS,
          new ArrayBlockingQueue<>(QUEUE),
          task -> new Thread(task, "reviewer-analysis"));

  public AnalysisJobs(PreparedRecords records, ModelRegistry models, ResultStore store) {
    this.records = records;
    this.models = models;
    this.store = store;
  }

  /**
   * @param record take した牌譜（失敗した時は呼び出し側が restore する）
   * @throws ReviewerException analysis_active・queue_full・shutting_down
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

  /** @throws ReviewerException job_not_found（無い・他の持ち主・終わってから 24 時間後） */
  public JobSnapshot snapshot(OwnerId owner, JobId id) {
    return owned(owner, id).snapshot();
  }

  /** 待ちか走りなら取り消しを頼む（状態は worker が変える）。 */
  public JobSnapshot cancel(OwnerId owner, JobId id) {
    Job job = owned(owner, id);
    job.requestCancel();
    return job.snapshot();
  }

  /** 終わってから 24 時間たった仕事を忘れる。 */
  public void sweep() {
    Instant limit = Instant.now().minus(RETENTION);
    jobs.values()
        .removeIf(job -> job.finishedAt() != null && job.finishedAt().isBefore(limit));
  }

  /** 全ての仕事を取り消し、走る仕事が止まるのを待つ。 */
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
