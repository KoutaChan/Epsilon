package com.epsilon.reviewer;

import com.epsilon.reviewer.RecordFetcher.SourceRecord;
import com.epsilon.reviewer.dto.ModelInfo;
import com.epsilon.reviewer.dto.PreparedRecord;
import com.epsilon.reviewer.dto.RecordMetadata;
import com.epsilon.reviewer.dto.ReviewResult;
import com.epsilon.reviewer.engine.ReviewEngine;
import com.epsilon.reviewer.model.ModelCatalog;
import com.epsilon.reviewer.model.ModelDefinition;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 検証済みの牌譜を解析し、結果の保存までを管理するサービス。 */
public final class ReviewService implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(ReviewService.class);
  private static final Duration RECORD_LIFETIME = Duration.ofMinutes(30);
  private static final Duration JOB_LIFETIME = Duration.ofHours(24);
  private static final long MAX_PREPARED_BYTES = 128L * 1024 * 1024;
  private final ReviewEngine engine;
  private final ModelCatalog catalog;
  private final ResultStore store;
  private final Path temporaryDirectory;
  private final Map<String, RecordEntry> records = new HashMap<>();
  private final Map<String, Job> jobs = new HashMap<>();
  private final ExecutorService executor =
      new ThreadPoolExecutor(1, 1, 0, TimeUnit.SECONDS, new ArrayBlockingQueue<>(16));
  private final ScheduledExecutorService cleanup = Executors.newSingleThreadScheduledExecutor();

  public ReviewService(
      ReviewEngine engine, ModelCatalog catalog, ResultStore store, Path temporaryDirectory) {
    this.engine = engine;
    this.catalog = catalog;
    this.store = store;
    this.temporaryDirectory = temporaryDirectory;
    cleanup.scheduleWithFixedDelay(this::cleanExpired, 1, 5, TimeUnit.MINUTES);
  }

  public synchronized Prepared prepare(String owner, SourceRecord source) {
    expire();
    if (records.values().stream().filter(record -> record.owner.equals(owner)).count() >= 4
        || records.values().stream().mapToLong(record -> record.source.bytes().length).sum()
                + source.bytes().length
            > MAX_PREPARED_BYTES)
      throw new ApiException(429, "too_many_records", "Prepared record limit exceeded.");
    PreparedRecord prepared;
    try {
      prepared = engine.prepareRecord(source.bytes(), source.fileName());
    } catch (IllegalArgumentException failure) {
      throw new ApiException(422, "invalid_record", failure.getMessage());
    }
    String id = UUID.randomUUID().toString();
    records.put(id, new RecordEntry(owner, source, prepared, Instant.now()));
    return new Prepared(id, prepared.metadata());
  }

  public synchronized Accepted analyze(
      String owner, String recordId, String modelId, String revision, boolean desktop) {
    expire();
    RecordEntry record = records.get(recordId);
    if (record == null || !record.owner.equals(owner))
      throw new ApiException(404, "record_not_found", "Prepared record not found.");
    if (jobs.values().stream().anyMatch(job -> job.owner.equals(owner) && !job.finished()))
      throw new ApiException(
          409, "analysis_active", "An analysis is already active for this owner.");
    ModelDefinition model = catalog.requireAvailableModelRevision(modelId, revision);
    Job job = new Job(owner, model, desktop);
    jobs.put(job.id, job);
    try {
      executor.submit(() -> run(job, record));
    } catch (RejectedExecutionException failure) {
      jobs.remove(job.id);
      throw new ApiException(429, "queue_full", "Analysis queue is full.");
    }
    records.remove(recordId);
    return new Accepted(job.id, model.info());
  }

  private void run(Job job, RecordEntry record) {
    try {
      synchronized (job) {
        if (job.cancelRequested) {
          job.status = Status.CANCELLED;
          return;
        }
        job.status = Status.RUNNING;
      }
      ReviewResult result;
      try (var lease =
          catalog.acquire(job.model.modelId(), job.model.revision(), temporaryDirectory)) {
        result =
            engine.analyze(
                record.prepared,
                lease.definition(),
                () -> job.cancelRequested,
                progress -> job.progress = Math.min(99, progress));
      }
      // 中止と確定は同じロックで直列化する。確定済み結果を cancelled と報告しない。
      synchronized (job) {
        if (job.cancelRequested) {
          job.status = Status.CANCELLED;
          return;
        }
        job.status = Status.SAVING;
        store.save(job.owner, result, record.source, job.desktop);
        job.resultId = result.resultId();
        job.progress = 100;
        job.status = Status.COMPLETED;
      }
    } catch (Exception failure) {
      synchronized (job) {
        if (job.cancelRequested
            || failure instanceof CancellationException
            || failure instanceof InterruptedException) {
          job.status = Status.CANCELLED;
        } else {
          job.status = Status.FAILED;
          job.error =
              failure instanceof ApiException api
                  ? api.error()
                  : new ApiError("analysis_failed", "Analysis or result persistence failed.");
          LOG.error("Analysis job {} failed.", job.id, failure);
        }
      }
    } finally {
      job.finishedAt = Instant.now();
    }
  }

  public JobSnapshot job(String owner, String id) {
    return requireOwnedAnalysisJob(owner, id).snapshot();
  }

  public JobSnapshot cancel(String owner, String id) {
    Job job = requireOwnedAnalysisJob(owner, id);
    synchronized (job) {
      if (!job.finished()) job.cancelRequested = true;
      return job.snapshot();
    }
  }

  private synchronized Job requireOwnedAnalysisJob(String owner, String id) {
    Job job = jobs.get(id);
    if (job == null || !job.owner.equals(owner))
      throw new ApiException(404, "job_not_found", "Analysis job not found.");
    return job;
  }

  private void cleanExpired() {
    try {
      expire();
      store.cleanTemporary(Instant.now());
    } catch (Exception failure) {
      LOG.error("Could not remove expired reviewer data.", failure);
    }
  }

  private synchronized void expire() {
    Instant now = Instant.now();
    records.values().removeIf(record -> record.createdAt.plus(RECORD_LIFETIME).isBefore(now));
    jobs.values()
        .removeIf(job -> job.finishedAt != null && job.finishedAt.plus(JOB_LIFETIME).isBefore(now));
  }

  @Override
  public void close() {
    cleanup.shutdownNow();
    synchronized (this) {
      jobs.values().forEach(job -> job.cancelRequested = true);
    }
    executor.shutdown();
    boolean interrupted = false;
    for (ExecutorService pending : List.of(cleanup, executor)) {
      while (!pending.isTerminated()) {
        try {
          pending.awaitTermination(1, TimeUnit.SECONDS);
        } catch (InterruptedException failure) {
          interrupted = true;
        }
      }
    }
    synchronized (this) {
      records.clear();
      jobs.clear();
    }
    if (interrupted) Thread.currentThread().interrupt();
  }

  public record Prepared(String recordId, RecordMetadata metadata) {}

  public record Accepted(String jobId, ModelInfo model) {}

  public record JobSnapshot(
      String jobId,
      String status,
      int progress,
      boolean cancelRequested,
      String resultId,
      ApiError error) {}

  private record RecordEntry(
      String owner, SourceRecord source, PreparedRecord prepared, Instant createdAt) {}

  private enum Status {
    QUEUED,
    RUNNING,
    SAVING,
    COMPLETED,
    FAILED,
    CANCELLED;

    boolean finished() {
      return this == COMPLETED || this == FAILED || this == CANCELLED;
    }
  }

  private static final class Job {
    final String id = UUID.randomUUID().toString();
    final String owner;
    final ModelDefinition model;
    final boolean desktop;
    volatile Status status = Status.QUEUED;
    volatile int progress;
    volatile boolean cancelRequested;
    volatile Instant finishedAt;
    String resultId;
    ApiError error;

    Job(String owner, ModelDefinition model, boolean desktop) {
      this.owner = owner;
      this.model = model;
      this.desktop = desktop;
    }

    boolean finished() {
      return status.finished();
    }

    synchronized JobSnapshot snapshot() {
      return new JobSnapshot(
          id,
          status.name().toLowerCase(java.util.Locale.ROOT),
          progress,
          cancelRequested,
          resultId,
          error);
    }
  }
}
