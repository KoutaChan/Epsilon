package com.epsilon.reviewer.analysis;

import com.epsilon.reviewer.engine.review.JobControl;
import com.epsilon.reviewer.error.ErrorBody;
import com.epsilon.reviewer.store.OwnerId;
import java.time.Instant;

/**
 * 1 つの解析の仕事の状態。状態の遷移と取り消しは監視で直列にし、進みと取り消しの印は解析の途中に鍵なしで読む。終わってからも 24 時間覚えるので、
 * 牌譜とモデルは持たない（worker に渡す手だけが持ち、走り終われば放す）。
 */
final class Job implements JobControl {
  final JobId id = JobId.random();
  final OwnerId owner;
  private volatile boolean cancelRequested;
  private volatile int progress;
  private JobStatus status = JobStatus.QUEUED;
  private String resultId;
  private ErrorBody error;
  private Instant finishedAt;

  Job(OwnerId owner) {
    this.owner = owner;
  }

  @Override
  public boolean cancelled() {
    return cancelRequested;
  }

  @Override
  public void progress(int percent) {
    progress = percent;
  }

  synchronized JobSnapshot snapshot() {
    return new JobSnapshot(id.value(), status, progress, resultId, error);
  }

  synchronized boolean finished() {
    return status.finished();
  }

  synchronized Instant finishedAt() {
    return finishedAt;
  }

  /** 待ちか走りの仕事だけを取り消す（保存の後は効かない）。 */
  synchronized void requestCancel() {
    if (status == JobStatus.QUEUED || status == JobStatus.RUNNING) cancelRequested = true;
  }

  /** @return 取り消されていれば false（cancelled にする） */
  synchronized boolean begin() {
    return advance(JobStatus.RUNNING);
  }

  /** @return 取り消されていれば false（cancelled にする。書きかけは捨てる） */
  synchronized boolean save() {
    return advance(JobStatus.SAVING);
  }

  synchronized void complete(String resultId) {
    this.resultId = resultId;
    progress = 100;
    finish(JobStatus.COMPLETED);
  }

  synchronized void fail(ErrorBody error) {
    this.error = error;
    finish(JobStatus.FAILED);
  }

  synchronized void cancel() {
    finish(JobStatus.CANCELLED);
  }

  private boolean advance(JobStatus next) {
    if (cancelRequested) {
      finish(JobStatus.CANCELLED);
      return false;
    }
    status = next;
    return true;
  }

  private void finish(JobStatus finished) {
    status = finished;
    finishedAt = Instant.now();
  }
}
