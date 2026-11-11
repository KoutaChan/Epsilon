package com.epsilon.reviewer.analysis;

import com.epsilon.reviewer.engine.review.JobControl;
import com.epsilon.reviewer.error.ErrorBody;
import com.epsilon.reviewer.store.OwnerId;
import java.time.Instant;

/**
 * 1 つの解析ジョブの状態。状態の遷移と取り消しは synchronized で 1 つずつ行う。進み具合と取り消しのフラグは、解析の途中にロックなしで読む。
 * 終わった後も設定の時間だけ覚えておくので、牌譜とモデルは持たない。それらを持つのは worker に渡すタスクだけで、実行が終われば解放する。
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

  /** 待ち中か実行中のジョブだけを取り消す。保存が始まった後は効かない。 */
  synchronized void requestCancel() {
    if (status == JobStatus.QUEUED || status == JobStatus.RUNNING) cancelRequested = true;
  }

  /** @return 取り消しが要求されていれば false（状態を cancelled にする） */
  synchronized boolean begin() {
    return advance(JobStatus.RUNNING);
  }

  /** @return 取り消しが要求されていれば false。状態を cancelled にし、呼び出し元は書きかけの文書を確定せずに捨てる */
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
