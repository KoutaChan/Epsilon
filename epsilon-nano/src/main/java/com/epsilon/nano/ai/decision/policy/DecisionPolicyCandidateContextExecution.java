package com.epsilon.nano.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * 一つのDecision デバイスパイプラインで、候補種類の条件付き文脈をネットワーク列へ写像する実行境界。
 *
 * <p>学習パラメーターは所有しない。各順伝播は分岐候補 10列とPASS 1列を一度だけ作り、返した出力を最終D2H完了まで再利用させない。
 */
public abstract class DecisionPolicyCandidateContextExecution implements AutoCloseable {

  /** 一つの方策順伝播を開始する。 */
  public abstract Forward beginForward(NDManager workingManager);

  /** 完了を証明できないデバイス処理を保持しているなら{@code true}を返す。 */
  public abstract boolean hasIncompleteWork();

  /** 実行単位を利用不能にした最初の失敗を返す。 */
  public abstract Throwable failure();

  /** 同じストリーム末尾の完了確認後、失敗順伝播が保持した出力を解放する。 */
  public abstract void releaseFailedForwardAfterCompletion();

  /** 一つの方策順伝播に属する候補文脈出力を所有する。 */
  public abstract static class Forward implements AutoCloseable {

    public abstract DecisionPolicyContextPool.MappedCandidateContexts pool(
        NDArray candidateEmbeddings,
        NDArray candidateScores,
        DecisionPolicyCandidates.CandidateMasks candidateMasks);

    public abstract AutoCloseable seal();

    public abstract void poison(Throwable failure);
  }

  static DecisionPolicyCandidateContextExecution eager(int executionSlots) {
    return new Eager(executionSlots);
  }

  private static final class Eager extends DecisionPolicyCandidateContextExecution {

    private final EagerForward[] activeForwards;
    private boolean closed;

    private Eager(int executionSlots) {
      if (executionSlots <= 0) {
        throw new IllegalArgumentException("executionSlots must be positive");
      }
      activeForwards = new EagerForward[executionSlots];
    }

    @Override
    public Forward beginForward(NDManager workingManager) {
      if (closed) {
        throw new IllegalStateException("candidate context execution is closed");
      }
      int slot = freeSlot();
      EagerForward forward = new EagerForward(this, slot);
      activeForwards[slot] = forward;
      return forward;
    }

    @Override
    public boolean hasIncompleteWork() {
      return false;
    }

    @Override
    public Throwable failure() {
      return null;
    }

    @Override
    public void releaseFailedForwardAfterCompletion() {
      for (EagerForward forward : activeForwards) {
        if (forward != null) {
          forward.close();
        }
      }
    }

    @Override
    public void close() {
      if (hasActiveForward()) {
        throw new IllegalStateException(
            "cannot close candidate context execution with an active forward");
      }
      closed = true;
    }

    private int freeSlot() {
      for (int slot = 0; slot < activeForwards.length; slot++) {
        if (activeForwards[slot] == null) {
          return slot;
        }
      }
      throw new IllegalStateException("no candidate context execution slot is available");
    }

    private boolean hasActiveForward() {
      for (EagerForward forward : activeForwards) {
        if (forward != null) {
          return true;
        }
      }
      return false;
    }

    private void finish(int slot, EagerForward forward) {
      if (activeForwards[slot] != forward) {
        throw new IllegalStateException(
            "candidate context forward does not own its execution slot");
      }
      activeForwards[slot] = null;
    }
  }

  private static final class EagerForward extends Forward {

    private final Eager owner;
    private final int slot;
    private boolean executed;
    private boolean sealed;
    private boolean finished;

    private EagerForward(Eager owner, int slot) {
      this.owner = owner;
      this.slot = slot;
    }

    @Override
    public DecisionPolicyContextPool.MappedCandidateContexts pool(
        NDArray candidateEmbeddings,
        NDArray candidateScores,
        DecisionPolicyCandidates.CandidateMasks candidateMasks) {
      if (finished || sealed || executed) {
        throw new IllegalStateException("candidate contexts cannot execute twice");
      }
      executed = true;
      return DecisionPolicyContextPool.poolMappedCandidateTypes(
          candidateEmbeddings, candidateScores, candidateMasks);
    }

    @Override
    public AutoCloseable seal() {
      if (finished || sealed || !executed) {
        throw new IllegalStateException("candidate context forward did not execute");
      }
      sealed = true;
      return this;
    }

    @Override
    public void poison(Throwable failure) {
      finish();
    }

    @Override
    public void close() {
      finish();
    }

    private void finish() {
      if (!finished) {
        finished = true;
        owner.finish(slot, this);
      }
    }
  }
}
