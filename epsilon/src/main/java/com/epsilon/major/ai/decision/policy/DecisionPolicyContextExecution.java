package com.epsilon.major.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;

/**
 * 一つのDecision デバイスパイプラインで、方策グラフの二分岐文脈を集約する実行境界。
 *
 * <p>学習ブロックとパラメーターは所有しない。推論順伝播ごとにCALL、KAN、KYUSHUの三出力を一度ずつ作り、返された出力の再利用を最終D2H完了まで {@link
 * Forward}が止める。
 */
public abstract class DecisionPolicyContextExecution implements AutoCloseable {

  public enum Site {
    CALL,
    KAN,
    KYUSHU
  }

  /** 一つの方策順伝播を開始する。 */
  public abstract Forward beginForward(NDManager workingManager);

  /** 完了を証明できないデバイス処理を保持しているなら{@code true}を返す。 */
  public abstract boolean hasIncompleteWork();

  /** 実行系を利用不能にした最初の失敗を返す。 */
  public abstract Throwable failure();

  /** 同じストリーム末尾の完了確認後、失敗順伝播が保持した出力を解放する。 */
  public abstract void releaseFailedForwardAfterCompletion();

  /** 一つの方策順伝播に属する文脈集約出力を所有する。 */
  public abstract static class Forward implements AutoCloseable {

    public abstract NDArray mixBinaryBranchContexts(
        Site site,
        NDArray baselineContext,
        NDArray selectedContext,
        NDArray selectedLogit,
        NDArray baselinePresence,
        NDArray selectedPresence);

    public abstract AutoCloseable seal();

    public abstract void poison(Throwable failure);
  }

  static DecisionPolicyContextExecution eager(int executionSlots) {
    return new Eager(executionSlots);
  }

  private static final class Eager extends DecisionPolicyContextExecution {

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
        throw new IllegalStateException("policy context execution is closed");
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
            "cannot close policy context execution with an active forward");
      }
      closed = true;
    }

    private int freeSlot() {
      for (int slot = 0; slot < activeForwards.length; slot++) {
        if (activeForwards[slot] == null) {
          return slot;
        }
      }
      throw new IllegalStateException("no policy context execution slot is available");
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
        throw new IllegalStateException("policy context forward does not own its execution slot");
      }
      activeForwards[slot] = null;
    }
  }

  private static final class EagerForward extends Forward {

    private final Eager owner;
    private final int slot;
    private final boolean[] executedSites = new boolean[Site.values().length];
    private int executedSiteCount;
    private boolean sealed;
    private boolean finished;

    private EagerForward(Eager owner, int slot) {
      this.owner = owner;
      this.slot = slot;
    }

    @Override
    public NDArray mixBinaryBranchContexts(
        Site site,
        NDArray baselineContext,
        NDArray selectedContext,
        NDArray selectedLogit,
        NDArray baselinePresence,
        NDArray selectedPresence) {
      if (finished || sealed || executedSites[site.ordinal()]) {
        throw new IllegalStateException("policy context site cannot execute twice: " + site);
      }
      executedSites[site.ordinal()] = true;
      executedSiteCount++;
      return DecisionPolicyContextPool.mixBinaryBranchContexts(
          baselineContext, selectedContext, selectedLogit, baselinePresence, selectedPresence);
    }

    @Override
    public AutoCloseable seal() {
      if (finished || sealed || executedSiteCount != executedSites.length) {
        throw new IllegalStateException(
            "policy context forward did not execute all sites: "
                + executedSiteCount
                + "/"
                + executedSites.length);
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
