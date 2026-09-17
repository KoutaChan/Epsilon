package com.epsilon.nano.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * 一つのDecision デバイスパイプラインで疎な二値分岐を実行する境界。
 *
 * <p>学習ブロックとパラメーターは所有せず、凍結済みパラメーターを参照する。各順伝播が返す密なスコアは、最終D2H完了まで{@link Forward}の利用権で再利用を止める。
 */
public abstract class DecisionPolicyIndexedAffineExecution implements AutoCloseable {

  public enum Site {
    RIICHI,
    CALL,
    RON,
    KAN,
    KYUSHU,
    TSUMO
  }

  /** 一つの方策順伝播を開始する。 */
  public abstract Forward beginForward(NDManager workingManager);

  /** 完了を証明できないデバイス処理を保持しているなら{@code true}を返す。 */
  public abstract boolean hasIncompleteWork();

  /** 実行単位を利用不能にした最初の失敗を返す。 */
  public abstract Throwable failure();

  /** 同じストリーム末尾の完了確認後に、失敗順伝播が保持した利用権を解放する。 */
  public abstract void releaseFailedForwardAfterCompletion();

  /** 一つの方策順伝播に属する有効な行だけを計算する分岐出力を所有する。 */
  public abstract static class Forward implements AutoCloseable {

    public abstract NDArray scoreRows(
        Site site,
        ParameterStore parameterStore,
        NDArray state,
        NDArray baselineBranch,
        NDArray selectedBranch,
        NDArray activeRows,
        long rowCount,
        PairList<String, Object> runtimeParameters);

    public abstract NDArray scoreActions(
        ParameterStore parameterStore,
        NDArray state,
        NDArray baselineBranch,
        NDArray selectedBranch,
        NDArray activeActions,
        long rowCount,
        int actionCapacity,
        PairList<String, Object> runtimeParameters);

    public abstract AutoCloseable seal();

    public abstract void poison(Throwable failure);
  }

  static DecisionPolicyIndexedAffineExecution eager(
      EpsilonBinaryBranchGate riichiGate,
      EpsilonBinaryBranchGate callGate,
      EpsilonBinaryBranchGate ronGate,
      EpsilonBinaryBranchGate kanGate,
      EpsilonBinaryBranchGate kyushuGate,
      EpsilonBinaryBranchGate tsumoGate,
      int executionSlots) {
    return new Eager(
        new EpsilonBinaryBranchGate[] {
          riichiGate, callGate, ronGate, kanGate, kyushuGate, tsumoGate
        },
        executionSlots);
  }

  private static final class Eager extends DecisionPolicyIndexedAffineExecution {

    private final EpsilonBinaryBranchGate[] gates;
    private final EagerForward[] activeForwards;
    private boolean closed;

    private Eager(EpsilonBinaryBranchGate[] gates, int executionSlots) {
      if (executionSlots <= 0) {
        throw new IllegalArgumentException("executionSlots must be positive");
      }
      this.gates = gates;
      activeForwards = new EagerForward[executionSlots];
    }

    @Override
    public Forward beginForward(NDManager workingManager) {
      if (closed) {
        throw new IllegalStateException("indexed affine execution is closed");
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
            "cannot close indexed affine execution with an active forward");
      }
      closed = true;
    }

    private int freeSlot() {
      for (int slot = 0; slot < activeForwards.length; slot++) {
        if (activeForwards[slot] == null) {
          return slot;
        }
      }
      throw new IllegalStateException("no indexed affine execution slot is available");
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
        throw new IllegalStateException("indexed affine forward does not own its execution slot");
      }
      activeForwards[slot] = null;
    }

    private EpsilonBinaryBranchGate gate(Site site) {
      return gates[site.ordinal()];
    }
  }

  private static final class EagerForward extends Forward {

    private final Eager owner;
    private final int slot;
    private boolean sealed;
    private boolean finished;

    private EagerForward(Eager owner, int slot) {
      this.owner = owner;
      this.slot = slot;
    }

    @Override
    public NDArray scoreRows(
        Site site,
        ParameterStore parameterStore,
        NDArray state,
        NDArray baselineBranch,
        NDArray selectedBranch,
        NDArray activeRows,
        long rowCount,
        PairList<String, Object> runtimeParameters) {
      return gate(site)
          .scoreRows(
              parameterStore,
              state,
              baselineBranch,
              selectedBranch,
              activeRows,
              rowCount,
              false,
              runtimeParameters);
    }

    @Override
    public NDArray scoreActions(
        ParameterStore parameterStore,
        NDArray state,
        NDArray baselineBranch,
        NDArray selectedBranch,
        NDArray activeActions,
        long rowCount,
        int actionCapacity,
        PairList<String, Object> runtimeParameters) {
      return gate(Site.RIICHI)
          .scoreActions(
              parameterStore,
              state,
              baselineBranch,
              selectedBranch,
              activeActions,
              rowCount,
              actionCapacity,
              false,
              runtimeParameters);
    }

    @Override
    public AutoCloseable seal() {
      if (finished || sealed) {
        throw new IllegalStateException("indexed affine forward cannot be sealed twice");
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

    private EpsilonBinaryBranchGate gate(Site site) {
      return owner.gate(site);
    }

    private void finish() {
      if (!finished) {
        finished = true;
        owner.finish(slot, this);
      }
    }
  }
}
