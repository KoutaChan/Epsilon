package com.epsilon.nano.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;

/**
 * 一つのDecision デバイスパイプラインで、4人の牌別文脈を合成する実行境界。
 *
 * <p>学習パラメーターの所有者は {@link EpsilonPlayerTileContextMixer} のまま保ち、凍結推論では実行計画だけを
 * パイプラインごとに所有する。返した永続出力の再利用時点は {@link Forward#seal()} が返す依存オブジェクトで管理する。
 */
public abstract class DecisionPolicyPlayerTileContextExecution implements AutoCloseable {

  /** 一つの方策順伝播を開始する。 */
  public abstract Forward beginForward(NDManager workingManager);

  /** 完了を証明できないデバイス処理を保持しているなら{@code true}を返す。 */
  public abstract boolean hasIncompleteWork();

  /** 実行単位を利用不能にした最初の失敗を返す。 */
  public abstract Throwable failure();

  /** 同じストリーム末尾の完了確認後、失敗順伝播が保持した出力を解放する。 */
  public abstract void releaseFailedForwardAfterCompletion();

  /** 一つの方策順伝播に属する牌別文脈出力を所有する。 */
  public abstract static class Forward implements AutoCloseable {

    public abstract NDArray mix(ParameterStore parameterStore, NDArray byPlayer);

    public abstract AutoCloseable seal();

    public abstract void poison(Throwable failure);
  }

  static DecisionPolicyPlayerTileContextExecution eager(
      EpsilonPlayerTileContextMixer mixer, int executionSlots) {
    return new Eager(mixer, executionSlots);
  }

  private static final class Eager extends DecisionPolicyPlayerTileContextExecution {

    private final EpsilonPlayerTileContextMixer mixer;
    private final EagerForward[] activeForwards;
    private boolean closed;

    private Eager(EpsilonPlayerTileContextMixer mixer, int executionSlots) {
      this.mixer = mixer;
      activeForwards = new EagerForward[executionSlots];
    }

    @Override
    public Forward beginForward(NDManager workingManager) {
      if (closed) {
        throw new IllegalStateException("player tile context execution is closed");
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
            "cannot close player tile context execution with an active forward");
      }
      closed = true;
    }

    private int freeSlot() {
      for (int slot = 0; slot < activeForwards.length; slot++) {
        if (activeForwards[slot] == null) {
          return slot;
        }
      }
      throw new IllegalStateException("no player tile context execution slot is available");
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
        throw new IllegalStateException("player tile context forward lost its execution slot");
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
    public NDArray mix(ParameterStore parameterStore, NDArray byPlayer) {
      if (finished || sealed || executed) {
        throw new IllegalStateException("player tile context cannot execute twice");
      }
      executed = true;
      return owner.mixer.mix(parameterStore, byPlayer, false);
    }

    @Override
    public AutoCloseable seal() {
      if (finished || sealed || !executed) {
        throw new IllegalStateException("player tile context was not executed");
      }
      sealed = true;
      return this;
    }

    @Override
    public void poison(Throwable failure) {
      close();
    }

    @Override
    public void close() {
      if (!finished) {
        finished = true;
        owner.finish(slot, this);
      }
    }
  }
}
