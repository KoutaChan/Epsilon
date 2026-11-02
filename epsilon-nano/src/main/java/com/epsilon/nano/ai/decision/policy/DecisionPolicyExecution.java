package com.epsilon.nano.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/**
 * 方策の1回の推論で使う演算と、その出力バッファの利用権をまとめて管理する。
 *
 * <p>線形変換、候補入力の連結、文脈の集約などの演算は独立して実行する。{@link Forward#seal()}
 * が返す依存オブジェクトを最終出力のホスト転送完了まで保持し、参照中のバッファの再利用を防ぐ。開始と終了は同じ投入スレッドから直列に行う。
 */
public final class DecisionPolicyExecution implements AutoCloseable {

  private final DecisionPolicyAffineExecution affine;
  private final DecisionPolicyIndexedAffineExecution indexedAffine;
  private final DecisionCandidateExecution representation;
  private final DecisionPolicyCandidateContextExecution candidateContext;
  private final DecisionPolicyContextExecution context;
  private final Forward[] activeForwards;
  private boolean closed;

  DecisionPolicyExecution(
      DecisionPolicyAffineExecution affine,
      DecisionPolicyIndexedAffineExecution indexedAffine,
      DecisionCandidateExecution representation,
      DecisionPolicyCandidateContextExecution candidateContext,
      DecisionPolicyContextExecution context,
      int executionSlots) {
    if (executionSlots <= 0) {
      throw new IllegalArgumentException("executionSlots must be positive");
    }
    this.affine = affine;
    this.indexedAffine = indexedAffine;
    this.representation = representation;
    this.candidateContext = candidateContext;
    this.context = context;
    activeForwards = new Forward[executionSlots];
  }

  /**
   * 空いている実行枠で一つの方策順伝播を開始する。
   *
   * <p>順伝播と、その{@link Forward#seal()}が返す依存オブジェクトは一対一である。複数順伝播を投入順に開始できるが、全メソッド
   * 呼び出しはデバイスパイプラインの投入スレッドが直列化する。
   */
  public Forward beginForward(NDManager workingManager) {
    if (closed) {
      throw new IllegalStateException("policy execution is closed");
    }
    int slot = freeSlot();
    DecisionPolicyAffineExecution.Forward affineComponentForward =
        affine.beginForward(workingManager);
    DecisionPolicyIndexedAffineExecution.Forward indexedComponentForward = null;
    DecisionCandidateExecution.Forward representationForward = null;
    DecisionPolicyCandidateContextExecution.Forward candidateContextComponentForward = null;
    DecisionPolicyContextExecution.Forward contextComponentForward = null;
    try {
      indexedComponentForward = indexedAffine.beginForward(workingManager);
      representationForward = representation.beginForward(workingManager);
      candidateContextComponentForward = candidateContext.beginForward(workingManager);
      contextComponentForward = context.beginForward(workingManager);
      Forward created =
          new Forward(
              this,
              slot,
              affineComponentForward,
              indexedComponentForward,
              representationForward,
              candidateContextComponentForward,
              contextComponentForward);
      activeForwards[slot] = created;
      return created;
    } catch (RuntimeException | Error failure) {
      if (contextComponentForward != null) {
        try {
          contextComponentForward.close();
        } catch (Throwable closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      if (candidateContextComponentForward != null) {
        try {
          candidateContextComponentForward.close();
        } catch (Throwable closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      if (representationForward != null) {
        try {
          representationForward.close();
        } catch (Throwable closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      if (indexedComponentForward != null) {
        try {
          indexedComponentForward.close();
        } catch (Throwable closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      try {
        // 構成要素開始中にはデバイス投入がないため、構築済み順伝播を逆順にキャンセルする。
        affineComponentForward.close();
      } catch (Throwable closeFailure) {
        failure.addSuppressed(closeFailure);
      }
      throw failure;
    }
  }

  /** 完了を証明できないデバイス処理を保持しているなら{@code true}を返す。 */
  public boolean hasIncompleteWork() {
    return affine.hasIncompleteWork()
        || indexedAffine.hasIncompleteWork()
        || representation.hasIncompleteWork()
        || candidateContext.hasIncompleteWork()
        || context.hasIncompleteWork();
  }

  /** 実行単位を利用不能にした最初の失敗を返す。正常なら{@code null}を返す。 */
  public Throwable failure() {
    Throwable affineFailure = affine.failure();
    if (affineFailure != null) {
      return affineFailure;
    }
    Throwable indexedFailure = indexedAffine.failure();
    if (indexedFailure != null) {
      return indexedFailure;
    }
    Throwable representationFailure = representation.failure();
    if (representationFailure != null) {
      return representationFailure;
    }
    Throwable candidateContextFailure = candidateContext.failure();
    return candidateContextFailure != null ? candidateContextFailure : context.failure();
  }

  /** 呼び出し側が同じストリーム末尾の完了を証明した後、失敗順伝播の利用権を解放する。 */
  public void releaseFailedForwardAfterCompletion() {
    Throwable failure = null;
    try {
      context.releaseFailedForwardAfterCompletion();
    } catch (Throwable closeFailure) {
      failure = closeFailure;
    }
    try {
      candidateContext.releaseFailedForwardAfterCompletion();
    } catch (Throwable closeFailure) {
      failure = addFailure(failure, closeFailure);
    }
    try {
      representation.releaseFailedForwardAfterCompletion();
    } catch (Throwable closeFailure) {
      failure = addFailure(failure, closeFailure);
    }
    try {
      indexedAffine.releaseFailedForwardAfterCompletion();
    } catch (Throwable closeFailure) {
      failure = addFailure(failure, closeFailure);
    }
    try {
      affine.releaseFailedForwardAfterCompletion();
    } catch (Throwable closeFailure) {
      failure = addFailure(failure, closeFailure);
    }
    for (Forward activeForward : activeForwards) {
      if (activeForward != null) {
        try {
          activeForward.releaseAfterCompletion();
        } catch (Throwable closeFailure) {
          failure = addFailure(failure, closeFailure);
        }
      }
    }
    rethrow(failure);
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    if (hasIncompleteWork()) {
      throw new IllegalStateException(
          "cannot close policy execution with incomplete device work", failure());
    }
    if (hasActiveForward()) {
      throw new IllegalStateException("cannot close policy execution with an active forward");
    }
    Throwable failure = null;
    try {
      context.close();
    } catch (Throwable closeFailure) {
      failure = closeFailure;
    }
    try {
      candidateContext.close();
    } catch (Throwable closeFailure) {
      failure = addFailure(failure, closeFailure);
    }
    try {
      representation.close();
    } catch (Throwable closeFailure) {
      failure = addFailure(failure, closeFailure);
    }
    try {
      indexedAffine.close();
    } catch (Throwable closeFailure) {
      failure = addFailure(failure, closeFailure);
    }
    try {
      affine.close();
    } catch (Throwable closeFailure) {
      failure = addFailure(failure, closeFailure);
    }
    if (failure == null) {
      closed = true;
    }
    rethrow(failure);
  }

  /** 一つの方策順伝播に属する各実行計画の利用権を束ねる。 */
  public static final class Forward implements DecisionCandidateForward {

    private final DecisionPolicyExecution owner;
    private final int slot;
    private final DecisionPolicyAffineExecution.Forward affine;
    private final DecisionPolicyIndexedAffineExecution.Forward indexedAffine;
    private final DecisionCandidateExecution.Forward representation;
    private final DecisionPolicyCandidateContextExecution.Forward candidateContext;
    private final DecisionPolicyContextExecution.Forward context;
    private Dependencies dependencies;
    private boolean sealed;
    private boolean poisoned;
    private boolean finished;

    private Forward(
        DecisionPolicyExecution owner,
        int slot,
        DecisionPolicyAffineExecution.Forward affine,
        DecisionPolicyIndexedAffineExecution.Forward indexedAffine,
        DecisionCandidateExecution.Forward representation,
        DecisionPolicyCandidateContextExecution.Forward candidateContext,
        DecisionPolicyContextExecution.Forward context) {
      this.owner = owner;
      this.slot = slot;
      this.affine = affine;
      this.indexedAffine = indexedAffine;
      this.representation = representation;
      this.candidateContext = candidateContext;
      this.context = context;
    }

    @Override
    public NDArray candidate(
        ParameterStore parameterStore,
        PairList<String, Object> runtimeParameters,
        NDArray actionFeatures,
        NDArray primaryTiles,
        NDArray transitions,
        NDArray playerContexts,
        NDArray stateContext) {
      return !representation.hasCandidatePrefix()
          ? affine.candidate(
              parameterStore,
              runtimeParameters,
              actionFeatures,
              primaryTiles,
              transitions,
              playerContexts,
              stateContext)
          : representation.candidate(
              parameterStore,
              runtimeParameters,
              actionFeatures,
              primaryTiles,
              transitions,
              playerContexts,
              stateContext);
    }

    @Override
    public NDArray compactCandidate(
        ParameterStore parameterStore,
        PairList<String, Object> runtimeParameters,
        NDArray actionFeatures,
        NDArray primaryTiles,
        NDArray transitions,
        NDArray playerContexts,
        NDArray states,
        NDArray stateRows) {
      return representation.compactCandidate(
          parameterStore,
          runtimeParameters,
          actionFeatures,
          primaryTiles,
          transitions,
          playerContexts,
          states,
          stateRows);
    }

    @Override
    public NDArray transition(
        ParameterStore parameterStore,
        PairList<String, Object> runtimeParameters,
        NDArray features,
        NDArray tileContexts,
        NDArray discardTiles,
        NDArray discardSafety,
        NDArray states,
        NDArray stateRows) {
      return affine.transition(
          parameterStore,
          runtimeParameters,
          features,
          tileContexts,
          discardTiles,
          discardSafety,
          states,
          stateRows);
    }

    NDArray alternative(
        ParameterStore parameterStore,
        PairList<String, Object> runtimeParameters,
        NDArray stateContext,
        NDArray candidateContexts) {
      return affine.alternative(parameterStore, runtimeParameters, stateContext, candidateContexts);
    }

    @Override
    public NDArray mixPlayerTileContexts(ParameterStore parameterStore, NDArray byPlayer) {
      return representation.mixPlayerTileContexts(parameterStore, byPlayer);
    }

    DecisionPolicyContextPool.MappedCandidateContexts candidateContexts(
        NDArray candidateEmbeddings,
        NDArray candidateScores,
        DecisionPolicyCandidates.CandidateMasks candidateMasks) {
      return candidateContext.pool(candidateEmbeddings, candidateScores, candidateMasks);
    }

    NDArray scoreRows(
        DecisionPolicyIndexedAffineExecution.Site site,
        ParameterStore parameterStore,
        NDArray state,
        NDArray baselineBranch,
        NDArray selectedBranch,
        NDArray activeRows,
        long rowCount,
        PairList<String, Object> runtimeParameters) {
      return indexedAffine.scoreRows(
          site,
          parameterStore,
          state,
          baselineBranch,
          selectedBranch,
          activeRows,
          rowCount,
          runtimeParameters);
    }

    NDArray scoreActions(
        ParameterStore parameterStore,
        NDArray state,
        NDArray baselineBranch,
        NDArray selectedBranch,
        NDArray activeActions,
        long rowCount,
        int actionCapacity,
        PairList<String, Object> runtimeParameters) {
      return indexedAffine.scoreActions(
          parameterStore,
          state,
          baselineBranch,
          selectedBranch,
          activeActions,
          rowCount,
          actionCapacity,
          runtimeParameters);
    }

    NDArray mixBinaryBranchContexts(
        DecisionPolicyContextExecution.Site site,
        NDArray baselineContext,
        NDArray selectedContext,
        NDArray selectedLogit,
        NDArray baselinePresence,
        NDArray selectedPresence) {
      return context.mixBinaryBranchContexts(
          site,
          baselineContext,
          selectedContext,
          selectedLogit,
          baselinePresence,
          selectedPresence);
    }

    /** 全実行計画の出力再利用を、返された依存オブジェクトが閉じられるまで禁止する。 */
    public AutoCloseable seal() {
      if (sealed || poisoned) {
        throw new IllegalStateException("policy forward cannot be sealed twice");
      }
      dependencies = new Dependencies();
      try {
        dependencies.add(affine.seal());
        dependencies.add(indexedAffine.seal());
        dependencies.add(representation.seal());
        dependencies.add(candidateContext.seal());
        dependencies.add(context.seal());
        sealed = true;
        return this::releaseAfterCompletion;
      } catch (RuntimeException | Error failure) {
        // 確定済み側も内部Forwardが利用権を所有している。完了前に閉じず、再利用禁止状態の資源回収へ渡す。
        poison(failure);
        throw failure;
      }
    }

    /** 最終出力完了へ到達できなかった失敗を各実行計画へ伝播する。 */
    public void poison(Throwable failure) {
      if (poisoned) {
        return;
      }
      poisoned = true;
      Throwable poisonFailure = null;
      try {
        affine.poison(failure);
      } catch (Throwable componentFailure) {
        poisonFailure = componentFailure;
      }
      try {
        indexedAffine.poison(failure);
      } catch (Throwable componentFailure) {
        poisonFailure = addFailure(poisonFailure, componentFailure);
      }
      try {
        candidateContext.poison(failure);
      } catch (Throwable componentFailure) {
        poisonFailure = addFailure(poisonFailure, componentFailure);
      }
      try {
        representation.poison(failure);
      } catch (Throwable componentFailure) {
        poisonFailure = addFailure(poisonFailure, componentFailure);
      }
      try {
        context.poison(failure);
      } catch (Throwable componentFailure) {
        poisonFailure = addFailure(poisonFailure, componentFailure);
      }
      if (poisonFailure != null) {
        failure.addSuppressed(poisonFailure);
      }
      if (!owner.hasIncompleteWork()) {
        try {
          releaseAfterCompletion();
        } catch (Throwable releaseFailure) {
          failure.addSuppressed(releaseFailure);
        }
      }
    }

    private void releaseAfterCompletion() {
      if (finished) {
        return;
      }
      if (dependencies != null) {
        dependencies.close();
        dependencies = null;
      }
      finished = true;
      owner.finish(slot, this);
    }
  }

  private static final class Dependencies implements AutoCloseable {

    private final AutoCloseable[] values;
    private int size;

    private Dependencies() {
      values = new AutoCloseable[5];
    }

    private void add(AutoCloseable value) {
      values[size++] = value;
    }

    @Override
    public void close() {
      Throwable failure = null;
      for (int index = size - 1; index >= 0; index--) {
        AutoCloseable value = values[index];
        if (value == null) {
          continue;
        }
        try {
          value.close();
          values[index] = null;
        } catch (Throwable closeFailure) {
          failure = addFailure(failure, closeFailure);
        }
      }
      rethrow(failure);
    }
  }

  private int freeSlot() {
    for (int slot = 0; slot < activeForwards.length; slot++) {
      if (activeForwards[slot] == null) {
        return slot;
      }
    }
    throw new IllegalStateException("no policy execution slot is available");
  }

  private boolean hasActiveForward() {
    for (Forward forward : activeForwards) {
      if (forward != null) {
        return true;
      }
    }
    return false;
  }

  private void finish(int slot, Forward forward) {
    if (activeForwards[slot] != forward) {
      throw new IllegalStateException("policy forward does not own its execution slot");
    }
    activeForwards[slot] = null;
  }

  private static Throwable addFailure(Throwable failure, Throwable additional) {
    if (failure == null) {
      return additional;
    }
    if (failure != additional) {
      failure.addSuppressed(additional);
    }
    return failure;
  }

  private static void rethrow(Throwable failure) {
    if (failure == null) {
      return;
    }
    if (failure instanceof RuntimeException runtimeException) {
      throw runtimeException;
    }
    if (failure instanceof Error error) {
      throw error;
    }
    throw new IllegalStateException(failure);
  }
}
