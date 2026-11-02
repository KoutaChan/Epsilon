package com.epsilon.nano.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.ai.model.EpsilonMaskedRows;
import com.epsilon.nano.ai.model.EpsilonMahjongStateEncoder;
import java.util.function.Supplier;

/** 候補射影と牌別文脈の共有推論計画。利用権は最終D2H完了まで保持する。 */
public final class DecisionCandidateExecution implements AutoCloseable {
  private final EpsilonFeatureFusion candidateFusion;
  private final EpsilonFeatureFusion transitionFusion;
  private final DecisionPolicyCandidatePrefixExecution prefix;
  private final DecisionPolicyPlayerTileContextExecution playerTile;
  private final Supplier<DecisionPolicyAffineExecution.PackedTransitionProjection>
      transitionFactory;
  private DecisionPolicyAffineExecution.PackedTransitionProjection packedTransition;
  private boolean closed;

  DecisionCandidateExecution(
      EpsilonFeatureFusion candidateFusion,
      EpsilonFeatureFusion transitionFusion,
      DecisionPolicyCandidatePrefixExecution prefix,
      DecisionPolicyPlayerTileContextExecution playerTile,
      Supplier<DecisionPolicyAffineExecution.PackedTransitionProjection> transitionFactory) {
    this.candidateFusion = candidateFusion;
    this.transitionFusion = transitionFusion;
    this.prefix = prefix;
    this.playerTile = playerTile;
    this.transitionFactory = transitionFactory;
  }

  public Forward beginForward(NDManager manager) {
    if (closed) throw new IllegalStateException("Candidate execution is closed");
    var prefixForward = prefix == null ? null : prefix.beginForward(manager);
    try {
      return new Forward(prefixForward, playerTile.beginForward(manager));
    } catch (RuntimeException | Error failure) {
      if (prefixForward != null) {
        try {
          prefixForward.close();
        } catch (Throwable closeFailure) {
          failure.addSuppressed(closeFailure);
        }
      }
      throw failure;
    }
  }

  public boolean hasIncompleteWork() {
    return (prefix != null && prefix.hasIncompleteWork()) || playerTile.hasIncompleteWork();
  }

  public Throwable failure() {
    Throwable failure = prefix == null ? null : prefix.failure();
    return failure == null ? playerTile.failure() : failure;
  }

  public void releaseFailedForwardAfterCompletion() {
    Throwable failure = null;
    try {
      playerTile.releaseFailedForwardAfterCompletion();
    } catch (Throwable problem) {
      failure = problem;
    }
    if (prefix != null) {
      try {
        prefix.releaseFailedForwardAfterCompletion();
      } catch (Throwable problem) {
        failure = addFailure(failure, problem);
      }
    }
    rethrow(failure);
  }

  @Override
  public void close() {
    if (closed) return;
    Throwable failure = null;
    try {
      playerTile.close();
    } catch (Throwable problem) {
      failure = problem;
    }
    if (prefix != null) {
      try {
        prefix.close();
      } catch (Throwable problem) {
        failure = addFailure(failure, problem);
      }
    }
    if (failure == null && packedTransition != null) {
      try {
        packedTransition.close();
      } catch (Throwable problem) {
        failure = problem;
      }
    }
    if (failure == null) closed = true;
    rethrow(failure);
  }

  /** 一順伝播で使う候補表現計画の利用権。Nanoの方策計画も同じ演算を借用する。 */
  public final class Forward implements DecisionCandidateForward, AutoCloseable {
    private final DecisionPolicyCandidatePrefixExecution.Forward prefixForward;
    private final DecisionPolicyPlayerTileContextExecution.Forward playerTileForward;
    private AutoCloseable prefixDependency;
    private AutoCloseable playerTileDependency;
    private boolean sealed;
    private boolean poisoned;
    private boolean finished;

    private Forward(
        DecisionPolicyCandidatePrefixExecution.Forward prefixForward,
        DecisionPolicyPlayerTileContextExecution.Forward playerTileForward) {
      this.prefixForward = prefixForward;
      this.playerTileForward = playerTileForward;
    }

    boolean hasCandidatePrefix() {
      return prefixForward != null;
    }

    @Override
    public NDArray candidate(
        ParameterStore parameters,
        PairList<String, Object> runtimeParameters,
        NDArray actionFeatures,
        NDArray primaryTiles,
        NDArray transitions,
        NDArray playerContexts,
        NDArray stateContext) {
      requireActive();
      return prefixForward == null
          ? EpsilonMahjongStateEncoder.silu(
              candidateFusion.fuse(
                  parameters,
                  false,
                  runtimeParameters,
                  actionFeatures,
                  primaryTiles,
                  transitions,
                  playerContexts,
                  stateContext))
          : prefixForward.project(
              parameters,
              runtimeParameters,
              actionFeatures,
              primaryTiles,
              transitions,
              playerContexts,
              stateContext);
    }

    @Override
    public NDArray compactCandidate(
        ParameterStore parameters,
        PairList<String, Object> runtimeParameters,
        NDArray actionFeatures,
        NDArray primaryTiles,
        NDArray transitions,
        NDArray playerContexts,
        NDArray states,
        NDArray stateRows) {
      requireActive();
      return prefixForward == null
          ? EpsilonMahjongStateEncoder.silu(
              candidateFusion.fuseIndexedState(
                  parameters,
                  runtimeParameters,
                  states,
                  stateRows,
                  actionFeatures,
                  primaryTiles,
                  transitions,
                  playerContexts))
          : prefixForward.projectCompact(
              parameters,
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
        ParameterStore parameters,
        PairList<String, Object> runtimeParameters,
        NDArray features,
        NDArray tileContexts,
        NDArray discardTiles,
        NDArray discardSafety,
        NDArray states,
        NDArray stateRows) {
      requireActive();
      if (transitionFactory != null) {
        // Nanoの方策実行が別途所有する場合は作らず、この共有表現が初めて使う時だけ確保する。
        if (packedTransition == null) packedTransition = transitionFactory.get();
        return packedTransition.project(
            stateRows, features, tileContexts, discardTiles, discardSafety, states);
      }
      NDArray stateContexts =
          EpsilonMaskedRows.gather(states, stateRows)
              .reshape(features.getShape().get(0), 1, 1, transitionFusion.outputSize());
      return EpsilonMahjongStateEncoder.silu(
          transitionFusion.fuse(
              parameters,
              false,
              runtimeParameters,
              features,
              tileContexts,
              discardTiles,
              discardSafety,
              stateContexts));
    }

    @Override
    public NDArray mixPlayerTileContexts(ParameterStore parameters, NDArray byPlayer) {
      requireActive();
      return playerTileForward.mix(parameters, byPlayer);
    }

    public AutoCloseable seal() {
      requireActive();
      try {
        if (prefixForward != null) prefixDependency = prefixForward.seal();
        playerTileDependency = playerTileForward.seal();
        sealed = true;
        return this;
      } catch (RuntimeException | Error failure) {
        poison(failure);
        throw failure;
      }
    }

    public void poison(Throwable failure) {
      if (poisoned) return;
      poisoned = true;
      if (prefixForward != null) {
        try {
          prefixForward.poison(failure);
        } catch (Throwable problem) {
          failure.addSuppressed(problem);
        }
      }
      try {
        playerTileForward.poison(failure);
      } catch (Throwable problem) {
        failure.addSuppressed(problem);
      }
    }

    private void requireActive() {
      if (finished || sealed || poisoned)
        throw new IllegalStateException("Candidate forward is no longer active");
    }

    @Override
    public void close() {
      if (finished) return;
      Throwable failure = null;
      try {
        if (playerTileDependency != null) playerTileDependency.close();
        else playerTileForward.close();
        playerTileDependency = null;
      } catch (Throwable problem) {
        failure = problem;
      }
      if (prefixForward != null) {
        try {
          if (prefixDependency != null) prefixDependency.close();
          else prefixForward.close();
          prefixDependency = null;
        } catch (Throwable problem) {
          failure = addFailure(failure, problem);
        }
      }
      if (failure == null) finished = true;
      rethrow(failure);
    }
  }

  private static Throwable addFailure(Throwable failure, Throwable additional) {
    if (failure == null) return additional;
    if (failure != additional) failure.addSuppressed(additional);
    return failure;
  }

  private static void rethrow(Throwable failure) {
    if (failure instanceof RuntimeException problem) throw problem;
    if (failure instanceof Error problem) throw problem;
    if (failure != null) throw new IllegalStateException(failure);
  }
}
