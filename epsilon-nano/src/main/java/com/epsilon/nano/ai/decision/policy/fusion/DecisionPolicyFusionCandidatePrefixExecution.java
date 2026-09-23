package com.epsilon.nano.ai.decision.policy.fusion;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.nano.ai.decision.policy.DecisionPolicyCandidatePrefixExecution;
import com.epsilon.nano.ai.decision.policy.EpsilonFeatureFusion;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 候補アフィン変換の先頭構成部分を、実際の候補数に合わせて連結する。
 *
 * <p>concatToTypeはOUTPUT_PACKと同じ型変換付きカーネルを使う。最大候補数分の永続出力は持たず、
 * 出力と入力の寿命を順伝播の管理元へ結び付ける。後段は従来と同じ一回のGEMMであり、 BF16への変換位置、バイアス、状態射影の加算位置を変えない。
 */
public final class DecisionPolicyFusionCandidatePrefixExecution
    extends DecisionPolicyCandidatePrefixExecution {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(DecisionPolicyFusionCandidatePrefixExecution.class);

  private final NDManager resourceManager;
  private final Linear stateProjection;
  private final NDArray prefixWeight;
  private final NDArray prefixBias;
  private final int prefixComponents;
  private final DataType outputDataType;
  private final FusionForward[] activeForwards;
  private Throwable failure;
  private boolean incomplete;
  private boolean closed;

  public DecisionPolicyFusionCandidatePrefixExecution(
      NDManager manager,
      ParameterStore parameterStore,
      EpsilonFeatureFusion candidateFusion,
      DataType expectedDataType,
      int maximumBatch,
      int executionSlots) {
    if (executionSlots <= 0) {
      throw new IllegalArgumentException("executionSlots must be positive");
    }
    prefixComponents = candidateFusion.fusedPrefixComponents();
    if (prefixComponents <= 0 || candidateFusion.componentCount() != prefixComponents + 1) {
      throw new IllegalArgumentException(
          "candidate prefix packing requires one broadcast component after the prefix");
    }
    stateProjection = candidateFusion.projection(prefixComponents);
    activeForwards = new FusionForward[executionSlots];
    resourceManager = manager.newSubManager();
    try {
      NDList weights = new NDList(prefixComponents);
      for (int component = 0; component < prefixComponents; component++) {
        weights.add(
            parameterStore.getValue(
                candidateFusion.projection(component).getDirectParameters().get("weight"),
                manager.getDevice(),
                false));
      }
      prefixWeight = NDArrays.concat(weights, 1);
      prefixWeight.attach(resourceManager);
      outputDataType = prefixWeight.getDataType();
      if (outputDataType != expectedDataType) {
        throw new IllegalArgumentException(
            "candidate prefix weight dtype differs from compute precision: "
                + outputDataType
                + "/"
                + expectedDataType);
      }
      prefixBias =
          parameterStore.getValue(
              candidateFusion.projection(0).getDirectParameters().get("bias"),
              manager.getDevice(),
              false);
      if (prefixBias.getDataType() != outputDataType) {
        throw new IllegalArgumentException("candidate prefix bias dtype differs from weight dtype");
      }
      LOGGER.info(
          "Decision candidate prefix pack: maximumBatch={}, slots={}, outputDtype={}, "
              + "outputWidth={}, storage=ACTIVE_SHAPE, retainedOutputBytes=0",
          maximumBatch,
          executionSlots,
          outputDataType,
          prefixWeight.getShape().get(1));
    } catch (RuntimeException | Error constructionFailure) {
      closeResource(constructionFailure, resourceManager);
      throw constructionFailure;
    }
  }

  @Override
  public Forward beginForward(NDManager workingManager) {
    if (closed) {
      throw new IllegalStateException("candidate prefix pack is closed");
    }
    if (failure != null) {
      throw new IllegalStateException("candidate prefix pack is poisoned", failure);
    }
    for (int slot = 0; slot < activeForwards.length; slot++) {
      if (activeForwards[slot] == null) {
        FusionForward forward = new FusionForward(this, slot, workingManager);
        activeForwards[slot] = forward;
        return forward;
      }
    }
    throw new IllegalStateException("no candidate prefix pack slot is available");
  }

  @Override
  public boolean hasIncompleteWork() {
    return incomplete;
  }

  @Override
  public Throwable failure() {
    return failure;
  }

  @Override
  public void releaseFailedForwardAfterCompletion() {
    if (incomplete) {
      for (FusionForward forward : activeForwards) {
        if (forward != null) {
          forward.close();
        }
      }
      incomplete = false;
    }
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    if (incomplete) {
      throw new IllegalStateException(
          "cannot close candidate prefix pack with incomplete device work", failure);
    }
    for (FusionForward forward : activeForwards) {
      if (forward != null) {
        throw new IllegalStateException(
            "cannot close candidate prefix pack with an active forward");
      }
    }
    resourceManager.close();
    closed = true;
  }

  /** 所有済み候補射影へ状態射影を加算し、同じバッファへSiLUを適用する。 */
  public static NDArray addStateProjectionAndActivate(NDArray projected, NDArray state) {
    return NDArrays.addBroadcastResidualToOwnedAndSilu(projected, state);
  }

  private void poison(Throwable cause) {
    if (failure == null) {
      failure = cause;
    }
    incomplete = true;
  }

  private static final class FusionForward extends Forward {

    private final DecisionPolicyFusionCandidatePrefixExecution owner;
    private final int slot;
    private NDManager workingManager;
    private boolean executed;
    private boolean sealed;
    private boolean finished;

    private FusionForward(
        DecisionPolicyFusionCandidatePrefixExecution owner, int slot, NDManager workingManager) {
      this.owner = owner;
      this.slot = slot;
      this.workingManager = workingManager;
    }

    @Override
    public NDArray project(
        ParameterStore parameterStore,
        PairList<String, Object> runtimeParameters,
        NDArray actionFeatures,
        NDArray primaryTiles,
        NDArray transitions,
        NDArray playerContexts,
        NDArray stateContext) {
      if (finished || sealed || executed) {
        throw new IllegalStateException("candidate prefix pack cannot execute twice");
      }
      NDArray[] components = {
        actionFeatures, primaryTiles, transitions, playerContexts, stateContext
      };
      try {
        NDList inputs = new NDList(owner.prefixComponents);
        long rows = actionFeatures.getShape().get(0);
        long actions = actionFeatures.getShape().get(1);
        for (int component = 0; component < components.length; component++) {
          components[component].attach(workingManager);
          if (component < owner.prefixComponents) {
            inputs.add(components[component].reshape(rows * actions, -1));
          }
        }
        NDArray projected;
        try (NDArray packed = NDArrays.concatToType(inputs, 1, owner.outputDataType)) {
          packed.attach(workingManager);
          projected =
              Linear.linear(packed, owner.prefixWeight, owner.prefixBias)
                  .singletonOrThrow()
                  .reshape(rows, actions, -1);
        }
        NDArray state =
            owner
                .stateProjection
                .forward(parameterStore, new NDList(stateContext), false, runtimeParameters)
                .singletonOrThrow();
        NDArray result = addStateProjectionAndActivate(projected, state);
        executed = true;
        return result;
      } catch (RuntimeException | Error submitFailure) {
        owner.poison(submitFailure);
        throw submitFailure;
      }
    }

    @Override
    public AutoCloseable seal() {
      if (finished || sealed || !executed) {
        throw new IllegalStateException("candidate prefix pack has no completed submission");
      }
      sealed = true;
      return this;
    }

    @Override
    public void poison(Throwable failure) {
      if (!finished) {
        owner.poison(failure);
      }
    }

    @Override
    public void close() {
      if (!finished) {
        workingManager = null;
        finished = true;
        owner.activeForwards[slot] = null;
      }
    }
  }

  private static void closeResource(Throwable failure, AutoCloseable resource) {
    try {
      resource.close();
    } catch (Throwable closeFailure) {
      failure.addSuppressed(closeFailure);
    }
  }
}
