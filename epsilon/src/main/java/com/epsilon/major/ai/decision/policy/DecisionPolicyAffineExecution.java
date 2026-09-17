package com.epsilon.major.ai.decision.policy;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.nn.transformer.IdEmbedding;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.major.ai.model.EpsilonMahjongStateEncoder;

/**
 * 一つのDecision デバイスパイプラインが使う3個のアフィン変換和の実行境界。
 *
 * <p>学習ブロックとパラメーターは所有せず、凍結済みパラメーターを参照する。投入スレッドは順伝播を直列に開始でき、実行枠ごとの{@link Forward}
 * を対応するまとめたスコアのホスト転送が完了するまで保持する。
 */
public abstract class DecisionPolicyAffineExecution implements AutoCloseable {

  /**
   * 一つの方策順伝播を開始する。
   *
   * @param workingManager 最終まとめたスコアのデバイス完了まで中間テンソルを所有するバッチ管理元
   * @return この順伝播で使うAFFINE_SUM 構成部分
   */
  public abstract Forward beginForward(NDManager workingManager);

  /** 完了を証明できないデバイス処理を保持しているなら{@code true}を返す。 */
  public abstract boolean hasIncompleteWork();

  /** この実行系を利用不能にした最初の失敗を返す。正常なら{@code null}を返す。 */
  public abstract Throwable failure();

  /**
   * 呼び出し側が同じストリーム末尾の完了を証明した後、失敗順伝播が保持した利用権を解放する。
   *
   * <p>実行系の再利用不可に設定状態は解除せず、資源解放だけを可能にする。
   */
  public abstract void releaseFailedForwardAfterCompletion();

  /**
   * 一つの順伝播で生成したアフィン変換出力と、その再利用を止める利用権を所有する。
   *
   * <p>{@link #seal()}後はまとめたスコア側へ所有権を渡す。正常時の{@link #close()}は最終D2H完了後にだけ呼ばれる。
   */
  public abstract static class Forward implements AutoCloseable {

    public abstract NDArray candidate(
        ParameterStore parameterStore,
        PairList<String, Object> runtimeParameters,
        NDArray actionFeatures,
        NDArray primaryTiles,
        NDArray transitions,
        NDArray playerContexts,
        NDArray stateContext);

    public abstract NDArray transition(
        ParameterStore parameterStore,
        PairList<String, Object> runtimeParameters,
        NDArray features,
        NDArray tileContexts,
        NDArray discardTiles,
        NDArray discardSafety,
        NDArray stateContexts);

    public abstract NDArray alternative(
        ParameterStore parameterStore,
        PairList<String, Object> runtimeParameters,
        NDArray stateContext,
        NDArray candidateContexts);

    /** アフィン変換出力の所有権を最終まとめたスコアへ渡せる状態に固定する。 */
    public abstract AutoCloseable seal();

    /**
     * 最終出力の完了証明へ到達できなかった失敗を記録する。
     *
     * <p>Fusion実装は既に投入した利用権を解放せず実行系を再利用不可に設定する。EAGER実装は保持資源を持たない。
     */
    public abstract void poison(Throwable failure);
  }

  /** 既存DJL演算をそのまま使い、複数順伝播を独立して保持する基準実装を作る。 */
  static DecisionPolicyAffineExecution eager(
      EpsilonFeatureFusion candidateFusion,
      EpsilonFeatureFusion transitionFusion,
      Linear alternativeHidden,
      IdEmbedding alternativeOffset,
      NDArray alternativeIds,
      int hiddenSize,
      int executionSlots,
      NDManager manager,
      ParameterStore parameterStore,
      DataType expectedDataType,
      boolean packedTransitionEnabled) {
    if (executionSlots <= 0) {
      throw new IllegalArgumentException("executionSlots must be positive");
    }
    return new Eager(
        candidateFusion,
        transitionFusion,
        alternativeHidden,
        alternativeOffset,
        alternativeIds,
        hiddenSize,
        executionSlots,
        packedTransitionEnabled
            ? new PackedTransitionProjection(
                manager, parameterStore, transitionFusion, expectedDataType)
            : null);
  }

  private static final class Eager extends DecisionPolicyAffineExecution {

    private final EagerForward[] activeForwards;
    private final EpsilonFeatureFusion candidateFusion;
    private final EpsilonFeatureFusion transitionFusion;
    private final Linear alternativeHidden;
    private final IdEmbedding alternativeOffset;
    private final NDArray alternativeIds;
    private final int hiddenSize;
    private final PackedTransitionProjection packedTransition;
    private boolean closed;

    private Eager(
        EpsilonFeatureFusion candidateFusion,
        EpsilonFeatureFusion transitionFusion,
        Linear alternativeHidden,
        IdEmbedding alternativeOffset,
        NDArray alternativeIds,
        int hiddenSize,
        int executionSlots,
        PackedTransitionProjection packedTransition) {
      if (executionSlots <= 0) {
        throw new IllegalArgumentException("executionSlots must be positive");
      }
      this.candidateFusion = candidateFusion;
      this.transitionFusion = transitionFusion;
      this.alternativeHidden = alternativeHidden;
      this.alternativeOffset = alternativeOffset;
      this.alternativeIds = alternativeIds;
      this.hiddenSize = hiddenSize;
      this.packedTransition = packedTransition;
      activeForwards = new EagerForward[executionSlots];
    }

    @Override
    public Forward beginForward(NDManager workingManager) {
      if (closed) {
        throw new IllegalStateException("affine execution is closed");
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
        throw new IllegalStateException("cannot close affine execution with an active forward");
      }
      if (packedTransition != null) {
        packedTransition.close();
      }
      closed = true;
    }

    private int freeSlot() {
      for (int slot = 0; slot < activeForwards.length; slot++) {
        if (activeForwards[slot] == null) {
          return slot;
        }
      }
      throw new IllegalStateException("no affine execution slot is available");
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
        throw new IllegalStateException("affine forward does not own its execution slot");
      }
      activeForwards[slot] = null;
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
    public NDArray candidate(
        ParameterStore parameterStore,
        PairList<String, Object> runtimeParameters,
        NDArray actionFeatures,
        NDArray primaryTiles,
        NDArray transitions,
        NDArray playerContexts,
        NDArray stateContext) {
      return EpsilonMahjongStateEncoder.silu(
          owner.candidateFusion.fuse(
              parameterStore,
              false,
              runtimeParameters,
              actionFeatures,
              primaryTiles,
              transitions,
              playerContexts,
              stateContext));
    }

    @Override
    public NDArray transition(
        ParameterStore parameterStore,
        PairList<String, Object> runtimeParameters,
        NDArray features,
        NDArray tileContexts,
        NDArray discardTiles,
        NDArray discardSafety,
        NDArray stateContexts) {
      if (owner.packedTransition != null) {
        return owner.packedTransition.project(
            features, tileContexts, discardTiles, discardSafety, stateContexts);
      }
      return EpsilonMahjongStateEncoder.silu(
          owner.transitionFusion.fuse(
              parameterStore,
              false,
              runtimeParameters,
              features,
              tileContexts,
              discardTiles,
              discardSafety,
              stateContexts));
    }

    @Override
    public NDArray alternative(
        ParameterStore parameterStore,
        PairList<String, Object> runtimeParameters,
        NDArray stateContext,
        NDArray candidateContexts) {
      NDManager batchManager = stateContext.getManager();
      NDArray offsets;
      try (NDManager embeddingScope = batchManager.newSubManager()) {
        embeddingScope.tempAttachAll(owner.alternativeIds);
        offsets =
            owner
                .alternativeOffset
                .forward(parameterStore, new NDList(owner.alternativeIds), false, runtimeParameters)
                .singletonOrThrow();
        batchManager.attachAll(offsets);
      }
      return EpsilonMahjongStateEncoder.silu(
          EpsilonPartitionedLinear.apply(
                  owner.alternativeHidden, parameterStore, false, stateContext, candidateContexts)
              .add(offsets.reshape(1, DecisionAlternative.NETWORK_SIZE, owner.hiddenSize)));
    }

    @Override
    public AutoCloseable seal() {
      if (finished || sealed) {
        throw new IllegalStateException("affine forward cannot be sealed twice");
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

  /** 凍結推論の遷移別入力元を必要な要素数だけのバッファへ連結し、二つのGEMMで射影する。 */
  static final class PackedTransitionProjection implements AutoCloseable {

    private final NDManager resourceManager;
    private final NDArray prefixWeight;
    private final NDArray stateWeight;
    private final NDArray bias;
    private final int prefixComponents;
    private final int hiddenSize;
    private final DataType dataType;
    private boolean closed;

    PackedTransitionProjection(
        NDManager manager,
        ParameterStore parameterStore,
        EpsilonFeatureFusion transitionFusion,
        DataType expectedDataType) {
      resourceManager = manager.newSubManager();
      prefixComponents = transitionFusion.fusedPrefixComponents();
      if (prefixComponents <= 0 || transitionFusion.componentCount() != prefixComponents + 1) {
        resourceManager.close();
        throw new IllegalArgumentException(
            "packed transition projection requires one state component after the prefix");
      }
      hiddenSize = transitionFusion.outputSize();
      try {
        NDList weights = new NDList(prefixComponents);
        for (int component = 0; component < prefixComponents; component++) {
          weights.add(
              parameterStore.getValue(
                  weight(transitionFusion.projection(component)), manager.getDevice(), false));
        }
        prefixWeight = NDArrays.concat(weights, 1);
        prefixWeight.attach(resourceManager);
        dataType = prefixWeight.getDataType();
        if (dataType != expectedDataType) {
          throw new IllegalArgumentException(
              "packed transition weight dtype differs from compute precision: "
                  + dataType
                  + "/"
                  + expectedDataType);
        }
        stateWeight =
            parameterStore.getValue(
                weight(transitionFusion.projection(prefixComponents)), manager.getDevice(), false);
        Parameter biasParameter = transitionFusion.projection(0).getDirectParameters().get("bias");
        bias = parameterStore.getValue(biasParameter, manager.getDevice(), false);
        if (stateWeight.getDataType() != dataType || bias.getDataType() != dataType) {
          throw new IllegalArgumentException(
              "packed transition parameters must use one floating-point data type");
        }
      } catch (RuntimeException | Error constructionFailure) {
        resourceManager.close();
        throw constructionFailure;
      }
    }

    NDArray project(NDArray... components) {
      if (closed) {
        throw new IllegalStateException("packed transition projection is closed");
      }
      if (components.length != prefixComponents + 1) {
        throw new IllegalArgumentException(
            "packed transition component count must be "
                + (prefixComponents + 1)
                + ": "
                + components.length);
      }
      int lastAxis = components[0].getShape().dimension() - 1;
      NDList prefixInputs = new NDList(prefixComponents);
      for (int component = 0; component < prefixComponents; component++) {
        prefixInputs.add(components[component]);
      }
      NDArray packed = NDArrays.concatToType(prefixInputs, lastAxis, dataType);
      NDArray projected = Linear.linear(packed, prefixWeight, null).singletonOrThrow();
      NDArray state =
          Linear.linear(components[prefixComponents], stateWeight, null).singletonOrThrow();
      return addBiasAndStateProjectionAndActivate(projected, bias, state, hiddenSize);
    }

    @Override
    public void close() {
      if (!closed) {
        closed = true;
        resourceManager.close();
      }
    }
  }

  /** バイアス、対応する状態射影、SiLUを段階丸めのまま所有済みバッファへ適用する。 */
  static NDArray addBiasAndStateProjectionAndActivate(
      NDArray projected, NDArray bias, NDArray state, int hiddenSize) {
    Shape shape = projected.getShape();
    long rows = shape.size() / hiddenSize;
    NDArrays.addBiasAndBroadcastResidualToOwnedAndSilu(
        projected.reshape(rows, 1, hiddenSize), bias, state.reshape(rows, 1, hiddenSize));
    return projected;
  }

  private static Parameter weight(Linear linear) {
    return linear.getDirectParameters().get("weight");
  }
}
