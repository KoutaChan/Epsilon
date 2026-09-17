package com.epsilon.pico.ai.model.fusion;

import ai.djl.engine.fusion.FusionCompilationReport;
import ai.djl.engine.fusion.FusionCompileConfig;
import ai.djl.engine.fusion.FusionCompiler;
import ai.djl.engine.fusion.FusionConstantBindings;
import ai.djl.engine.fusion.FusionExecutable;
import ai.djl.engine.fusion.FusionInvocation;
import ai.djl.engine.fusion.FusionOutputLease;
import ai.djl.engine.fusion.FusionPlan;
import ai.djl.engine.fusion.FusionRecipe;
import ai.djl.engine.fusion.FusionSession;
import ai.djl.engine.fusion.FusionSessionConfig;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import com.epsilon.pico.ai.model.EpsilonMahjongStateEncoder;
import com.epsilon.pico.ai.model.EpsilonMahjongStateReadout;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 1つのデバイスで、方策・価値関数それぞれの特徴集約と出力バッファを管理する。
 *
 * <p>入力の先頭トークンをクエリの元としてネイティブ演算内で参照し、中間の複製を避ける。方策だけの推論では1つ、学習側の対局生成では方策と価値を一緒に計算する。後続の計算と最終結果のホスト転送が終わるまで出力を保持する。
 */
public final class EpsilonMahjongStateReadoutGroupExecution implements AutoCloseable {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(EpsilonMahjongStateReadoutGroupExecution.class);
  private static final float LAYER_NORM_EPSILON = 1.0e-5f;
  // DJLの通常autocast LayerNormが返す型。後段射影の低精度データ型とは独立である。
  private static final DataType MEMORY_DATA_TYPE = DataType.FLOAT32;
  private static final DataType MASK_DATA_TYPE = DataType.FLOAT32;

  private final FusionRecipe.Dimension batch;
  private final FusionRecipe.Input memoryInput;
  private final FusionRecipe.Input maskInput;
  private final FusionRecipe.Output policyOutput;
  private final FusionRecipe.Output valueOutput;
  private final Forward[] activeForwards;
  private FusionPlan plan;
  private FusionExecutable executable;
  private FusionSession session;
  private Throwable failure;
  private boolean unknownSubmission;
  private boolean incomplete;
  private boolean closed;

  private EpsilonMahjongStateReadoutGroupExecution(
      FusionRecipe.Dimension batch,
      FusionRecipe.Input memoryInput,
      FusionRecipe.Input maskInput,
      FusionRecipe.Output policyOutput,
      FusionRecipe.Output valueOutput,
      FusionPlan plan,
      FusionExecutable executable,
      FusionSession session,
      int executionSlots) {
    this.batch = batch;
    this.memoryInput = memoryInput;
    this.maskInput = maskInput;
    this.policyOutput = policyOutput;
    this.valueOutput = valueOutput;
    this.plan = plan;
    this.executable = executable;
    this.session = session;
    activeForwards = new Forward[executionSlots];
  }

  /**
   * 重みを固定した特徴量の集約を結び付け、一つまたは二つの特徴量の集約を実行する永続実行計画を作る。
   *
   * @param manager 実行計画・セッション・出力循環バッファを所有するGPU 管理元
   * @param maximumBatch 実行計画が受理する最大バッチ
   * @param executionSlots 同時に保持する未回収順伝播数
   * @param policyParameters 方策特徴量の集約の名前付きパラメータービュー
   * @param valueParameters 価値特徴量の集約のパラメータービュー。方策のみでは{@code null}
   * @return パイプライン-局所的な状態-特徴量の集約 Fusion実行境界
   */
  public static EpsilonMahjongStateReadoutGroupExecution create(
      NDManager manager,
      int maximumBatch,
      int executionSlots,
      EpsilonMahjongStateReadout.FrozenParameterView policyParameters,
      EpsilonMahjongStateReadout.FrozenParameterView valueParameters) {
    if (!manager.getDevice().isGpu()) {
      throw new UnsupportedOperationException("stateReadout=FUSION requires a GPU device");
    }
    int hiddenSize = policyParameters.hiddenSize();
    if (valueParameters != null && valueParameters.hiddenSize() != hiddenSize) {
      throw new IllegalArgumentException("Policy and Value readouts must use one hidden width");
    }

    List<NDArray> constants = new ArrayList<>(valueParameters == null ? 17 : 34);
    FusionRecipe.Builder builder = FusionRecipe.builder("epsilon-mahjong-state-readouts");
    FusionRecipe.Dimension batch = builder.addDimension("batch", maximumBatch);
    FusionRecipe.Input memoryInput =
        builder.addInput(
            "memory",
            FusionRecipe.TensorSpec.of(
                MEMORY_DATA_TYPE,
                batch,
                EpsilonMahjongStateEncoder.ENTITY_TOKEN_COUNT,
                hiddenSize));
    FusionRecipe.Input maskInput =
        builder.addInput(
            "validMask",
            FusionRecipe.TensorSpec.of(
                MASK_DATA_TYPE, batch, EpsilonMahjongStateEncoder.ENTITY_TOKEN_COUNT));
    FusionRecipe.SingleQueryCrossAttentionReadoutGroupBuilder groupBuilder =
        builder
            .singleQueryCrossAttentionReadoutGroup(
                "readouts",
                memoryInput,
                memoryInput,
                maskInput,
                EpsilonMahjongStateEncoder.ATTENTION_HEADS)
            .optQueryIndex(0)
            .optEpsilon(LAYER_NORM_EPSILON);
    appendReadout(builder, groupBuilder, constants, "policy", policyParameters);
    if (valueParameters != null) {
      appendReadout(builder, groupBuilder, constants, "value", valueParameters);
    }
    FusionRecipe.SingleQueryCrossAttentionReadoutGroup group = groupBuilder.build();
    FusionRecipe.Output policyOutput = builder.addOutput("policyState", group.getReadoutState(0));
    FusionRecipe.Output valueOutput =
        valueParameters == null ? null : builder.addOutput("valueState", group.getReadoutState(1));
    FusionRecipe recipe = builder.build();

    FusionConstantBindings.Builder bindings = FusionConstantBindings.builder(recipe);
    for (int index = 0; index < constants.size(); index++) {
      bindings.bind(recipe.getConstants().get(index), constants.get(index));
    }

    FusionCompiler compiler = manager.getEngine().newFusionCompiler(manager.getDevice());
    FusionPlan plan = null;
    FusionExecutable executable = null;
    FusionSession session = null;
    try {
      plan = compiler.prepare(recipe, FusionCompileConfig.defaults());
      FusionCompilationReport report = plan.getCompilationReport();
      if (!report.isNativeOnly()) {
        throw new IllegalStateException("state readout plan is not native-only");
      }
      executable = plan.bind(bindings.build());
      session =
          executable.newSession(
              manager, FusionSessionConfig.builder().optOutputSlotCount(executionSlots).build());
      int readoutCount = valueParameters == null ? 1 : 2;
      LOGGER.info(
          "Decision state readout fusion: readouts={}, maximumBatch={}, slots={}, "
              + "executableStorageBytes={}, retainedSessionStorageBytes={}, "
              + "requiredExecutionLaneStorageBytes={}",
          readoutCount,
          maximumBatch,
          executionSlots,
          report.getExecutableStorageBytes(),
          report.getRetainedSessionStorageBytes(executionSlots),
          report.getRequiredExecutionLaneStorageBytes());
      return new EpsilonMahjongStateReadoutGroupExecution(
          batch,
          memoryInput,
          maskInput,
          policyOutput,
          valueOutput,
          plan,
          executable,
          session,
          executionSlots);
    } catch (RuntimeException | Error constructionFailure) {
      closeResource(constructionFailure, session);
      closeResource(constructionFailure, executable);
      closeResource(constructionFailure, plan);
      throw constructionFailure;
    }
  }

  /** 名前付きのパラメータービューを処理構成定数へ宣言し、従来と同じ順序で特徴量の集約を追加する。 */
  private static void appendReadout(
      FusionRecipe.Builder builder,
      FusionRecipe.SingleQueryCrossAttentionReadoutGroupBuilder group,
      List<NDArray> constants,
      String prefix,
      EpsilonMahjongStateReadout.FrozenParameterView parameters) {
    group.addReadout(
        addConstant(builder, constants, prefix + "QuerySeedWeight", parameters.querySeedWeight()),
        addConstant(builder, constants, prefix + "QuerySeedBias", parameters.querySeedBias()),
        addConstant(builder, constants, prefix + "QueryWeight", parameters.queryWeight()),
        addConstant(builder, constants, prefix + "QueryBias", parameters.queryBias()),
        addConstant(builder, constants, prefix + "KeyValueWeight", parameters.keyValueWeight()),
        addConstant(builder, constants, prefix + "ContextWeight", parameters.contextWeight()),
        addConstant(builder, constants, prefix + "ContextBias", parameters.contextBias()),
        addConstant(builder, constants, prefix + "QueryNormWeight", parameters.queryNormWeight()),
        addConstant(builder, constants, prefix + "QueryNormBias", parameters.queryNormBias()),
        addConstant(
            builder,
            constants,
            prefix + "FeedForwardNormWeight",
            parameters.feedForwardNormWeight()),
        addConstant(
            builder, constants, prefix + "FeedForwardNormBias", parameters.feedForwardNormBias()),
        addConstant(
            builder,
            constants,
            prefix + "FeedForwardExpansionWeight",
            parameters.feedForwardExpansionWeight()),
        addConstant(
            builder,
            constants,
            prefix + "FeedForwardExpansionBias",
            parameters.feedForwardExpansionBias()),
        addConstant(
            builder,
            constants,
            prefix + "FeedForwardProjectionWeight",
            parameters.feedForwardProjectionWeight()),
        addConstant(
            builder,
            constants,
            prefix + "FeedForwardProjectionBias",
            parameters.feedForwardProjectionBias()),
        addConstant(builder, constants, prefix + "OutputNormWeight", parameters.outputNormWeight()),
        addConstant(builder, constants, prefix + "OutputNormBias", parameters.outputNormBias()));
  }

  /** ブロック所有のNDArray参照を複製せず定数対応付け順へ記録する。 */
  private static FusionRecipe.Constant addConstant(
      FusionRecipe.Builder builder, List<NDArray> constants, String name, NDArray value) {
    constants.add(value);
    return builder.addConstant(
        name, FusionRecipe.TensorSpec.fixed(value.getDataType(), value.getShape().getShape()));
  }

  /** 空き循環バッファ位置を予約し、一つのネットワーク順伝播を開始する。 */
  public Forward beginForward(NDManager workingManager) {
    if (closed) {
      throw new IllegalStateException("state readout execution is closed");
    }
    if (failure != null) {
      throw new IllegalStateException("state readout execution is poisoned", failure);
    }
    for (int slot = 0; slot < activeForwards.length; slot++) {
      if (activeForwards[slot] == null) {
        Forward forward = new Forward(this, slot, workingManager);
        activeForwards[slot] = forward;
        return forward;
      }
    }
    throw new IllegalStateException("no state readout slot is available");
  }

  public boolean hasIncompleteWork() {
    return incomplete || unknownSubmission;
  }

  public Throwable failure() {
    return failure;
  }

  public void releaseFailedForwardAfterCompletion() {
    if (!hasIncompleteWork()) {
      return;
    }
    Throwable releaseFailure = null;
    unknownSubmission = false;
    for (Forward forward : activeForwards) {
      if (forward == null || !forward.poisoned) {
        continue;
      }
      try {
        forward.releaseAfterCompletion();
      } catch (Throwable closeFailure) {
        if (failure == null) {
          failure = closeFailure;
        }
        releaseFailure = addFailure(releaseFailure, closeFailure);
      }
    }
    incomplete = hasPoisonedForward() || unknownSubmission;
    rethrow(releaseFailure);
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    if (hasIncompleteWork()) {
      throw new IllegalStateException("state readout execution has incomplete device work");
    }
    for (Forward forward : activeForwards) {
      if (forward != null) {
        throw new IllegalStateException("state readout execution has an active forward");
      }
    }
    Throwable closeFailure = null;
    if (session != null) {
      try {
        session.close();
        session = null;
      } catch (Throwable failure) {
        closeFailure = failure;
      }
    }
    if (session == null && executable != null) {
      try {
        executable.close();
        executable = null;
      } catch (Throwable failure) {
        closeFailure = addFailure(closeFailure, failure);
      }
    }
    if (session == null && executable == null && plan != null) {
      try {
        plan.close();
        plan = null;
      } catch (Throwable failure) {
        closeFailure = addFailure(closeFailure, failure);
      }
    }
    closed = session == null && executable == null && plan == null;
    rethrow(closeFailure);
  }

  private void finish(int slot, Forward forward) {
    if (activeForwards[slot] != forward) {
      throw new IllegalStateException("state readout forward does not own its slot");
    }
    activeForwards[slot] = null;
  }

  private boolean hasPoisonedForward() {
    for (Forward forward : activeForwards) {
      if (forward != null && forward.poisoned) {
        return true;
      }
    }
    return false;
  }

  /** 一回の特徴量の集約グループ実行から得た目的別状態。 */
  public record Readouts(NDArray policyStateEmbedding, NDArray valueStateEmbedding) {}

  /** 一つのネットワーク順伝播が保持する特徴量の集約出力利用権。 */
  public static final class Forward {

    private final EpsilonMahjongStateReadoutGroupExecution owner;
    private final int slot;
    private final NDManager workingManager;
    private FusionOutputLease lease;
    private NDArray activePolicyState;
    private NDArray activeValueState;
    private boolean evaluated;
    private boolean sealed;
    private boolean poisoned;
    private boolean finished;

    private Forward(
        EpsilonMahjongStateReadoutGroupExecution owner, int slot, NDManager workingManager) {
      this.owner = owner;
      this.slot = slot;
      this.workingManager = workingManager;
    }

    public Readouts read(EpsilonMahjongStateEncoder.EncodedMemory memory) {
      if (evaluated || sealed || poisoned) {
        throw new IllegalStateException("state readout forward can evaluate only once");
      }
      NDArray entityMemory = memory.entityEmbeddings();
      NDArray validMask = memory.entityMask();
      long rowCount = entityMemory.getShape().get(0);
      requireInputDataType(owner.memoryInput, entityMemory);
      requireInputDataType(owner.maskInput, validMask);
      if (entityMemory.getManager() != workingManager) {
        entityMemory.attach(workingManager);
      }
      if (validMask.getManager() != workingManager) {
        validMask.attach(workingManager);
      }
      boolean submitAttempted = false;
      try (FusionInvocation invocation = owner.session.acquire()) {
        invocation.setInput(owner.memoryInput, entityMemory);
        invocation.setInput(owner.maskInput, validMask);
        invocation.setDimension(owner.batch, rowCount);
        submitAttempted = true;
        lease = invocation.submit();
        activePolicyState = lease.get(owner.policyOutput).get(NDIndex.sliceAxis(0, 0, rowCount));
        activePolicyState.attach(workingManager);
        if (owner.valueOutput != null) {
          activeValueState = lease.get(owner.valueOutput).get(NDIndex.sliceAxis(0, 0, rowCount));
          activeValueState.attach(workingManager);
        }
        evaluated = true;
        return new Readouts(activePolicyState, activeValueState);
      } catch (RuntimeException | Error evaluationFailure) {
        if (submitAttempted && lease == null) {
          poisonUnknownSubmission(evaluationFailure);
        } else {
          poison(evaluationFailure);
        }
        throw evaluationFailure;
      }
    }

    public AutoCloseable seal() {
      if (!evaluated || sealed || poisoned) {
        throw new IllegalStateException("state readout forward cannot be sealed");
      }
      sealed = true;
      return this::releaseAfterCompletion;
    }

    public void poison(Throwable cause) {
      if (poisoned) {
        return;
      }
      poisoned = true;
      if (owner.failure == null) {
        owner.failure = cause;
      }
      boolean submittedWork = lease != null || owner.unknownSubmission;
      owner.incomplete |= submittedWork;
      if (!submittedWork) {
        try {
          releaseAfterCompletion();
        } catch (Throwable releaseFailure) {
          cause.addSuppressed(releaseFailure);
        }
      }
    }

    /** 完了トークンを返さず失敗した投入を、ストリーム完了確認まで再利用禁止にする。 */
    public void poisonUnknownSubmission(Throwable cause) {
      owner.unknownSubmission = true;
      poison(cause);
    }

    private void releaseAfterCompletion() {
      if (finished) {
        return;
      }
      Throwable releaseFailure = null;
      if (activeValueState != null) {
        try {
          activeValueState.close();
          activeValueState = null;
        } catch (Throwable closeFailure) {
          releaseFailure = closeFailure;
        }
      }
      if (activePolicyState != null) {
        try {
          activePolicyState.close();
          activePolicyState = null;
        } catch (Throwable closeFailure) {
          releaseFailure = addFailure(releaseFailure, closeFailure);
        }
      }
      if (activeValueState == null && activePolicyState == null && lease != null) {
        releaseFailure = closeResource(releaseFailure, lease);
        if (releaseFailure == null) {
          lease = null;
        }
      }
      if (releaseFailure == null && activeValueState == null && activePolicyState == null) {
        finished = true;
        owner.finish(slot, this);
      } else if (releaseFailure != null) {
        poisoned = true;
        owner.incomplete = true;
        if (owner.failure == null) {
          owner.failure = releaseFailure;
        }
      }
      rethrow(releaseFailure);
    }

    public boolean hasActiveOutputViews() {
      return activePolicyState != null || activeValueState != null;
    }

    private static void requireInputDataType(FusionRecipe.Input input, NDArray value) {
      DataType expected = input.getSpec().getDataType();
      DataType actual = value.getDataType();
      if (actual != expected) {
        throw new IllegalArgumentException(
            "state readout input "
                + input.getName()
                + " dtype differs from its recipe: "
                + actual
                + "/"
                + expected
                + ", shape="
                + value.getShape());
      }
    }
  }

  private static Throwable closeResource(Throwable failure, AutoCloseable resource) {
    if (resource == null) {
      return failure;
    }
    try {
      resource.close();
    } catch (Throwable closeFailure) {
      return addFailure(failure, closeFailure);
    }
    return failure;
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
