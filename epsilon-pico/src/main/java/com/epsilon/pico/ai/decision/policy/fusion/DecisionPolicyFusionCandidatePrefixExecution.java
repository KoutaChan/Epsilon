package com.epsilon.pico.ai.decision.policy.fusion;

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
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.pico.ai.decision.input.DecisionInputSchema;
import com.epsilon.pico.ai.decision.policy.DecisionPolicyCandidatePrefixExecution;
import com.epsilon.pico.ai.decision.policy.EpsilonFeatureFusion;
import com.epsilon.runtime.InferenceProfile;
import java.util.ArrayList;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 候補アフィン変換の先頭構成要素をdjl-rocmの型付き OUTPUT_PACKで直接連結する。
 *
 * <p>四つの候補別構成要素は{@code [rows * actions, width]} ビューとして一つの再利用するバッファへ書き、同時に射影
 * 重みのデータ型へ変換する。後段は学習ブロックと同じ一回のrocBLAS GEMMを使うため、AFFINE_SUMのように 構成要素ごとのGEMMへ分割しない。候補全体の{@code
 * cat}とautocast 複製だけを一つのHIP 起動へ置き換える。
 *
 * <p>一位置の最大永続出力は {@code maxBatch * MAX_LEGAL_ACTIONS * prefixWidth * dtypeBytes}
 * である。実測バッチの行動幅ではなくスキーマ上限を確保するため、設定やVRAM見積りも40候補を用いる。
 */
public final class DecisionPolicyFusionCandidatePrefixExecution
    extends DecisionPolicyCandidatePrefixExecution {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(DecisionPolicyFusionCandidatePrefixExecution.class);

  private final NDManager resourceManager;
  private final FusionCompiler compiler;
  private final Linear stateProjection;
  private final NDArray prefixWeight;
  private final NDArray prefixBias;
  private final int[] inputWidths;
  private final int hiddenSize;
  private final int maximumItems;
  private final int executionSlots;
  private final DataType outputDataType;
  private final FusionForward[] activeForwards;
  private final ArrayList<FusionOutputLease> poisonedLeases = new ArrayList<>();
  private Program program;
  private Throwable failure;
  private boolean unknownSubmission;
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
    this.executionSlots = executionSlots;
    hiddenSize = candidateFusion.outputSize();
    int prefixComponents = candidateFusion.fusedPrefixComponents();
    if (prefixComponents <= 0 || candidateFusion.componentCount() != prefixComponents + 1) {
      throw new IllegalArgumentException(
          "candidate prefix packing requires one broadcast component after the prefix");
    }
    stateProjection = candidateFusion.projection(prefixComponents);
    inputWidths = new int[prefixComponents];
    for (int component = 0; component < prefixComponents; component++) {
      inputWidths[component] = candidateFusion.inputSize(component);
    }
    maximumItems = Math.multiplyExact(maximumBatch, DecisionInputSchema.MAX_LEGAL_ACTIONS);
    activeForwards = new FusionForward[executionSlots];
    resourceManager = manager.newSubManager();

    try {
      compiler = manager.getEngine().newFusionCompiler(manager.getDevice());
      NDList weights = new NDList(prefixComponents);
      for (int component = 0; component < prefixComponents; component++) {
        weights.add(
            parameterStore.getValue(
                weight(candidateFusion.projection(component)), manager.getDevice(), false));
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
      long bytesPerSlot =
          Math.multiplyExact(
              Math.multiplyExact((long) maximumItems, prefixWeight.getShape().get(1)),
              outputDataType.getNumOfBytes());
      LOGGER.info(
          "Decision candidate prefix pack: maximumItems={}, maximumActionCapacity={}, "
              + "inputWidths={}, outputWidth={}, outputDtype={}, slots={}, "
              + "retainedOutputBytesPerSlot={}, maximumRetainedOutputBytes={}",
          maximumItems,
          DecisionInputSchema.MAX_LEGAL_ACTIONS,
          Arrays.toString(inputWidths),
          prefixWeight.getShape().get(1),
          outputDataType,
          executionSlots,
          bytesPerSlot,
          Math.multiplyExact(bytesPerSlot, executionSlots));
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
    int slot = freeSlot();
    FusionForward forward = new FusionForward(this, slot, workingManager);
    activeForwards[slot] = forward;
    return forward;
  }

  @Override
  public boolean hasIncompleteWork() {
    return incomplete || unknownSubmission || !poisonedLeases.isEmpty();
  }

  @Override
  public Throwable failure() {
    return failure;
  }

  @Override
  public void releaseFailedForwardAfterCompletion() {
    if (!hasIncompleteWork()) {
      return;
    }
    Throwable releaseFailure = null;
    for (int index = poisonedLeases.size() - 1; index >= 0; index--) {
      try {
        poisonedLeases.get(index).close();
        poisonedLeases.remove(index);
      } catch (Throwable closeFailure) {
        releaseFailure = addFailure(releaseFailure, closeFailure);
      }
    }
    unknownSubmission = false;
    for (FusionForward forward : activeForwards) {
      if (forward != null) {
        try {
          forward.close();
        } catch (Throwable closeFailure) {
          releaseFailure = addFailure(releaseFailure, closeFailure);
        }
      }
    }
    incomplete = hasActiveForward() || unknownSubmission || !poisonedLeases.isEmpty();
    rethrow(releaseFailure);
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    if (hasIncompleteWork()) {
      throw new IllegalStateException(
          "cannot close candidate prefix pack with incomplete device work", failure);
    }
    if (hasActiveForward()) {
      throw new IllegalStateException("cannot close candidate prefix pack with an active forward");
    }
    Throwable closeFailure = null;
    if (program != null) {
      try {
        program.close();
        program = null;
      } catch (Throwable failure) {
        closeFailure = failure;
      }
    }
    if (program == null) {
      closeFailure = closeResource(closeFailure, resourceManager);
    }
    if (closeFailure == null) {
      closed = true;
    }
    rethrow(closeFailure);
  }

  private NDArray submit(
      FusionForward owner,
      ParameterStore parameterStore,
      PairList<String, Object> runtimeParameters,
      NDArray[] components) {
    long rows = components[0].getShape().get(0);
    long actionCapacity = components[0].getShape().get(1);
    long itemCount = Math.multiplyExact(rows, actionCapacity);
    NDArray[] packedInputs = new NDArray[inputWidths.length];
    for (int component = 0; component < packedInputs.length; component++) {
      packedInputs[component] = components[component].reshape(itemCount, inputWidths[component]);
    }
    Program current = program(packedInputs);
    FusionOutputLease lease = null;
    NDArray packed = null;
    boolean submitAttempted = false;
    try (FusionInvocation invocation = current.acquire()) {
      try (var ignored = InferenceProfile.section("policy.prefix_pack")) {
        for (int input = 0; input < packedInputs.length; input++) {
          invocation.setInput(current.input(input), packedInputs[input]);
        }
        invocation.setDimension(current.items(), itemCount);
        submitAttempted = true;
        lease = invocation.submit();
        packed = lease.get(current.output()).get(NDIndex.sliceAxis(0, 0, itemCount));
        packed.attach(owner.workingManager());
        owner.retain(components);
        owner.retain(packedInputs);
        owner.setLease(lease);
      }

      NDArray projected;
      try (var ignored = InferenceProfile.section("policy.prefix_linear")) {
        projected =
            Linear.linear(packed, prefixWeight, prefixBias)
                .singletonOrThrow()
                .reshape(rows, actionCapacity, hiddenSize);
      }
      NDArray state =
          stateProjection
              .forward(
                  parameterStore,
                  new NDList(components[components.length - 1]),
                  false,
                  runtimeParameters)
              .singletonOrThrow();
      return addStateProjectionAndActivate(projected, state);
    } catch (RuntimeException | Error submitFailure) {
      if (lease != null && owner.lease() == null) {
        try {
          lease.synchronize();
          lease.close();
        } catch (RuntimeException | Error completionFailure) {
          submitFailure.addSuppressed(completionFailure);
          poisonedLeases.add(lease);
          incomplete = true;
        }
      } else if (submitAttempted && lease == null) {
        unknownSubmission = true;
        incomplete = true;
      }
      closeResource(submitFailure, packed);
      poison(submitFailure, submitAttempted);
      throw submitFailure;
    }
  }

  private Program program(NDArray[] inputs) {
    DataType[] inputTypes = new DataType[inputs.length];
    for (int input = 0; input < inputs.length; input++) {
      inputTypes[input] = inputs[input].getDataType();
      requireFloatingType(inputTypes[input]);
    }
    if (program == null) {
      program =
          new Program(
              compiler,
              resourceManager,
              inputWidths,
              inputTypes,
              outputDataType,
              maximumItems,
              executionSlots);
    } else if (!program.matches(inputTypes)) {
      throw new IllegalStateException(
          "candidate prefix input dtypes changed after binding: expected="
              + program.inputTypes()
              + ", actual="
              + Arrays.toString(inputTypes));
    }
    return program;
  }

  /**
   * 所有済み候補射影へ状態射影をブロードキャスト加算し、同じバッファへSiLUを適用する。
   *
   * <p>候補 LayerNormより前の数値境界だけを置換する。合法手マスクはLayerNorm後に適用する必要があるため、ここでは移動しない。
   *
   * @param projected 呼び出し側が所有する {@code [batch, actions, hidden]} 射影
   * @param state {@code [batch, 1, hidden]} 状態射影
   * @return 更新済みの {@code projected}
   */
  public static NDArray addStateProjectionAndActivate(NDArray projected, NDArray state) {
    return NDArrays.addBroadcastResidualToOwnedAndSilu(projected, state);
  }

  private int freeSlot() {
    for (int slot = 0; slot < activeForwards.length; slot++) {
      if (activeForwards[slot] == null) {
        return slot;
      }
    }
    throw new IllegalStateException("no candidate prefix pack slot is available");
  }

  private boolean hasActiveForward() {
    for (FusionForward forward : activeForwards) {
      if (forward != null) {
        return true;
      }
    }
    return false;
  }

  private void finish(int slot, FusionForward forward) {
    if (activeForwards[slot] != forward) {
      throw new IllegalStateException("candidate prefix pack slot ownership was lost");
    }
    activeForwards[slot] = null;
  }

  private void poison(Throwable cause, boolean submittedWork) {
    if (failure == null) {
      failure = cause;
    }
    incomplete |= submittedWork;
  }

  private static Parameter weight(Linear projection) {
    return projection.getDirectParameters().get("weight");
  }

  private static void requireFloatingType(DataType dataType) {
    if (dataType != DataType.FLOAT16
        && dataType != DataType.BFLOAT16
        && dataType != DataType.FLOAT32) {
      throw new IllegalArgumentException(
          "candidate prefix inputs require a floating-point dtype: " + dataType);
    }
  }

  private static final class Program implements AutoCloseable {

    private final DataType[] inputTypes;
    private final FusionRecipe.Dimension items;
    private final FusionRecipe.Input[] inputs;
    private final FusionRecipe.Output output;
    private FusionPlan plan;
    private FusionExecutable executable;
    private FusionSession session;

    private Program(
        FusionCompiler compiler,
        NDManager resourceManager,
        int[] inputWidths,
        DataType[] inputTypes,
        DataType outputDataType,
        int maximumItems,
        int executionSlots) {
      this.inputTypes = inputTypes.clone();
      FusionRecipe.Builder builder = FusionRecipe.builder("epsilon-policy-candidate-prefix-pack");
      items = builder.addDimension("items", maximumItems);
      inputs = new FusionRecipe.Input[inputWidths.length];
      FusionRecipe.OutputPackBuilder pack = builder.outputPack("features");
      for (int input = 0; input < inputs.length; input++) {
        inputs[input] =
            builder.addInput(
                "component" + input,
                FusionRecipe.TensorSpec.of(inputTypes[input], items, inputWidths[input]));
        pack.addSource(inputs[input]);
      }
      FusionRecipe.OutputPack packed = pack.optOutputDataType(outputDataType).build();
      output = builder.addOutput("features", packed);
      FusionRecipe recipe = builder.build();
      FusionPlan createdPlan = null;
      FusionExecutable createdExecutable = null;
      FusionSession createdSession = null;
      try {
        createdPlan = compiler.prepare(recipe, FusionCompileConfig.defaults());
        FusionCompilationReport report = createdPlan.getCompilationReport();
        if (!report.isNativeOnly()) {
          throw new IllegalStateException("candidate prefix pack plan is not native-only");
        }
        createdExecutable = createdPlan.bind(FusionConstantBindings.builder(recipe).build());
        createdSession =
            createdExecutable.newSession(
                resourceManager,
                FusionSessionConfig.builder().optOutputSlotCount(executionSlots).build());
        LOGGER.info(
            "Decision candidate prefix pack prepared: inputDtypes={}, backend={}, commands={}, "
                + "outputSlots={}, executableStorageBytes={}, retainedSessionStorageBytes={}, "
                + "requiredExecutionLaneStorageBytes={}",
            Arrays.toString(inputTypes),
            report.getBackend(),
            report.getCommandCount(),
            executionSlots,
            report.getExecutableStorageBytes(),
            report.getRetainedSessionStorageBytes(executionSlots),
            report.getRequiredExecutionLaneStorageBytes());
      } catch (RuntimeException | Error constructionFailure) {
        closeResource(constructionFailure, createdSession);
        closeResource(constructionFailure, createdExecutable);
        closeResource(constructionFailure, createdPlan);
        throw constructionFailure;
      }
      plan = createdPlan;
      executable = createdExecutable;
      session = createdSession;
    }

    private boolean matches(DataType[] actual) {
      return Arrays.equals(inputTypes, actual);
    }

    private String inputTypes() {
      return Arrays.toString(inputTypes);
    }

    private FusionInvocation acquire() {
      return session.acquire();
    }

    private FusionRecipe.Dimension items() {
      return items;
    }

    private FusionRecipe.Input input(int index) {
      return inputs[index];
    }

    private FusionRecipe.Output output() {
      return output;
    }

    @Override
    public void close() {
      Throwable closeFailure = null;
      closeFailure = closeResource(closeFailure, session);
      if (closeFailure == null) {
        session = null;
        closeFailure = closeResource(null, executable);
      }
      if (closeFailure == null) {
        executable = null;
        closeFailure = closeResource(null, plan);
      }
      if (closeFailure == null) {
        plan = null;
      }
      rethrow(closeFailure);
    }
  }

  private static final class FusionForward extends Forward {

    private final DecisionPolicyFusionCandidatePrefixExecution owner;
    private final int slot;
    private final NDArray[] retained = new NDArray[10];
    private NDManager workingManager;
    private FusionOutputLease lease;
    private int retainedCount;
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
      NDArray result = owner.submit(this, parameterStore, runtimeParameters, components);
      executed = true;
      return result;
    }

    private void retain(NDArray[] arrays) {
      for (NDArray array : arrays) {
        if (array.getManager() != workingManager) {
          array.attach(workingManager);
        }
        retained[retainedCount++] = array;
      }
    }

    private NDManager workingManager() {
      return workingManager;
    }

    private FusionOutputLease lease() {
      return lease;
    }

    private void setLease(FusionOutputLease lease) {
      this.lease = lease;
    }

    @Override
    public AutoCloseable seal() {
      if (finished || sealed || !executed || lease == null) {
        throw new IllegalStateException("candidate prefix pack has no completed submission");
      }
      sealed = true;
      return this;
    }

    @Override
    public void poison(Throwable failure) {
      if (finished) {
        return;
      }
      owner.poison(failure, lease != null || owner.hasIncompleteWork());
      if (!owner.hasIncompleteWork()) {
        close();
      }
    }

    @Override
    public void close() {
      if (finished) {
        return;
      }
      if (lease != null) {
        lease.close();
      }
      Arrays.fill(retained, null);
      retainedCount = 0;
      lease = null;
      workingManager = null;
      finished = true;
      owner.finish(slot, this);
    }
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
