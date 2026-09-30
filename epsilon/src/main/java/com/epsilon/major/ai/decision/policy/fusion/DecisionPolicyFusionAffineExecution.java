package com.epsilon.major.ai.decision.policy.fusion;

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
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.nn.transformer.IdEmbedding;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.major.ai.decision.input.DecisionInputSchema;
import com.epsilon.major.ai.decision.policy.DecisionAlternative;
import com.epsilon.major.ai.decision.policy.DecisionPolicyAffineExecution;
import com.epsilon.major.ai.decision.policy.EpsilonFeatureFusion;
import java.util.ArrayList;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** djl-rocmのAFFINE_SUMを使い、方策順伝播の3個のアフィン変換和をパイプラインの実行枠間で再利用する。 */
public final class DecisionPolicyFusionAffineExecution extends DecisionPolicyAffineExecution {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(DecisionPolicyFusionAffineExecution.class);
  private final NDManager constantsManager;
  private final FusionCompiler compiler;
  private final ParameterStore parameterStore;
  private final EpsilonFeatureFusion candidateFusion;
  private final EpsilonFeatureFusion transitionFusion;
  private final Linear alternativeHidden;
  private final DataType dataType;
  private final int hiddenSize;
  private final int maxBatch;
  private final int maxTransitionExtent;
  private final int executionSlots;
  private final NDArray alternativeSelectors;
  private final NDArray[] alternativeWeights;
  private final FusionForward[] activeForwards;
  private final AffineProgram[] candidatePrograms;
  private final ArrayList<AffineProgram> retiredPrograms;
  private AffineProgram transitionProgram;
  private AffineProgram alternativeProgram;
  private Throwable failure;
  private long candidateCalls;
  private long transitionCalls;
  private long alternativeCalls;
  private long submitNanos;
  private long activeLeadingElements;
  private long cacheHits;
  private long cacheMisses;
  private long growths;
  private boolean incomplete;
  private boolean closed;
  private boolean summaryLogged;
  private boolean constantsReleased;
  private boolean resourcesReleased;

  public DecisionPolicyFusionAffineExecution(
      NDManager manager,
      ParameterStore parameterStore,
      EpsilonFeatureFusion candidateFusion,
      EpsilonFeatureFusion transitionFusion,
      Linear alternativeHidden,
      IdEmbedding alternativeOffset,
      int hiddenSize,
      DataType expectedDataType,
      int maxBatch,
      int multiTransitionMaxBatch,
      int executionSlots) {
    if (executionSlots <= 0) {
      throw new IllegalArgumentException("executionSlots must be positive");
    }
    this.parameterStore = parameterStore;
    this.candidateFusion = candidateFusion;
    this.transitionFusion = transitionFusion;
    this.alternativeHidden = alternativeHidden;
    this.hiddenSize = hiddenSize;
    this.maxBatch = maxBatch;
    this.executionSlots = executionSlots;
    int maximumActionCapacity =
        DecisionInputSchema.LEGAL_ACTION_BUCKETS[
            DecisionInputSchema.LEGAL_ACTION_BUCKETS.length - 1];
    int maximumTransitionCapacity =
        DecisionInputSchema.ACTION_TRANSITION_BUCKETS[
            DecisionInputSchema.ACTION_TRANSITION_BUCKETS.length - 1];
    maxTransitionExtent =
        Math.max(
            Math.multiplyExact(maxBatch, maximumActionCapacity),
            Math.multiplyExact(
                multiTransitionMaxBatch,
                Math.multiplyExact(maximumActionCapacity, maximumTransitionCapacity)));
    constantsManager = manager.newSubManager();
    candidatePrograms = new AffineProgram[maximumActionCapacity + 1];
    activeForwards = new FusionForward[executionSlots];
    retiredPrograms = new ArrayList<>();
    try {
      compiler = manager.getEngine().newFusionCompiler(manager.getDevice());
      dataType = weight(candidateFusion.projection(0)).getDataType();
      requireFloatingType(dataType);
      if (dataType != expectedDataType) {
        throw new IllegalArgumentException(
            "AFFINE_SUM parameter dtype differs from compute precision: "
                + dataType
                + "/"
                + expectedDataType);
      }
      validateParameterTypes();
      int alternatives = DecisionAlternative.NETWORK_SIZE;
      float[] selectors = new float[alternatives * alternatives];
      for (int index = 0; index < alternatives; index++) {
        selectors[index * alternatives + index] = 1.0f;
      }
      alternativeSelectors =
          constantsManager
              .create(selectors)
              .reshape(1, alternatives, alternatives)
              .toType(dataType, false);
      alternativeWeights = splitAlternativeWeights(alternativeOffset);
    } catch (RuntimeException | Error constructionFailure) {
      closeResource(constructionFailure, constantsManager);
      throw constructionFailure;
    }
  }

  @Override
  public Forward beginForward(NDManager workingManager) {
    requireUsable();
    int slot = freeForwardSlot();
    FusionForward forward = new FusionForward(this, slot, workingManager);
    activeForwards[slot] = forward;
    return forward;
  }

  @Override
  public boolean hasIncompleteWork() {
    return incomplete || programHasIncompleteWork();
  }

  @Override
  public Throwable failure() {
    return failure;
  }

  @Override
  public void releaseFailedForwardAfterCompletion() {
    if (!incomplete && !programHasIncompleteWork()) {
      return;
    }
    Throwable releaseFailure = null;
    releaseFailure = releasePoisonedLease(releaseFailure, alternativeProgram);
    for (AffineProgram program : candidatePrograms) {
      releaseFailure = releasePoisonedLease(releaseFailure, program);
    }
    releaseFailure = releasePoisonedLease(releaseFailure, transitionProgram);
    for (AffineProgram program : retiredPrograms) {
      releaseFailure = releasePoisonedLease(releaseFailure, program);
    }
    for (FusionForward forward : activeForwards) {
      if (forward != null) {
        releaseFailure = closeResource(releaseFailure, forward);
      }
    }
    incomplete = hasActiveForward() || programHasIncompleteWork();
    rethrow(releaseFailure);
  }

  @Override
  public void close() {
    if (resourcesReleased) {
      return;
    }
    closed = true;
    if (hasIncompleteWork()) {
      throw new IllegalStateException(
          "cannot release poisoned affine resources without final output completion", failure);
    }
    if (hasActiveForward()) {
      throw new IllegalStateException("cannot close affine execution with an active forward");
    }
    if (!summaryLogged) {
      LOGGER.info(
          "Decision affine fusion summary: candidateCalls={}, transitionCalls={}, "
              + "alternativeCalls={}, submitMs={}, activeLeadingElements={}, cacheHits={}, "
              + "cacheMisses={}, growths={}, programs={}, currentPlanExecutableStorageBytes={}, "
              + "currentPlanRetainedSessionStorageBytes={}, "
              + "currentPlanIsolatedLaneStorageUpperBoundBytes={}, "
              + "currentPlanLogicalWorkspaceBytes={}",
          candidateCalls,
          transitionCalls,
          alternativeCalls,
          submitNanos / 1_000_000.0,
          activeLeadingElements,
          cacheHits,
          cacheMisses,
          growths,
          programCount(),
          currentPlanExecutableStorageBytes(),
          currentPlanRetainedSessionStorageBytes(),
          currentPlanIsolatedLaneStorageUpperBoundBytes(),
          currentPlanLogicalWorkspaceBytes());
      summaryLogged = true;
    }
    Throwable closeFailure = null;
    if (alternativeProgram != null) {
      try {
        alternativeProgram.close();
        alternativeProgram = null;
      } catch (Throwable failure) {
        closeFailure = addFailure(closeFailure, failure);
      }
    }
    for (int actionCapacity = candidatePrograms.length - 1; actionCapacity >= 0; actionCapacity--) {
      AffineProgram program = candidatePrograms[actionCapacity];
      if (program != null) {
        try {
          program.close();
          candidatePrograms[actionCapacity] = null;
        } catch (Throwable failure) {
          closeFailure = addFailure(closeFailure, failure);
        }
      }
    }
    if (transitionProgram != null) {
      try {
        transitionProgram.close();
        transitionProgram = null;
      } catch (Throwable failure) {
        closeFailure = addFailure(closeFailure, failure);
      }
    }
    for (int index = retiredPrograms.size() - 1; index >= 0; index--) {
      try {
        retiredPrograms.get(index).close();
        retiredPrograms.remove(index);
      } catch (Throwable failure) {
        closeFailure = addFailure(closeFailure, failure);
      }
    }
    if (programCount() == 0 && !constantsReleased) {
      try {
        constantsManager.close();
        constantsReleased = true;
      } catch (Throwable failure) {
        closeFailure = addFailure(closeFailure, failure);
      }
    }
    resourcesReleased = constantsReleased;
    rethrow(closeFailure);
  }

  private NDArray submitCandidate(
      FusionForward owner,
      NDArray actionFeatures,
      NDArray primaryTiles,
      NDArray transitions,
      NDArray playerContexts,
      NDArray stateContext) {
    int rows = Math.toIntExact(actionFeatures.getShape().get(0));
    int actionCapacity = Math.toIntExact(actionFeatures.getShape().get(1));
    if (actionCapacity <= 0 || actionCapacity >= candidatePrograms.length) {
      throw new IllegalArgumentException(
          "unsupported candidate action capacity: " + actionCapacity);
    }
    owner.requireUnused(Site.CANDIDATE);
    try {
      long startedNanos = System.nanoTime();
      NDArray[] inputs = {actionFeatures, primaryTiles, transitions, playerContexts, stateContext};
      AffineProgram program = candidateProgram(actionCapacity, rows, inputs);
      owner.retainInputs(inputs);
      NDArray result =
          owner.add(Site.CANDIDATE, program.submit(owner.workingManager(), rows, inputs));
      recordSubmit(Site.CANDIDATE, rows, startedNanos);
      return result;
    } catch (RuntimeException | Error submitFailure) {
      poison(submitFailure, owner.outputCount() > 0 || programHasIncompleteWork());
      throw submitFailure;
    }
  }

  private NDArray submitTransition(
      FusionForward owner,
      NDArray features,
      NDArray tileContexts,
      NDArray discardTiles,
      NDArray discardSafety,
      NDArray stateContexts) {
    int presentCount = Math.toIntExact(features.getShape().get(0));
    owner.requireUnused(Site.TRANSITION);
    try {
      long startedNanos = System.nanoTime();
      NDArray[] inputs = {features, tileContexts, discardTiles, discardSafety, stateContexts};
      AffineProgram program = transitionProgram(presentCount, inputs);
      owner.retainInputs(inputs);
      NDArray result =
          owner.add(Site.TRANSITION, program.submit(owner.workingManager(), presentCount, inputs));
      recordSubmit(Site.TRANSITION, presentCount, startedNanos);
      return result;
    } catch (RuntimeException | Error submitFailure) {
      poison(submitFailure, owner.outputCount() > 0 || programHasIncompleteWork());
      throw submitFailure;
    }
  }

  private NDArray submitAlternative(
      FusionForward owner, NDArray stateContext, NDArray candidateContexts) {
    int rows = Math.toIntExact(candidateContexts.getShape().get(0));
    owner.requireUnused(Site.ALTERNATIVE);
    try {
      long startedNanos = System.nanoTime();
      NDArray[] inputs = {stateContext, candidateContexts};
      AffineProgram program = alternativeProgram(rows, inputs);
      owner.retainInputs(inputs);
      NDArray result =
          owner.add(Site.ALTERNATIVE, program.submit(owner.workingManager(), rows, inputs));
      recordSubmit(Site.ALTERNATIVE, rows, startedNanos);
      return result;
    } catch (RuntimeException | Error submitFailure) {
      poison(submitFailure, owner.outputCount() > 0 || programHasIncompleteWork());
      throw submitFailure;
    }
  }

  private AffineProgram candidateProgram(int actionCapacity, int rows, NDArray[] inputs) {
    AffineProgram current = candidatePrograms[actionCapacity];
    if (current != null && current.capacity() >= rows) {
      cacheHits++;
      return current;
    }
    recordCacheMiss(current);
    int capacity = grownCapacity(rows, maxBatch);
    retireForGrowth(current);
    candidatePrograms[actionCapacity] = null;
    AffineProgram created =
        createCandidateProgram(actionCapacity, capacity, inputDataTypes(inputs));
    candidatePrograms[actionCapacity] = created;
    return created;
  }

  private AffineProgram transitionProgram(int presentCount, NDArray[] inputs) {
    if (transitionProgram != null && transitionProgram.capacity() >= presentCount) {
      cacheHits++;
      return transitionProgram;
    }
    recordCacheMiss(transitionProgram);
    int capacity = grownCapacity(presentCount, maxTransitionExtent);
    retireForGrowth(transitionProgram);
    transitionProgram = null;
    transitionProgram = createTransitionProgram(capacity, inputDataTypes(inputs));
    return transitionProgram;
  }

  private AffineProgram alternativeProgram(int rows, NDArray[] inputs) {
    if (alternativeProgram != null && alternativeProgram.capacity() >= rows) {
      cacheHits++;
      return alternativeProgram;
    }
    recordCacheMiss(alternativeProgram);
    int capacity = grownCapacity(rows, maxBatch);
    retireForGrowth(alternativeProgram);
    alternativeProgram = null;
    alternativeProgram = createAlternativeProgram(capacity, inputDataTypes(inputs));
    return alternativeProgram;
  }

  private AffineProgram createCandidateProgram(
      int actionCapacity, int capacity, DataType[] inputDataTypes) {
    FusionRecipe.Builder builder =
        FusionRecipe.builder(
            "epsilon-decision-candidate-affine-a" + actionCapacity + "-b" + capacity);
    FusionRecipe.Dimension rows = builder.addDimension("rows", capacity);
    FusionRecipe.Input[] inputs =
        new FusionRecipe.Input[] {
          builder.addInput(
              "actionFeatures",
              FusionRecipe.TensorSpec.of(inputDataTypes[0], rows, actionCapacity, hiddenSize)),
          builder.addInput(
              "primaryTiles",
              FusionRecipe.TensorSpec.of(inputDataTypes[1], rows, actionCapacity, hiddenSize)),
          builder.addInput(
              "transitions",
              FusionRecipe.TensorSpec.of(inputDataTypes[2], rows, actionCapacity, hiddenSize)),
          builder.addInput(
              "playerContexts",
              FusionRecipe.TensorSpec.of(
                  inputDataTypes[3], rows, actionCapacity, candidateFusion.inputSize(3))),
          builder.addInput(
              "stateContext", FusionRecipe.TensorSpec.of(inputDataTypes[4], rows, 1, hiddenSize))
        };
    FusionConstantBindings.Builder bindings;
    FusionRecipe.AffineSumBuilder affine = builder.affineSum("candidateHidden", hiddenSize);
    FusionRecipe.Constant[] weights =
        addProjectionWeights(builder, affine, inputs, candidateFusion);
    FusionRecipe.Constant bias = addBias(builder, candidateFusion.outputSize());
    FusionRecipe.AffineSum value =
        affine.optBias(bias).optActivation(FusionRecipe.Activation.SILU).build();
    FusionRecipe.Output output = builder.addOutput("candidateHidden", value);
    FusionRecipe recipe = builder.build();
    bindings = bindProjectionConstants(recipe, weights, bias, candidateFusion);
    return prepare(Site.CANDIDATE, capacity, rows, inputs, output, recipe, bindings);
  }

  private AffineProgram createTransitionProgram(int capacity, DataType[] inputDataTypes) {
    FusionRecipe.Builder builder =
        FusionRecipe.builder("epsilon-decision-transition-affine-p" + capacity);
    FusionRecipe.Dimension present = builder.addDimension("present", capacity);
    FusionRecipe.Input[] inputs = new FusionRecipe.Input[transitionFusion.componentCount()];
    for (int component = 0; component < inputs.length; component++) {
      inputs[component] =
          builder.addInput(
              "component" + component,
              FusionRecipe.TensorSpec.of(
                  inputDataTypes[component], present, 1, 1, transitionFusion.inputSize(component)));
    }
    FusionRecipe.AffineSumBuilder affine = builder.affineSum("transitionHidden", hiddenSize);
    FusionRecipe.Constant[] weights =
        addProjectionWeights(builder, affine, inputs, transitionFusion);
    FusionRecipe.Constant bias = addBias(builder, transitionFusion.outputSize());
    FusionRecipe.AffineSum value =
        affine.optBias(bias).optActivation(FusionRecipe.Activation.SILU).build();
    FusionRecipe.Output output = builder.addOutput("transitionHidden", value);
    FusionRecipe recipe = builder.build();
    FusionConstantBindings.Builder bindings =
        bindProjectionConstants(recipe, weights, bias, transitionFusion);
    return prepare(Site.TRANSITION, capacity, present, inputs, output, recipe, bindings);
  }

  private AffineProgram createAlternativeProgram(int capacity, DataType[] inputDataTypes) {
    int alternatives = DecisionAlternative.NETWORK_SIZE;
    FusionRecipe.Builder builder =
        FusionRecipe.builder("epsilon-decision-alternative-affine-b" + capacity);
    FusionRecipe.Dimension rows = builder.addDimension("rows", capacity);
    FusionRecipe.Input state =
        builder.addInput(
            "stateContext", FusionRecipe.TensorSpec.of(inputDataTypes[0], rows, 1, hiddenSize));
    FusionRecipe.Input candidates =
        builder.addInput(
            "candidateContexts",
            FusionRecipe.TensorSpec.of(inputDataTypes[1], rows, alternatives, hiddenSize));
    FusionRecipe.Constant selectors =
        builder.addConstant(
            "alternativeSelectors",
            FusionRecipe.TensorSpec.fixed(dataType, 1, alternatives, alternatives));
    FusionRecipe.Value[] sources = {state, candidates, selectors};
    FusionRecipe.AffineSumBuilder affine = builder.affineSum("alternativeHidden", hiddenSize);
    FusionRecipe.Constant[] weights = new FusionRecipe.Constant[sources.length];
    for (int component = 0; component < sources.length; component++) {
      weights[component] =
          builder.addConstant(
              "weight" + component,
              FusionRecipe.TensorSpec.fixed(
                  dataType, hiddenSize, component == 2 ? alternatives : hiddenSize));
      affine.addTerm(sources[component], weights[component]);
    }
    Parameter biasParameter = alternativeHidden.getDirectParameters().get("bias");
    FusionRecipe.Constant bias =
        builder.addConstant("bias", FusionRecipe.TensorSpec.fixed(dataType, hiddenSize));
    FusionRecipe.AffineSum value =
        affine.optBias(bias).optActivation(FusionRecipe.Activation.SILU).build();
    FusionRecipe.Output output = builder.addOutput("alternativeHidden", value);
    FusionRecipe recipe = builder.build();
    FusionConstantBindings.Builder bindings = FusionConstantBindings.builder(recipe);
    bindings.bind(selectors, alternativeSelectors);
    for (int component = 0; component < weights.length; component++) {
      bindings.bind(weights[component], alternativeWeights[component]);
    }
    bindings.bind(
        bias, parameterStore.getValue(biasParameter, constantsManager.getDevice(), false));
    return prepare(
        Site.ALTERNATIVE,
        capacity,
        rows,
        new FusionRecipe.Input[] {state, candidates},
        output,
        recipe,
        bindings);
  }

  private FusionRecipe.Constant[] addProjectionWeights(
      FusionRecipe.Builder builder,
      FusionRecipe.AffineSumBuilder affine,
      FusionRecipe.Input[] inputs,
      EpsilonFeatureFusion fusion) {
    FusionRecipe.Constant[] weights = new FusionRecipe.Constant[inputs.length];
    for (int component = 0; component < inputs.length; component++) {
      weights[component] =
          builder.addConstant(
              "weight" + component,
              FusionRecipe.TensorSpec.fixed(
                  dataType, fusion.outputSize(), fusion.inputSize(component)));
      affine.addTerm(inputs[component], weights[component]);
    }
    return weights;
  }

  private FusionRecipe.Constant addBias(FusionRecipe.Builder builder, int outputSize) {
    return builder.addConstant("bias", FusionRecipe.TensorSpec.fixed(dataType, outputSize));
  }

  private FusionConstantBindings.Builder bindProjectionConstants(
      FusionRecipe recipe,
      FusionRecipe.Constant[] weights,
      FusionRecipe.Constant bias,
      EpsilonFeatureFusion fusion) {
    FusionConstantBindings.Builder bindings = FusionConstantBindings.builder(recipe);
    for (int component = 0; component < weights.length; component++) {
      bindings.bind(weights[component], weight(fusion.projection(component)));
    }
    Parameter biasParameter = fusion.projection(0).getDirectParameters().get("bias");
    bindings.bind(
        bias, parameterStore.getValue(biasParameter, constantsManager.getDevice(), false));
    return bindings;
  }

  private AffineProgram prepare(
      Site site,
      int capacity,
      FusionRecipe.Dimension dimension,
      FusionRecipe.Input[] inputs,
      FusionRecipe.Output output,
      FusionRecipe recipe,
      FusionConstantBindings.Builder bindings) {
    long startedNanos = System.nanoTime();
    FusionPlan plan = null;
    FusionExecutable executable = null;
    FusionSession session = null;
    try {
      plan = compiler.prepare(recipe, FusionCompileConfig.defaults());
      FusionCompilationReport report = plan.getCompilationReport();
      if (!report.isNativeOnly()) {
        throw new IllegalStateException("AFFINE_SUM plan is not native-only: " + recipe.getName());
      }
      executable = plan.bind(bindings.build());
      session =
          executable.newSession(
              constantsManager,
              FusionSessionConfig.builder().optOutputSlotCount(executionSlots).build());
      LOGGER.info(
          "Decision affine fusion prepared: site={}, capacity={}, dtype={}, sourceDtypes={},"
              + " backend={}, commands={}, executableStorageBytes={}, executionStorageBytes={},"
              + " retainedSessionStorageBytes={}, requiredExecutionLaneStorageBytes={},"
              + " workspaceBytes={}, nativeOnly={}, compileMs={}",
          site,
          capacity,
          dataType,
          Arrays.stream(inputs).map(input -> input.getSpec().getDataType()).toList(),
          report.getBackend(),
          report.getCommandCount(),
          report.getExecutableStorageBytes(),
          report.getExecutionStorageBytes(),
          report.getRetainedSessionStorageBytes(executionSlots),
          report.getRequiredExecutionLaneStorageBytes(),
          report.getWorkspaceBytes(),
          report.isNativeOnly(),
          (System.nanoTime() - startedNanos) / 1_000_000.0);
      return new AffineProgram(
          site,
          capacity,
          dimension,
          inputs,
          output,
          plan,
          executable,
          session,
          report.getExecutableStorageBytes(),
          report.getRetainedSessionStorageBytes(executionSlots),
          report.getRequiredExecutionLaneStorageBytes(),
          report.getWorkspaceBytes());
    } catch (RuntimeException | Error prepareFailure) {
      closeResource(prepareFailure, session);
      closeResource(prepareFailure, executable);
      closeResource(prepareFailure, plan);
      throw prepareFailure;
    }
  }

  /** 列部分領域は非連続ビューになるため、ネイティブ定数関連付け用に構築時の一度だけ連続化する。 */
  private NDArray[] splitAlternativeWeights(IdEmbedding alternativeOffset) {
    NDArray weight = weight(alternativeHidden);
    NDArray[] slices = new NDArray[3];
    for (int component = 0; component < 2; component++) {
      NDArray view = weight.get(":,{}:{}", component * hiddenSize, (component + 1) * hiddenSize);
      try {
        slices[component] = constantsManager.zeros(view.getShape(), view.getDataType());
        slices[component].set(new NDIndex(":,:"), view);
        slices[component].attach(constantsManager);
      } finally {
        view.close();
      }
    }
    NDArray offset =
        parameterStore.getValue(
            alternativeOffset.getDirectParameters().get("embedding"),
            constantsManager.getDevice(),
            false);
    requireType(offset, "alternativeOffset.embedding");
    NDArray transposedOffset = offset.transpose();
    try {
      slices[2] =
          constantsManager.zeros(transposedOffset.getShape(), transposedOffset.getDataType());
      slices[2].set(new NDIndex(":,:"), transposedOffset);
      slices[2].attach(constantsManager);
    } finally {
      transposedOffset.close();
    }
    return slices;
  }

  private NDArray weight(Linear linear) {
    Parameter parameter = linear.getDirectParameters().get("weight");
    return parameterStore.getValue(parameter, constantsManager.getDevice(), false);
  }

  private void validateParameterTypes() {
    validateFusionTypes(candidateFusion, "candidateFusion");
    validateFusionTypes(transitionFusion, "transitionFusion");
    requireType(weight(alternativeHidden), "alternativeHidden.weight");
    Parameter alternativeBias = alternativeHidden.getDirectParameters().get("bias");
    requireType(
        parameterStore.getValue(alternativeBias, constantsManager.getDevice(), false),
        "alternativeHidden.bias");
  }

  private void validateFusionTypes(EpsilonFeatureFusion fusion, String label) {
    for (int component = 0; component < fusion.componentCount(); component++) {
      requireType(
          weight(fusion.projection(component)), label + ".component" + component + ".weight");
    }
    Parameter bias = fusion.projection(0).getDirectParameters().get("bias");
    requireType(
        parameterStore.getValue(bias, constantsManager.getDevice(), false),
        label + ".component0.bias");
  }

  private void requireType(NDArray array, String label) {
    if (array.getDataType() != dataType) {
      throw new IllegalStateException(label + " dtype differs from " + dataType);
    }
  }

  private static void requireFloatingType(DataType dataType) {
    if (dataType != DataType.FLOAT16
        && dataType != DataType.BFLOAT16
        && dataType != DataType.FLOAT32) {
      throw new IllegalArgumentException("AFFINE_SUM requires a floating-point dtype: " + dataType);
    }
  }

  private static DataType[] inputDataTypes(NDArray[] inputs) {
    DataType[] dataTypes = new DataType[inputs.length];
    for (int index = 0; index < inputs.length; index++) {
      dataTypes[index] = inputs[index].getDataType();
      requireFloatingType(dataTypes[index]);
    }
    return dataTypes;
  }

  private void requireUsable() {
    if (closed) {
      throw new IllegalStateException("affine execution is closed");
    }
    if (failure != null) {
      throw new IllegalStateException("affine execution is poisoned", failure);
    }
  }

  private int freeForwardSlot() {
    for (int slot = 0; slot < activeForwards.length; slot++) {
      if (activeForwards[slot] == null) {
        return slot;
      }
    }
    throw new IllegalStateException("no affine execution slot is available");
  }

  private boolean hasActiveForward() {
    for (FusionForward forward : activeForwards) {
      if (forward != null) {
        return true;
      }
    }
    return false;
  }

  private void finishForward(int slot, FusionForward forward) {
    if (activeForwards[slot] != forward) {
      throw new IllegalStateException("affine forward does not own its execution slot");
    }
    activeForwards[slot] = null;
  }

  private static Throwable releasePoisonedLease(Throwable failure, AffineProgram program) {
    return program == null ? failure : program.releaseIncompleteSubmissionAfterCompletion(failure);
  }

  private void poison(Throwable poisonCause, boolean hasIncompleteWork) {
    if (failure == null) {
      failure = poisonCause;
    }
    incomplete |= hasIncompleteWork;
  }

  private boolean programHasIncompleteWork() {
    if (transitionProgram != null && transitionProgram.hasIncompleteWork()) {
      return true;
    }
    if (alternativeProgram != null && alternativeProgram.hasIncompleteWork()) {
      return true;
    }
    if (Arrays.stream(candidatePrograms)
        .anyMatch(program -> program != null && program.hasIncompleteWork())) {
      return true;
    }
    return retiredPrograms.stream().anyMatch(AffineProgram::hasIncompleteWork);
  }

  /** 実行中利用権を持ち得る旧容量を新規投入対象から外す。 */
  private void retireForGrowth(AffineProgram program) {
    if (program != null) {
      program.retire();
      if (!program.isReleased()) {
        retiredPrograms.add(program);
      }
    }
  }

  private void recordCacheMiss(AffineProgram current) {
    cacheMisses++;
    if (current != null) {
      growths++;
    }
  }

  private void recordSubmit(Site site, int activeExtent, long startedNanos) {
    switch (site) {
      case CANDIDATE -> candidateCalls++;
      case TRANSITION -> transitionCalls++;
      case ALTERNATIVE -> alternativeCalls++;
    }
    activeLeadingElements += activeExtent;
    submitNanos += System.nanoTime() - startedNanos;
  }

  private int programCount() {
    int count = transitionProgram == null ? 0 : 1;
    count += alternativeProgram == null ? 0 : 1;
    for (AffineProgram program : candidatePrograms) {
      if (program != null) {
        count++;
      }
    }
    for (AffineProgram program : retiredPrograms) {
      if (!program.isReleased()) {
        count++;
      }
    }
    return count;
  }

  private long currentPlanRetainedSessionStorageBytes() {
    long bytes = transitionProgram == null ? 0L : transitionProgram.retainedSessionStorageBytes();
    bytes += alternativeProgram == null ? 0L : alternativeProgram.retainedSessionStorageBytes();
    for (AffineProgram program : candidatePrograms) {
      if (program != null) {
        bytes += program.retainedSessionStorageBytes();
      }
    }
    for (AffineProgram program : retiredPrograms) {
      bytes += program.retainedSessionStorageBytes();
    }
    return bytes;
  }

  private long currentPlanExecutableStorageBytes() {
    long bytes = transitionProgram == null ? 0L : transitionProgram.executableStorageBytes();
    bytes += alternativeProgram == null ? 0L : alternativeProgram.executableStorageBytes();
    for (AffineProgram program : candidatePrograms) {
      if (program != null) {
        bytes += program.executableStorageBytes();
      }
    }
    for (AffineProgram program : retiredPrograms) {
      bytes += program.executableStorageBytes();
    }
    return bytes;
  }

  // 現在の実行計画要求を共有なしで合算する。解放済み実行計画が実行系に残した容量は含まれず、
  // 既存実行系の実常駐量に対する上限ではない。
  private long currentPlanIsolatedLaneStorageUpperBoundBytes() {
    long bytes =
        transitionProgram == null ? 0L : transitionProgram.requiredExecutionLaneStorageBytes();
    bytes +=
        alternativeProgram == null ? 0L : alternativeProgram.requiredExecutionLaneStorageBytes();
    for (AffineProgram program : candidatePrograms) {
      if (program != null) {
        bytes += program.requiredExecutionLaneStorageBytes();
      }
    }
    for (AffineProgram program : retiredPrograms) {
      bytes += program.requiredExecutionLaneStorageBytes();
    }
    return bytes;
  }

  private long currentPlanLogicalWorkspaceBytes() {
    long bytes = transitionProgram == null ? 0L : transitionProgram.workspaceBytes();
    bytes += alternativeProgram == null ? 0L : alternativeProgram.workspaceBytes();
    for (AffineProgram program : candidatePrograms) {
      if (program != null) {
        bytes += program.workspaceBytes();
      }
    }
    for (AffineProgram program : retiredPrograms) {
      bytes += program.workspaceBytes();
    }
    return bytes;
  }

  private static int grownCapacity(int required, int maximum) {
    if (required <= 0 || required > maximum) {
      throw new IllegalArgumentException(
          "active affine extent is outside configured maximum: " + required + "/" + maximum);
    }
    long rounded = 1L;
    while (rounded < required) {
      rounded <<= 1;
    }
    return Math.toIntExact(Math.min(maximum, rounded));
  }

  private enum Site {
    TRANSITION,
    CANDIDATE,
    ALTERNATIVE
  }

  private static final class FusionForward extends Forward {

    private final DecisionPolicyFusionAffineExecution owner;
    private final int slot;
    private final LeasedOutput[] outputs;
    private NDManager workingManager;
    private int outputCount;
    private boolean sealed;
    private boolean finished;

    private FusionForward(
        DecisionPolicyFusionAffineExecution owner, int slot, NDManager workingManager) {
      this.owner = owner;
      this.slot = slot;
      outputs = new LeasedOutput[Site.values().length];
      this.workingManager = workingManager;
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
      return owner.submitCandidate(
          this, actionFeatures, primaryTiles, transitions, playerContexts, stateContext);
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
      return owner.submitTransition(
          this, features, tileContexts, discardTiles, discardSafety, stateContexts);
    }

    @Override
    public NDArray alternative(
        ParameterStore parameterStore,
        PairList<String, Object> runtimeParameters,
        NDArray stateContext,
        NDArray candidateContexts) {
      return owner.submitAlternative(this, stateContext, candidateContexts);
    }

    private NDArray add(Site site, LeasedOutput output) {
      requireUnused(site);
      int index = site.ordinal();
      outputs[index] = output;
      outputCount++;
      return output.array();
    }

    private void requireUnused(Site site) {
      if (outputs[site.ordinal()] != null) {
        throw new IllegalStateException("affine site was submitted twice: " + site);
      }
    }

    /** FusionInvocationが参照する動的入力を、最終D2H完了までバッチ所有に移す。 */
    private void retainInputs(NDArray... arrays) {
      for (NDArray array : arrays) {
        if (array.getManager() != workingManager) {
          array.attach(workingManager);
        }
      }
    }

    private int outputCount() {
      return outputCount;
    }

    private NDManager workingManager() {
      return workingManager;
    }

    @Override
    public AutoCloseable seal() {
      if (finished || sealed) {
        throw new IllegalStateException("affine forward cannot be sealed twice");
      }
      if (outputCount != outputs.length) {
        throw new IllegalStateException(
            "affine forward did not execute all sites: " + outputCount + "/" + outputs.length);
      }
      sealed = true;
      return this;
    }

    @Override
    public void poison(Throwable failure) {
      if (finished) {
        return;
      }
      owner.poison(failure, outputCount > 0 || owner.programHasIncompleteWork());
      if (!owner.hasIncompleteWork()) {
        workingManager = null;
        finished = true;
        owner.finishForward(slot, this);
      }
    }

    @Override
    public void close() {
      if (finished) {
        return;
      }
      Throwable closeFailure = null;
      for (int index = outputs.length - 1; index >= 0; index--) {
        LeasedOutput output = outputs[index];
        if (output != null) {
          try {
            output.close();
            outputs[index] = null;
            outputCount--;
          } catch (Throwable failure) {
            closeFailure = addFailure(closeFailure, failure);
          }
        }
      }
      if (outputCount == 0) {
        workingManager = null;
        finished = true;
        owner.finishForward(slot, this);
      }
      rethrow(closeFailure);
    }
  }

  private static final class AffineProgram implements AutoCloseable {

    private final Site site;
    private final int capacity;
    private final FusionRecipe.Dimension dimension;
    private final FusionRecipe.Input[] inputs;
    private final FusionRecipe.Output output;
    private FusionPlan plan;
    private FusionExecutable executable;
    private FusionSession session;
    private final long executableStorageBytes;
    private final long retainedSessionStorageBytes;
    private final long requiredExecutionLaneStorageBytes;
    private final long workspaceBytes;
    private FusionOutputLease poisonedLease;
    private int activeLeases;
    private boolean unknownSubmission;
    private boolean retired;

    private AffineProgram(
        Site site,
        int capacity,
        FusionRecipe.Dimension dimension,
        FusionRecipe.Input[] inputs,
        FusionRecipe.Output output,
        FusionPlan plan,
        FusionExecutable executable,
        FusionSession session,
        long executableStorageBytes,
        long retainedSessionStorageBytes,
        long requiredExecutionLaneStorageBytes,
        long workspaceBytes) {
      this.site = site;
      this.capacity = capacity;
      this.dimension = dimension;
      this.inputs = inputs;
      this.output = output;
      this.plan = plan;
      this.executable = executable;
      this.session = session;
      this.executableStorageBytes = executableStorageBytes;
      this.retainedSessionStorageBytes = retainedSessionStorageBytes;
      this.requiredExecutionLaneStorageBytes = requiredExecutionLaneStorageBytes;
      this.workspaceBytes = workspaceBytes;
    }

    private int capacity() {
      return capacity;
    }

    private long retainedSessionStorageBytes() {
      return isReleased() ? 0L : retainedSessionStorageBytes;
    }

    private long executableStorageBytes() {
      return isReleased() ? 0L : executableStorageBytes;
    }

    private long requiredExecutionLaneStorageBytes() {
      return isReleased() ? 0L : requiredExecutionLaneStorageBytes;
    }

    private long workspaceBytes() {
      return isReleased() ? 0L : workspaceBytes;
    }

    private boolean isReleased() {
      return session == null && executable == null && plan == null;
    }

    private boolean hasIncompleteWork() {
      return poisonedLease != null || unknownSubmission;
    }

    private LeasedOutput submit(NDManager workingManager, int activeExtent, NDArray... arrays) {
      for (int input = 0; input < inputs.length; input++) {
        DataType expected = inputs[input].getSpec().getDataType();
        DataType actual = arrays[input].getDataType();
        if (actual != expected) {
          throw new IllegalArgumentException(
              "AFFINE_SUM "
                  + site
                  + " input "
                  + inputs[input].getName()
                  + " dtype differs from its recipe: "
                  + actual
                  + "/"
                  + expected);
        }
      }
      FusionOutputLease lease = null;
      NDArray active = null;
      boolean submitAttempted = false;
      try (FusionInvocation invocation = session.acquire()) {
        for (int input = 0; input < inputs.length; input++) {
          invocation.setInput(inputs[input], arrays[input]);
        }
        invocation.setDimension(dimension, activeExtent);
        submitAttempted = true;
        lease = invocation.submit();
        NDArray storage = lease.get(output);
        active = storage.get(NDIndex.sliceAxis(0, 0, activeExtent));
        active.attach(workingManager);
        activeLeases++;
        return new LeasedOutput(active, lease, this);
      } catch (RuntimeException | Error submitFailure) {
        if (lease != null) {
          releaseFailedSubmission(lease, submitFailure);
        } else if (submitAttempted) {
          unknownSubmission = true;
        }
        closeResource(submitFailure, active);
        throw submitFailure;
      }
    }

    private void releaseFailedSubmission(FusionOutputLease lease, Throwable failure) {
      try {
        lease.synchronize();
        lease.close();
      } catch (RuntimeException | Error completionFailure) {
        failure.addSuppressed(completionFailure);
        poisonedLease = lease;
      }
    }

    private Throwable releaseIncompleteSubmissionAfterCompletion(Throwable failure) {
      unknownSubmission = false;
      if (poisonedLease != null) {
        try {
          poisonedLease.close();
          poisonedLease = null;
        } catch (Throwable closeFailure) {
          return addFailure(failure, closeFailure);
        }
      }
      if (retired && activeLeases == 0) {
        releaseResources();
      }
      return failure;
    }

    /** 容量拡張後は新規投入を止め、最後の利用権返却時にネイティブ資源を解放する。 */
    private void retire() {
      retired = true;
      if (activeLeases == 0 && !hasIncompleteWork()) {
        releaseResources();
      }
    }

    private void leaseReleased() {
      activeLeases--;
      if (retired && activeLeases == 0 && !hasIncompleteWork()) {
        releaseResources();
      }
    }

    @Override
    public void close() {
      if (hasIncompleteWork()) {
        throw new IllegalStateException("affine program has incomplete work: " + site);
      }
      if (activeLeases != 0) {
        throw new IllegalStateException("affine program has active leases: " + site);
      }
      releaseResources();
    }

    private void releaseResources() {
      if (session != null) {
        session.close();
        session = null;
      }
      if (executable != null) {
        executable.close();
        executable = null;
      }
      if (plan != null) {
        plan.close();
        plan = null;
      }
    }
  }

  private static final class LeasedOutput implements AutoCloseable {

    private final NDArray array;
    private final AffineProgram program;
    private FusionOutputLease lease;
    private boolean arrayReleased;

    private LeasedOutput(NDArray array, FusionOutputLease lease, AffineProgram program) {
      this.array = array;
      this.lease = lease;
      this.program = program;
    }

    private NDArray array() {
      return array;
    }

    @Override
    public void close() {
      if (!arrayReleased) {
        array.close();
        arrayReleased = true;
      }
      if (lease != null) {
        lease.close();
        lease = null;
        program.leaseReleased();
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
