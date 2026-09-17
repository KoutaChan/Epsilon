package com.epsilon.nano.ai.decision.policy.fusion;

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
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.epsilon.nano.ai.decision.policy.DecisionPolicyIndexedAffineExecution;
import com.epsilon.nano.ai.decision.policy.EpsilonBinaryBranchGate;
import java.util.ArrayList;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * djl-rocm の INDEXED_AFFINE を使い、方策の二分岐を有効な行だけで計算する。
 *
 * <p>リーチ、鳴き、ロン、槓、九種九牌、ツモの実行計画を個別に保持する。出力の最大行数と作業領域の容量は別々に拡張する。有効な行がない場合はゼロのテンソルを返し、実行計画を作らない。
 */
public final class DecisionPolicyFusionIndexedAffineExecution
    extends DecisionPolicyIndexedAffineExecution {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(DecisionPolicyFusionIndexedAffineExecution.class);

  private final NDManager constantsManager;
  private final FusionCompiler compiler;
  private final ParameterStore parameterStore;
  private final EpsilonBinaryBranchGate[] gates;
  private final DataType dataType;
  private final int hiddenSize;
  private final int maxBatch;
  private final int executionSlots;
  private final GateProgram[] rowPrograms;
  private final GateProgram[] riichiPrograms;
  private final FusionForward[] activeForwards;
  private final ArrayList<GateProgram> retiredPrograms;
  private final long[] emptySubmitsBySite;
  private Throwable failure;
  private long totalSubmits;
  private long submitNanos;
  private long activePositions;
  private long emptySubmits;
  private long cacheHits;
  private long cacheMisses;
  private long growths;
  private boolean incomplete;
  private boolean closed;
  private boolean summaryLogged;
  private boolean constantsReleased;
  private boolean resourcesReleased;

  public DecisionPolicyFusionIndexedAffineExecution(
      NDManager manager,
      ParameterStore parameterStore,
      EpsilonBinaryBranchGate riichiGate,
      EpsilonBinaryBranchGate callGate,
      EpsilonBinaryBranchGate ronGate,
      EpsilonBinaryBranchGate kanGate,
      EpsilonBinaryBranchGate kyushuGate,
      EpsilonBinaryBranchGate tsumoGate,
      int hiddenSize,
      DataType expectedDataType,
      int maxBatch,
      int executionSlots) {
    if (executionSlots <= 0) {
      throw new IllegalArgumentException("executionSlots must be positive");
    }
    this.parameterStore = parameterStore;
    gates =
        new EpsilonBinaryBranchGate[] {
          riichiGate, callGate, ronGate, kanGate, kyushuGate, tsumoGate
        };
    this.hiddenSize = hiddenSize;
    this.maxBatch = maxBatch;
    this.executionSlots = executionSlots;
    constantsManager = manager.newSubManager();
    rowPrograms = new GateProgram[Site.values().length];
    int maximumActionCapacity =
        DecisionInputSchema.LEGAL_ACTION_BUCKETS[
            DecisionInputSchema.LEGAL_ACTION_BUCKETS.length - 1];
    riichiPrograms = new GateProgram[maximumActionCapacity + 1];
    activeForwards = new FusionForward[executionSlots];
    retiredPrograms = new ArrayList<>();
    emptySubmitsBySite = new long[Site.values().length];
    try {
      compiler = manager.getEngine().newFusionCompiler(manager.getDevice());
      dataType = weight(riichiGate.hidden()).getDataType();
      requireFloatingType(dataType);
      if (dataType != expectedDataType) {
        throw new IllegalArgumentException(
            "INDEXED_AFFINE parameter dtype differs from compute precision: "
                + dataType
                + "/"
                + expectedDataType);
      }
      validateParameterTypes();
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
    for (GateProgram program : rowPrograms) {
      releaseFailure = releasePoisonedLease(releaseFailure, program);
    }
    for (GateProgram program : riichiPrograms) {
      releaseFailure = releasePoisonedLease(releaseFailure, program);
    }
    for (GateProgram program : retiredPrograms) {
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
          "cannot release poisoned indexed affine resources without final output completion",
          failure);
    }
    if (hasActiveForward()) {
      throw new IllegalStateException(
          "cannot close indexed affine execution with an active forward");
    }
    logSummary();
    Throwable closeFailure = null;
    closeFailure = closePrograms(closeFailure, riichiPrograms);
    closeFailure = closePrograms(closeFailure, rowPrograms);
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

  private NDArray submitRows(
      FusionForward owner,
      Site site,
      NDArray state,
      NDArray baselineBranch,
      NDArray selectedBranch,
      NDArray indices,
      int rowCount) {
    owner.requireUnused(site);
    try {
      long startedNanos = System.nanoTime();
      if (indices.size() == 0) {
        NDArray result = owner.workingManager().zeros(new Shape(rowCount, 1), state.getDataType());
        owner.addUnleased(site);
        recordEmptySubmit(site, startedNanos);
        return result;
      }
      NDArray[] inputs = owner.inputs(site, indices, state, baselineBranch, selectedBranch);
      int activeCount = Math.toIntExact(indices.size());
      GateProgram program = rowProgram(site, rowCount, activeCount, inputs);
      owner.retainInputs(inputs);
      LeasedOutput leased =
          program.submit(owner.workingManager(), rowCount, rowCount, activeCount, inputs);
      NDArray result = owner.addLeased(site, leased);
      recordSubmit(indices.size(), startedNanos);
      return result;
    } catch (RuntimeException | Error submitFailure) {
      poison(submitFailure, owner.leasedOutputCount() > 0 || programHasIncompleteWork());
      throw submitFailure;
    }
  }

  private NDArray submitActions(
      FusionForward owner,
      NDArray state,
      NDArray baselineBranch,
      NDArray selectedBranch,
      NDArray indices,
      int rowCount,
      int actionCapacity) {
    Site site = Site.RIICHI;
    owner.requireUnused(site);
    try {
      long startedNanos = System.nanoTime();
      if (indices.size() == 0) {
        NDArray result =
            owner.workingManager().zeros(new Shape(rowCount, actionCapacity), state.getDataType());
        owner.addUnleased(site);
        recordEmptySubmit(site, startedNanos);
        return result;
      }
      int denseRows = Math.multiplyExact(rowCount, actionCapacity);
      NDArray[] inputs =
          owner.inputs(
              site,
              indices,
              state,
              baselineBranch.reshape(denseRows, hiddenSize),
              selectedBranch.reshape(denseRows, hiddenSize));
      int activeCount = Math.toIntExact(indices.size());
      GateProgram program = riichiProgram(actionCapacity, rowCount, activeCount, inputs);
      owner.retainInputs(inputs);
      LeasedOutput leased =
          program.submit(owner.workingManager(), rowCount, denseRows, activeCount, inputs);
      NDArray result = owner.addLeased(site, leased).reshape(rowCount, actionCapacity);
      recordSubmit(indices.size(), startedNanos);
      return result;
    } catch (RuntimeException | Error submitFailure) {
      poison(submitFailure, owner.leasedOutputCount() > 0 || programHasIncompleteWork());
      throw submitFailure;
    }
  }

  private GateProgram rowProgram(Site site, int rows, int activeRows, NDArray[] inputs) {
    GateProgram current = rowPrograms[site.ordinal()];
    if (current != null
        && current.batchCapacity() >= rows
        && current.activeCapacity() >= activeRows) {
      current.requireInputTypes(inputs);
      cacheHits++;
      return current;
    }
    recordCacheMiss(current);
    int batchCapacity = grownCapacity(rows, maxBatch);
    int activeCapacity = grownCapacity(activeRows, batchCapacity);
    if (current != null) {
      batchCapacity = Math.max(batchCapacity, current.batchCapacity());
      activeCapacity = Math.max(activeCapacity, current.activeCapacity());
    }
    retireForGrowth(current);
    rowPrograms[site.ordinal()] = null;
    GateProgram created =
        createProgram(site, 1, batchCapacity, activeCapacity, inputDataTypes(inputs));
    rowPrograms[site.ordinal()] = created;
    return created;
  }

  private GateProgram riichiProgram(
      int actionCapacity, int rows, int activeRows, NDArray[] inputs) {
    if (actionCapacity <= 0 || actionCapacity >= riichiPrograms.length) {
      throw new IllegalArgumentException("unsupported RIICHI action capacity: " + actionCapacity);
    }
    GateProgram current = riichiPrograms[actionCapacity];
    if (current != null
        && current.batchCapacity() >= rows
        && current.activeCapacity() >= activeRows) {
      current.requireInputTypes(inputs);
      cacheHits++;
      return current;
    }
    recordCacheMiss(current);
    int batchCapacity = grownCapacity(rows, maxBatch);
    int destinationCapacity = Math.multiplyExact(batchCapacity, actionCapacity);
    int activeCapacity = grownCapacity(activeRows, destinationCapacity);
    if (current != null) {
      batchCapacity = Math.max(batchCapacity, current.batchCapacity());
      activeCapacity = Math.max(activeCapacity, current.activeCapacity());
    }
    retireForGrowth(current);
    riichiPrograms[actionCapacity] = null;
    GateProgram created =
        createProgram(
            Site.RIICHI, actionCapacity, batchCapacity, activeCapacity, inputDataTypes(inputs));
    riichiPrograms[actionCapacity] = created;
    return created;
  }

  private GateProgram createProgram(
      Site site,
      int actionCapacity,
      int batchCapacity,
      int activeCapacity,
      DataType[] inputDataTypes) {
    int destinationCapacity = Math.multiplyExact(batchCapacity, actionCapacity);
    FusionRecipe.Builder builder =
        FusionRecipe.builder(
            "epsilon-decision-indexed-affine-"
                + site.name().toLowerCase()
                + "-a"
                + actionCapacity
                + "-b"
                + batchCapacity);
    FusionRecipe.Dimension batch = builder.addDimension("batch", batchCapacity);
    FusionRecipe.Dimension destination =
        actionCapacity == 1 ? batch : builder.addDimension("destination", destinationCapacity);
    FusionRecipe.Dimension active = builder.addDimension("active", activeCapacity);
    FusionRecipe.Input indices =
        builder.addInput("activePositions", FusionRecipe.TensorSpec.of(inputDataTypes[0], active));
    FusionRecipe.Input state =
        builder.addInput("state", FusionRecipe.TensorSpec.of(inputDataTypes[1], batch, hiddenSize));
    FusionRecipe.Input baseline =
        builder.addInput(
            "baseline", FusionRecipe.TensorSpec.of(inputDataTypes[2], destination, hiddenSize));
    FusionRecipe.Input selected =
        builder.addInput(
            "selected", FusionRecipe.TensorSpec.of(inputDataTypes[3], destination, hiddenSize));

    EpsilonBinaryBranchGate gate = gate(site);
    Linear hidden = gate.hidden();
    Linear score = gate.scoreHead();
    FusionRecipe.Constant hiddenWeight =
        builder.addConstant(
            "hiddenWeight", FusionRecipe.TensorSpec.fixed(dataType, hiddenSize, hiddenSize * 3L));
    FusionRecipe.Constant hiddenBias =
        builder.addConstant("hiddenBias", FusionRecipe.TensorSpec.fixed(dataType, hiddenSize));
    FusionRecipe.Constant scoreWeight =
        builder.addConstant("scoreWeight", FusionRecipe.TensorSpec.fixed(dataType, 1, hiddenSize));
    FusionRecipe.Constant scoreBias =
        builder.addConstant("scoreBias", FusionRecipe.TensorSpec.fixed(dataType, 1));
    FusionRecipe.IndexedAffine value =
        builder
            .indexedAffine("score", indices, destination)
            .addSource(state, actionCapacity)
            .addSource(baseline, 1)
            .addSource(selected, 1)
            .setHiddenWeight(hiddenWeight)
            .optHiddenBias(hiddenBias)
            .optActivation(FusionRecipe.Activation.SILU)
            .setOutputWeight(scoreWeight)
            .optOutputBias(scoreBias)
            .build();
    FusionRecipe.Output output = builder.addOutput("score", value);
    FusionRecipe recipe = builder.build();
    FusionConstantBindings.Builder bindings = FusionConstantBindings.builder(recipe);
    bindings
        .bind(hiddenWeight, weight(hidden))
        .bind(hiddenBias, bias(hidden))
        .bind(scoreWeight, weight(score))
        .bind(scoreBias, bias(score));
    return prepare(
        site,
        actionCapacity,
        batchCapacity,
        activeCapacity,
        batch,
        destination,
        active,
        new FusionRecipe.Input[] {indices, state, baseline, selected},
        output,
        recipe,
        bindings);
  }

  private GateProgram prepare(
      Site site,
      int actionCapacity,
      int batchCapacity,
      int activeCapacity,
      FusionRecipe.Dimension batch,
      FusionRecipe.Dimension destination,
      FusionRecipe.Dimension active,
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
        throw new IllegalStateException(
            "INDEXED_AFFINE plan is not native-only: " + recipe.getName());
      }
      executable = plan.bind(bindings.build());
      session =
          executable.newSession(
              constantsManager,
              FusionSessionConfig.builder().optOutputSlotCount(executionSlots).build());
      LOGGER.info(
          "Decision indexed affine fusion prepared: site={}, actionCapacity={}, batchCapacity={},"
              + " activeCapacity={}, dtype={}, inputDtypes={}, backend={}, commands={},"
              + " executableStorageBytes={}, executionStorageBytes={},"
              + " retainedSessionStorageBytes={}, requiredExecutionLaneStorageBytes={},"
              + " workspaceBytes={}, nativeOnly={}, compileMs={}",
          site,
          actionCapacity,
          batchCapacity,
          activeCapacity,
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
      return new GateProgram(
          site,
          actionCapacity,
          batchCapacity,
          activeCapacity,
          batch,
          destination,
          active,
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

  private EpsilonBinaryBranchGate gate(Site site) {
    return gates[site.ordinal()];
  }

  private NDArray weight(Linear linear) {
    return parameterStore.getValue(
        linear.getDirectParameters().get("weight"), constantsManager.getDevice(), false);
  }

  private NDArray bias(Linear linear) {
    Parameter parameter = linear.getDirectParameters().get("bias");
    if (parameter == null) {
      throw new IllegalStateException("indexed affine gate requires a bias");
    }
    return parameterStore.getValue(parameter, constantsManager.getDevice(), false);
  }

  private void validateParameterTypes() {
    for (Site site : Site.values()) {
      EpsilonBinaryBranchGate gate = gate(site);
      requireType(weight(gate.hidden()), site + ".hidden.weight");
      requireType(bias(gate.hidden()), site + ".hidden.bias");
      requireType(weight(gate.scoreHead()), site + ".score.weight");
      requireType(bias(gate.scoreHead()), site + ".score.bias");
    }
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
      throw new IllegalArgumentException(
          "INDEXED_AFFINE requires a floating-point dtype: " + dataType);
    }
  }

  private static DataType[] inputDataTypes(NDArray[] inputs) {
    DataType[] dataTypes = new DataType[inputs.length];
    dataTypes[0] = inputs[0].getDataType();
    if (dataTypes[0] != DataType.INT32 && dataTypes[0] != DataType.INT64) {
      throw new IllegalArgumentException(
          "INDEXED_AFFINE positions require INT32 or INT64: " + dataTypes[0]);
    }
    for (int index = 1; index < inputs.length; index++) {
      dataTypes[index] = inputs[index].getDataType();
      requireFloatingType(dataTypes[index]);
    }
    return dataTypes;
  }

  private void requireUsable() {
    if (closed) {
      throw new IllegalStateException("indexed affine execution is closed");
    }
    if (failure != null) {
      throw new IllegalStateException("indexed affine execution is poisoned", failure);
    }
  }

  private int freeForwardSlot() {
    for (int slot = 0; slot < activeForwards.length; slot++) {
      if (activeForwards[slot] == null) {
        return slot;
      }
    }
    throw new IllegalStateException(
        "indexed affine execution has no free forward slot: " + executionSlots);
  }

  private boolean hasActiveForward() {
    return Arrays.stream(activeForwards).anyMatch(forward -> forward != null);
  }

  private void finishForward(int slot, FusionForward forward) {
    if (activeForwards[slot] != forward) {
      throw new IllegalStateException("indexed affine forward slot ownership was lost: " + slot);
    }
    activeForwards[slot] = null;
  }

  private void poison(Throwable poisonCause, boolean hasIncompleteWork) {
    if (failure == null) {
      failure = poisonCause;
    }
    incomplete |= hasIncompleteWork;
  }

  private boolean programHasIncompleteWork() {
    for (GateProgram program : rowPrograms) {
      if (program != null && program.hasIncompleteWork()) {
        return true;
      }
    }
    for (GateProgram program : riichiPrograms) {
      if (program != null && program.hasIncompleteWork()) {
        return true;
      }
    }
    return retiredPrograms.stream().anyMatch(GateProgram::hasIncompleteWork);
  }

  private static Throwable releasePoisonedLease(Throwable failure, GateProgram program) {
    return program == null ? failure : program.releaseIncompleteSubmissionAfterCompletion(failure);
  }

  /** 実行中利用権を持ち得る旧容量を新規投入対象から外す。 */
  private void retireForGrowth(GateProgram program) {
    if (program != null) {
      program.retire();
      if (!program.isReleased()) {
        retiredPrograms.add(program);
      }
    }
  }

  private void recordCacheMiss(GateProgram current) {
    cacheMisses++;
    if (current != null) {
      growths++;
    }
  }

  private void recordSubmit(long activeCount, long startedNanos) {
    totalSubmits++;
    activePositions += activeCount;
    submitNanos += System.nanoTime() - startedNanos;
  }

  private void recordEmptySubmit(Site site, long startedNanos) {
    totalSubmits++;
    emptySubmits++;
    emptySubmitsBySite[site.ordinal()]++;
    submitNanos += System.nanoTime() - startedNanos;
  }

  private int programCount() {
    int count = 0;
    for (GateProgram program : rowPrograms) {
      count += program == null ? 0 : 1;
    }
    for (GateProgram program : riichiPrograms) {
      count += program == null ? 0 : 1;
    }
    for (GateProgram program : retiredPrograms) {
      if (!program.isReleased()) {
        count++;
      }
    }
    return count;
  }

  private long currentPlanRetainedSessionStorageBytes() {
    return storageBytes(StorageKind.RETAINED_SESSION);
  }

  private long currentPlanExecutableStorageBytes() {
    return storageBytes(StorageKind.EXECUTABLE);
  }

  // 現在の実行計画要求を共有なしで合算する。解放済み実行計画が実行単位に残した容量は含まれず、
  // 既存実行単位の実常駐量に対する上限ではない。
  private long currentPlanIsolatedLaneStorageUpperBoundBytes() {
    return storageBytes(StorageKind.REQUIRED_EXECUTION_LANE);
  }

  private long currentPlanLogicalWorkspaceBytes() {
    return storageBytes(StorageKind.WORKSPACE);
  }

  private long storageBytes(StorageKind kind) {
    long bytes = 0;
    for (GateProgram program : rowPrograms) {
      bytes += program == null ? 0L : program.storageBytes(kind);
    }
    for (GateProgram program : riichiPrograms) {
      bytes += program == null ? 0L : program.storageBytes(kind);
    }
    for (GateProgram program : retiredPrograms) {
      bytes += program.storageBytes(kind);
    }
    return bytes;
  }

  private void logSummary() {
    if (summaryLogged) {
      return;
    }
    LOGGER.info(
        "Decision indexed affine fusion summary: submits={}, submitMs={}, activePositions={},"
            + " emptySubmits={}, emptySubmitsBySite={}, cacheHits={}, cacheMisses={}, growths={},"
            + " programs={}, currentPlanExecutableStorageBytes={},"
            + " currentPlanRetainedSessionStorageBytes={},"
            + " currentPlanIsolatedLaneStorageUpperBoundBytes={},"
            + " currentPlanLogicalWorkspaceBytes={}",
        totalSubmits,
        submitNanos / 1_000_000.0,
        activePositions,
        emptySubmits,
        Arrays.toString(emptySubmitsBySite),
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

  private static Throwable closePrograms(Throwable failure, GateProgram[] programs) {
    for (int index = programs.length - 1; index >= 0; index--) {
      GateProgram program = programs[index];
      if (program == null) {
        continue;
      }
      try {
        program.close();
        programs[index] = null;
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    return failure;
  }

  private static int grownCapacity(int required, int maximum) {
    if (required <= 0 || required > maximum) {
      throw new IllegalArgumentException(
          "indexed affine batch extent is outside configured maximum: " + required + "/" + maximum);
    }
    long rounded = 1L;
    while (rounded < required) {
      rounded <<= 1;
    }
    return Math.toIntExact(Math.min(maximum, rounded));
  }

  private enum StorageKind {
    EXECUTABLE,
    RETAINED_SESSION,
    REQUIRED_EXECUTION_LANE,
    WORKSPACE
  }

  private static final class FusionForward extends Forward {

    private final DecisionPolicyFusionIndexedAffineExecution owner;
    private final int slot;
    private final LeasedOutput[] outputs;
    private final boolean[] executedSites;
    private final NDArray[][] inputs;
    private NDManager workingManager;
    private int executedSiteCount;
    private int leasedOutputCount;
    private boolean sealed;
    private boolean finished;

    private FusionForward(
        DecisionPolicyFusionIndexedAffineExecution owner, int slot, NDManager workingManager) {
      this.owner = owner;
      this.slot = slot;
      outputs = new LeasedOutput[Site.values().length];
      executedSites = new boolean[Site.values().length];
      inputs = new NDArray[Site.values().length][4];
      this.workingManager = workingManager;
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
      return owner.submitRows(
          this, site, state, baselineBranch, selectedBranch, activeRows, Math.toIntExact(rowCount));
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
      return owner.submitActions(
          this,
          state,
          baselineBranch,
          selectedBranch,
          activeActions,
          Math.toIntExact(rowCount),
          actionCapacity);
    }

    private NDArray[] inputs(
        Site site, NDArray indices, NDArray state, NDArray baseline, NDArray selected) {
      NDArray[] values = inputs[site.ordinal()];
      values[0] = indices;
      values[1] = state;
      values[2] = baseline;
      values[3] = selected;
      return values;
    }

    private NDArray addLeased(Site site, LeasedOutput output) {
      requireUnused(site);
      outputs[site.ordinal()] = output;
      executedSites[site.ordinal()] = true;
      executedSiteCount++;
      leasedOutputCount++;
      return output.array();
    }

    /** 有効な位置がない実行箇所を実行済みにし、永続Fusion バッファを借用せずに完了させる。 */
    private void addUnleased(Site site) {
      requireUnused(site);
      executedSites[site.ordinal()] = true;
      executedSiteCount++;
    }

    private void requireUnused(Site site) {
      if (executedSites[site.ordinal()]) {
        throw new IllegalStateException("indexed affine site was submitted twice: " + site);
      }
    }

    private void retainInputs(NDArray[] arrays) {
      for (NDArray array : arrays) {
        if (array.getManager() != workingManager) {
          array.attach(workingManager);
        }
      }
    }

    private int leasedOutputCount() {
      return leasedOutputCount;
    }

    private NDManager workingManager() {
      return workingManager;
    }

    @Override
    public AutoCloseable seal() {
      if (finished || sealed) {
        throw new IllegalStateException("indexed affine forward cannot be sealed twice");
      }
      if (executedSiteCount != outputs.length) {
        throw new IllegalStateException(
            "indexed affine forward did not execute all sites: "
                + executedSiteCount
                + "/"
                + outputs.length);
      }
      sealed = true;
      return this;
    }

    @Override
    public void poison(Throwable failure) {
      if (finished) {
        return;
      }
      owner.poison(failure, leasedOutputCount > 0 || owner.programHasIncompleteWork());
      if (!owner.hasIncompleteWork()) {
        clear();
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
            leasedOutputCount--;
          } catch (Throwable failure) {
            closeFailure = addFailure(closeFailure, failure);
          }
        }
      }
      if (leasedOutputCount == 0) {
        clear();
        finished = true;
        owner.finishForward(slot, this);
      }
      rethrow(closeFailure);
    }

    private void clear() {
      for (NDArray[] siteInputs : inputs) {
        Arrays.fill(siteInputs, null);
      }
      Arrays.fill(executedSites, false);
      workingManager = null;
    }
  }

  private static final class GateProgram implements AutoCloseable {

    private final Site site;
    private final int actionCapacity;
    private final int batchCapacity;
    private final int activeCapacity;
    private final FusionRecipe.Dimension batchDimension;
    private final FusionRecipe.Dimension destinationDimension;
    private final FusionRecipe.Dimension activeDimension;
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

    private GateProgram(
        Site site,
        int actionCapacity,
        int batchCapacity,
        int activeCapacity,
        FusionRecipe.Dimension batchDimension,
        FusionRecipe.Dimension destinationDimension,
        FusionRecipe.Dimension activeDimension,
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
      this.actionCapacity = actionCapacity;
      this.batchCapacity = batchCapacity;
      this.activeCapacity = activeCapacity;
      this.batchDimension = batchDimension;
      this.destinationDimension = destinationDimension;
      this.activeDimension = activeDimension;
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

    private int batchCapacity() {
      return batchCapacity;
    }

    private int activeCapacity() {
      return activeCapacity;
    }

    private long storageBytes(StorageKind kind) {
      if (isReleased()) {
        return 0L;
      }
      return switch (kind) {
        case EXECUTABLE -> executableStorageBytes;
        case RETAINED_SESSION -> retainedSessionStorageBytes;
        case REQUIRED_EXECUTION_LANE -> requiredExecutionLaneStorageBytes;
        case WORKSPACE -> workspaceBytes;
      };
    }

    private boolean isReleased() {
      return session == null && executable == null && plan == null;
    }

    private boolean hasIncompleteWork() {
      return poisonedLease != null || unknownSubmission;
    }

    private void requireInputTypes(NDArray[] arrays) {
      for (int input = 0; input < inputs.length; input++) {
        DataType expected = inputs[input].getSpec().getDataType();
        DataType actual = arrays[input].getDataType();
        if (actual != expected) {
          throw new IllegalArgumentException(
              "INDEXED_AFFINE "
                  + site
                  + " input "
                  + inputs[input].getName()
                  + " dtype differs from its recipe: "
                  + actual
                  + "/"
                  + expected);
        }
      }
    }

    private LeasedOutput submit(
        NDManager workingManager,
        int batchExtent,
        int destinationExtent,
        int activeExtent,
        NDArray[] arrays) {
      requireInputTypes(arrays);
      FusionOutputLease lease = null;
      NDArray active = null;
      boolean submitAttempted = false;
      try (FusionInvocation invocation = session.acquire()) {
        for (int input = 0; input < inputs.length; input++) {
          invocation.setInput(inputs[input], arrays[input]);
        }
        invocation.setDimension(batchDimension, batchExtent);
        if (destinationDimension != batchDimension) {
          invocation.setDimension(destinationDimension, destinationExtent);
        }
        invocation.setDimension(activeDimension, activeExtent);
        submitAttempted = true;
        lease = invocation.submit();
        NDArray storage = lease.get(output);
        active = storage.get(NDIndex.sliceAxis(0, 0, destinationExtent));
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
        throw new IllegalStateException(
            "indexed affine program has incomplete work: " + site + "/" + actionCapacity);
      }
      if (activeLeases != 0) {
        throw new IllegalStateException(
            "indexed affine program has active leases: " + site + "/" + actionCapacity);
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
    private final GateProgram program;
    private FusionOutputLease lease;
    private boolean arrayReleased;

    private LeasedOutput(NDArray array, FusionOutputLease lease, GateProgram program) {
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
