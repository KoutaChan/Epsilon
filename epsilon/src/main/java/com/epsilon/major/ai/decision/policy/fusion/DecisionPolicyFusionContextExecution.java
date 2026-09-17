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
import com.epsilon.major.ai.decision.policy.DecisionPolicyContextExecution;
import java.util.ArrayList;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * djl-rocmのBINARY_BRANCH_BLENDを使い、三つの方策文脈集約を永続実行する。
 *
 * <p>CALL、KAN、KYUSHUは、それぞれ最初に受け取った実入力データ型で独立したプログラムを一度だけ構築する。低精度候補でも、集約の有無により
 * コンテキストと存在マスクの実データ型は処理箇所間で異なり得るため、プログラムは処理箇所間で共有しない。各処理箇所のプログラムは {@code executionSlots}
 * 個のセッションバッファを持ち、同じ順伝播の最終D2Hが完了するまで有効なビュー、入力参照、出力利用権をまとめて保持する。
 */
public final class DecisionPolicyFusionContextExecution extends DecisionPolicyContextExecution {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(DecisionPolicyFusionContextExecution.class);

  private final NDManager resourceManager;
  private final FusionCompiler compiler;
  private final int hiddenSize;
  private final int maximumBatch;
  private final int executionSlots;
  private final Program[] programs;
  private final FusionForward[] activeForwards;
  private final ArrayList<FusionOutputLease> poisonedLeases = new ArrayList<>();
  private Throwable failure;
  private boolean unknownSubmission;
  private boolean incomplete;
  private boolean closed;

  public DecisionPolicyFusionContextExecution(
      NDManager manager,
      int hiddenSize,
      DataType controlDataType,
      int maximumBatch,
      int executionSlots) {
    if (executionSlots <= 0) {
      throw new IllegalArgumentException("executionSlots must be positive");
    }
    requireFloatingType(controlDataType);
    this.hiddenSize = hiddenSize;
    this.maximumBatch = maximumBatch;
    this.executionSlots = executionSlots;
    programs = new Program[Site.values().length];
    activeForwards = new FusionForward[executionSlots];
    resourceManager = manager.newSubManager();

    try {
      compiler = manager.getEngine().newFusionCompiler(manager.getDevice());
      LOGGER.info(
          "Decision binary context fusion: maximumBatch={}, hiddenSize={}, slots={}, "
              + "outputSlots={}, expectedControlDtype={}, programBinding=PER_SITE_ACTUAL_DTYPES",
          maximumBatch,
          hiddenSize,
          executionSlots,
          Math.multiplyExact(Site.values().length, executionSlots),
          controlDataType);
    } catch (RuntimeException | Error constructionFailure) {
      closeResource(constructionFailure, resourceManager);
      throw constructionFailure;
    }
  }

  @Override
  public Forward beginForward(NDManager workingManager) {
    if (closed) {
      throw new IllegalStateException("policy context fusion is closed");
    }
    if (failure != null) {
      throw new IllegalStateException("policy context fusion is poisoned", failure);
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
      FusionOutputLease lease = poisonedLeases.get(index);
      try {
        lease.close();
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
          "cannot close policy context fusion with incomplete device work", failure);
    }
    if (hasActiveForward()) {
      throw new IllegalStateException("cannot close policy context fusion with an active forward");
    }
    Throwable closeFailure = null;
    for (int index = programs.length - 1; index >= 0; index--) {
      Program program = programs[index];
      if (program == null) {
        continue;
      }
      try {
        program.close();
        programs[index] = null;
      } catch (Throwable failure) {
        closeFailure = addFailure(closeFailure, failure);
      }
    }
    if (!hasProgram()) {
      closeFailure = closeResource(closeFailure, resourceManager);
    }
    if (closeFailure == null) {
      closed = true;
    }
    rethrow(closeFailure);
  }

  private LeasedOutput submit(FusionForward owner, Site site, NDArray[] arrays) {
    long rowCount = arrays[0].getShape().get(0);
    Program currentProgram = program(site, arrays);
    FusionOutputLease lease = null;
    NDArray active = null;
    boolean submitAttempted = false;
    try (FusionInvocation invocation = currentProgram.acquire()) {
      for (int inputIndex = 0; inputIndex < arrays.length; inputIndex++) {
        invocation.setInput(currentProgram.input(inputIndex), arrays[inputIndex]);
      }
      invocation.setDimension(currentProgram.batch(), rowCount);
      submitAttempted = true;
      lease = invocation.submit();
      active = lease.get(currentProgram.output()).get(NDIndex.sliceAxis(0, 0, rowCount));
      active.attach(owner.workingManager());
      return new LeasedOutput(active, lease);
    } catch (RuntimeException | Error submitFailure) {
      if (lease != null) {
        try {
          lease.synchronize();
          lease.close();
        } catch (RuntimeException | Error completionFailure) {
          submitFailure.addSuppressed(completionFailure);
          poisonedLeases.add(lease);
          incomplete = true;
        }
      } else if (submitAttempted) {
        unknownSubmission = true;
        incomplete = true;
      }
      closeResource(submitFailure, active);
      poison(submitFailure, submitAttempted);
      throw submitFailure;
    }
  }

  private Program program(Site site, NDArray[] arrays) {
    for (int index = 0; index < arrays.length; index++) {
      requireFloatingType(arrays[index].getDataType());
    }
    int programIndex = site.ordinal();
    Program program = programs[programIndex];
    if (program == null) {
      DataType[] inputTypes = new DataType[arrays.length];
      for (int index = 0; index < arrays.length; index++) {
        inputTypes[index] = arrays[index].getDataType();
      }
      program =
          new Program(
              compiler,
              resourceManager,
              site,
              hiddenSize,
              maximumBatch,
              executionSlots,
              inputTypes);
      programs[programIndex] = program;
    } else if (!program.matches(arrays)) {
      throw new IllegalStateException(
          "binary context input dtypes changed after program binding: site="
              + site
              + ", expected="
              + program.inputTypes()
              + ", actual="
              + inputTypes(arrays));
    }
    return program;
  }

  private boolean hasProgram() {
    for (Program program : programs) {
      if (program != null) {
        return true;
      }
    }
    return false;
  }

  private static String inputTypes(NDArray[] arrays) {
    DataType[] types = new DataType[arrays.length];
    for (int index = 0; index < arrays.length; index++) {
      types[index] = arrays[index].getDataType();
    }
    return Arrays.toString(types);
  }

  private int freeSlot() {
    for (int slot = 0; slot < activeForwards.length; slot++) {
      if (activeForwards[slot] == null) {
        return slot;
      }
    }
    throw new IllegalStateException("no policy context fusion slot is available");
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
      throw new IllegalStateException("policy context fusion slot ownership was lost");
    }
    activeForwards[slot] = null;
  }

  private void poison(Throwable cause, boolean submittedWork) {
    if (failure == null) {
      failure = cause;
    }
    incomplete |= submittedWork;
  }

  private static void requireFloatingType(DataType dataType) {
    if (dataType != DataType.FLOAT16
        && dataType != DataType.BFLOAT16
        && dataType != DataType.FLOAT32) {
      throw new IllegalArgumentException(
          "binary context inputs require a floating-point dtype: " + dataType);
    }
  }

  private static final class Program implements AutoCloseable {

    private final DataType[] inputTypes;
    private final FusionRecipe.Dimension batch;
    private final FusionRecipe.Input[] inputs;
    private final FusionRecipe.Output output;
    private FusionPlan plan;
    private FusionExecutable executable;
    private FusionSession session;

    private Program(
        FusionCompiler compiler,
        NDManager resourceManager,
        Site site,
        int hiddenSize,
        int maximumBatch,
        int executionSlots,
        DataType[] inputTypes) {
      this.inputTypes = inputTypes.clone();
      FusionRecipe.Builder builder = FusionRecipe.builder("epsilon-policy-binary-context-" + site);
      batch = builder.addDimension("batch", maximumBatch);
      FusionRecipe.Input baselineContext =
          builder.addInput(
              "baselineContext", FusionRecipe.TensorSpec.of(inputTypes[0], batch, hiddenSize));
      FusionRecipe.Input selectedContext =
          builder.addInput(
              "selectedContext", FusionRecipe.TensorSpec.of(inputTypes[1], batch, hiddenSize));
      FusionRecipe.Input selectedLogit =
          builder.addInput("selectedLogit", FusionRecipe.TensorSpec.of(inputTypes[2], batch, 1));
      FusionRecipe.Input baselinePresence =
          builder.addInput("baselinePresence", FusionRecipe.TensorSpec.of(inputTypes[3], batch, 1));
      FusionRecipe.Input selectedPresence =
          builder.addInput("selectedPresence", FusionRecipe.TensorSpec.of(inputTypes[4], batch, 1));
      inputs =
          new FusionRecipe.Input[] {
            baselineContext, selectedContext, selectedLogit, baselinePresence, selectedPresence
          };
      FusionRecipe.BinaryBranchBlend blended =
          builder.binaryBranchBlend(
              "blended",
              baselineContext,
              selectedContext,
              selectedLogit,
              baselinePresence,
              selectedPresence);
      output = builder.addOutput("context", blended);
      FusionRecipe recipe = builder.build();
      FusionPlan createdPlan = null;
      FusionExecutable createdExecutable = null;
      FusionSession createdSession = null;
      try {
        createdPlan = compiler.prepare(recipe, FusionCompileConfig.defaults());
        FusionCompilationReport report = createdPlan.getCompilationReport();
        if (!report.isNativeOnly()) {
          throw new IllegalStateException("policy binary context plan is not native-only");
        }
        createdExecutable = createdPlan.bind(FusionConstantBindings.builder(recipe).build());
        createdSession =
            createdExecutable.newSession(
                resourceManager,
                FusionSessionConfig.builder().optOutputSlotCount(executionSlots).build());
        LOGGER.info(
            "Decision binary context fusion program prepared: site={}, inputDtypes={}, backend={},"
                + " commands={}, outputSlots={}, executableStorageBytes={},"
                + " retainedSessionStorageBytes={}, requiredExecutionLaneStorageBytes={}",
            site,
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

    private boolean matches(NDArray[] arrays) {
      if (inputTypes.length != arrays.length) {
        return false;
      }
      for (int index = 0; index < inputTypes.length; index++) {
        if (inputTypes[index] != arrays[index].getDataType()) {
          return false;
        }
      }
      return true;
    }

    private String inputTypes() {
      return Arrays.toString(inputTypes);
    }

    private FusionInvocation acquire() {
      return session.acquire();
    }

    private FusionRecipe.Dimension batch() {
      return batch;
    }

    private FusionRecipe.Input input(int index) {
      return inputs[index];
    }

    private FusionRecipe.Output output() {
      return output;
    }

    @Override
    public void close() {
      Throwable failure = null;
      if (session != null) {
        try {
          session.close();
          session = null;
        } catch (Throwable closeFailure) {
          failure = addFailure(failure, closeFailure);
        }
      }
      if (session == null && executable != null) {
        try {
          executable.close();
          executable = null;
        } catch (Throwable closeFailure) {
          failure = addFailure(failure, closeFailure);
        }
      }
      if (session == null && executable == null && plan != null) {
        try {
          plan.close();
          plan = null;
        } catch (Throwable closeFailure) {
          failure = addFailure(failure, closeFailure);
        }
      }
      rethrow(failure);
    }
  }

  private static final class FusionForward extends Forward {

    private final DecisionPolicyFusionContextExecution owner;
    private final int slot;
    private final NDArray[][] inputs = new NDArray[Site.values().length][5];
    private final LeasedOutput[] outputs = new LeasedOutput[Site.values().length];
    private final boolean[] executedSites = new boolean[Site.values().length];
    private NDManager workingManager;
    private int executedSiteCount;
    private int leasedOutputCount;
    private boolean sealed;
    private boolean finished;

    private FusionForward(
        DecisionPolicyFusionContextExecution owner, int slot, NDManager workingManager) {
      this.owner = owner;
      this.slot = slot;
      this.workingManager = workingManager;
    }

    @Override
    public NDArray mixBinaryBranchContexts(
        Site site,
        NDArray baselineContext,
        NDArray selectedContext,
        NDArray selectedLogit,
        NDArray baselinePresence,
        NDArray selectedPresence) {
      if (finished || sealed || executedSites[site.ordinal()]) {
        throw new IllegalStateException("policy context site cannot execute twice: " + site);
      }
      NDArray[] siteInputs = inputs[site.ordinal()];
      siteInputs[0] = baselineContext;
      siteInputs[1] = selectedContext;
      siteInputs[2] = selectedLogit;
      siteInputs[3] = baselinePresence;
      siteInputs[4] = selectedPresence;
      retainInputs(siteInputs);
      LeasedOutput leased = owner.submit(this, site, siteInputs);
      outputs[site.ordinal()] = leased;
      executedSites[site.ordinal()] = true;
      executedSiteCount++;
      leasedOutputCount++;
      return leased.array();
    }

    private void retainInputs(NDArray[] values) {
      for (NDArray value : values) {
        if (value.getManager() != workingManager) {
          value.attach(workingManager);
        }
      }
    }

    private NDManager workingManager() {
      return workingManager;
    }

    @Override
    public AutoCloseable seal() {
      if (finished || sealed || executedSiteCount != outputs.length) {
        throw new IllegalStateException(
            "policy context forward did not execute all sites: "
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
      owner.poison(failure, leasedOutputCount > 0 || owner.hasIncompleteWork());
      if (!owner.hasIncompleteWork()) {
        clear();
        finished = true;
        owner.finish(slot, this);
      }
    }

    @Override
    public void close() {
      if (finished) {
        return;
      }
      Throwable closeFailure = null;
      for (int index = outputs.length - 1; index >= 0; index--) {
        LeasedOutput leased = outputs[index];
        if (leased == null) {
          continue;
        }
        try {
          leased.close();
          outputs[index] = null;
          leasedOutputCount--;
        } catch (Throwable failure) {
          owner.poison(failure, true);
          closeFailure = addFailure(closeFailure, failure);
        }
      }
      if (leasedOutputCount == 0) {
        clear();
        finished = true;
        owner.finish(slot, this);
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

  private static final class LeasedOutput implements AutoCloseable {

    private final NDArray array;
    private FusionOutputLease lease;
    private boolean arrayReleased;

    private LeasedOutput(NDArray array, FusionOutputLease lease) {
      this.array = array;
      this.lease = lease;
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
