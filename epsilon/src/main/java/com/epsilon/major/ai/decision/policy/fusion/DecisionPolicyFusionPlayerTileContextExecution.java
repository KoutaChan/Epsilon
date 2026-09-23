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
import ai.djl.training.ParameterStore;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.major.ai.decision.policy.DecisionPolicyPlayerTileContextExecution;
import com.epsilon.major.ai.decision.policy.EpsilonPlayerTileContextMixer;
import java.util.ArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 4家の牌別文脈をdjl-rocmのPROJECTED_RESIDUAL_MLPで合成する凍結推論実行。
 *
 * <p>実行計画とパラメーターはパイプライン全体で共有し、各実行枠は最大 {@code [maxBatch,34,context]} の出力と処理段階 作業領域だけを持つ。{@code
 * maxBatch=4096}、{@code context=128}では、実行枠ごとの論理データ本体は FLOAT16/BFLOAT16で約136 MiB、FLOAT32で約272
 * MiBとなる。入力のプレイヤー軸を並べ替える一時値と出力利用権は、同じバッチの最終D2Hが完了するまで {@link FusionForward}が保持する。
 */
public final class DecisionPolicyFusionPlayerTileContextExecution
    extends DecisionPolicyPlayerTileContextExecution {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(DecisionPolicyFusionPlayerTileContextExecution.class);

  private final NDManager resourceManager;
  private final EpsilonPlayerTileContextMixer mixer;
  private final Program program;
  private final FusionForward[] activeForwards;
  private final ArrayList<FusionOutputLease> poisonedLeases = new ArrayList<>();
  private Throwable failure;
  private boolean unknownSubmission;
  private boolean incomplete;
  private boolean closed;

  public DecisionPolicyFusionPlayerTileContextExecution(
      NDManager manager,
      ParameterStore parameterStore,
      EpsilonPlayerTileContextMixer mixer,
      DataType dataType,
      int maximumBatch,
      int executionSlots) {
    this.mixer = mixer;
    activeForwards = new FusionForward[executionSlots];
    resourceManager = manager.newSubManager();
    Program createdProgram = null;
    try {
      Linear combinedProjection = mixer.skipAndHiddenProjection();
      Linear outputProjection = mixer.residualProjection();
      NDArray combinedWeight = value(parameterStore, combinedProjection, "weight", manager, false);
      NDArray combinedBias = value(parameterStore, combinedProjection, "bias", manager, false);
      NDArray outputWeight = value(parameterStore, outputProjection, "weight", manager, false);
      createdProgram =
          new Program(
              manager.getEngine().newFusionCompiler(manager.getDevice()),
              resourceManager,
              combinedWeight,
              combinedBias,
              outputWeight,
              dataType,
              maximumBatch,
              executionSlots);
    } catch (RuntimeException | Error constructionFailure) {
      closeResource(constructionFailure, createdProgram);
      closeResource(constructionFailure, resourceManager);
      throw constructionFailure;
    }
    program = createdProgram;
  }

  @Override
  public Forward beginForward(NDManager workingManager) {
    if (closed) {
      throw new IllegalStateException("player tile context fusion is closed");
    }
    if (failure != null) {
      throw new IllegalStateException("player tile context fusion is poisoned", failure);
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
    incomplete = hasActiveForward() || !poisonedLeases.isEmpty();
    rethrow(releaseFailure);
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    if (hasIncompleteWork()) {
      throw new IllegalStateException(
          "cannot close player tile context fusion with incomplete device work", failure);
    }
    if (hasActiveForward()) {
      throw new IllegalStateException(
          "cannot close player tile context fusion with an active forward");
    }
    Throwable closeFailure = closeResource(null, program);
    if (closeFailure == null) {
      closeFailure = closeResource(null, resourceManager);
    }
    if (closeFailure == null) {
      closed = true;
    }
    rethrow(closeFailure);
  }

  private NDArray submit(FusionForward owner, NDArray byPlayer) {
    NDArray input = mixer.flatten(byPlayer);
    long rowCount = input.getShape().get(0);
    FusionOutputLease lease = null;
    NDArray active = null;
    boolean submitAttempted = false;
    try (FusionInvocation invocation = program.acquire()) {
      invocation.setInput(program.input(), input);
      invocation.setDimension(program.batch(), rowCount);
      submitAttempted = true;
      lease = invocation.submit();
      active = lease.get(program.output()).get(NDIndex.sliceAxis(0, 0, rowCount));
      input.attach(owner.workingManager());
      active.attach(owner.workingManager());
      owner.bind(active, lease);
      return active;
    } catch (RuntimeException | Error submitFailure) {
      closeResource(submitFailure, active);
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
      poison(submitFailure, submitAttempted);
      throw submitFailure;
    }
  }

  private int freeSlot() {
    for (int slot = 0; slot < activeForwards.length; slot++) {
      if (activeForwards[slot] == null) {
        return slot;
      }
    }
    throw new IllegalStateException("no player tile context fusion slot is available");
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
      throw new IllegalStateException("player tile context fusion slot ownership was lost");
    }
    activeForwards[slot] = null;
  }

  private void poison(Throwable cause, boolean submittedWork) {
    if (failure == null) {
      failure = cause;
    }
    incomplete |= submittedWork;
  }

  private static NDArray value(
      ParameterStore parameterStore,
      Linear projection,
      String name,
      NDManager manager,
      boolean training) {
    Parameter parameter = projection.getDirectParameters().get(name);
    return parameterStore.getValue(parameter, manager.getDevice(), training);
  }

  private static final class Program implements AutoCloseable {

    private final FusionRecipe.Dimension batch;
    private final FusionRecipe.Input input;
    private final FusionRecipe.Output output;
    private FusionPlan plan;
    private FusionExecutable executable;
    private FusionSession session;

    private Program(
        FusionCompiler compiler,
        NDManager resourceManager,
        NDArray combinedWeight,
        NDArray combinedBias,
        NDArray outputWeight,
        DataType dataType,
        int maximumBatch,
        int executionSlots) {
      if (combinedWeight.getDataType() != dataType
          || combinedBias.getDataType() != dataType
          || outputWeight.getDataType() != dataType) {
        throw new IllegalArgumentException(
            "player tile context parameters differ from compute precision");
      }
      long contextWidth = outputWeight.getShape().get(0);
      FusionRecipe.Builder builder = FusionRecipe.builder("epsilon-player-tile-context");
      batch = builder.addDimension("batch", maximumBatch);
      input =
          builder.addInput(
              "input",
              FusionRecipe.TensorSpec.of(
                  dataType, batch, Tile.NUM_TILE_TYPES, GameState.NUM_PLAYERS * contextWidth));
      FusionRecipe.Constant combinedWeightConstant =
          builder.addConstant(
              "combinedWeight",
              FusionRecipe.TensorSpec.fixed(dataType, contextWidth * 2L, contextWidth * 4L));
      FusionRecipe.Constant combinedBiasConstant =
          builder.addConstant(
              "combinedBias", FusionRecipe.TensorSpec.fixed(dataType, contextWidth * 2L));
      FusionRecipe.Constant outputWeightConstant =
          builder.addConstant(
              "outputWeight", FusionRecipe.TensorSpec.fixed(dataType, contextWidth, contextWidth));
      FusionRecipe.ProjectedResidualMlp mixed =
          builder
              .projectedResidualMlp("mixed", input)
              .setCombinedWeight(combinedWeightConstant)
              .setCombinedBias(combinedBiasConstant)
              .setOutputWeight(outputWeightConstant)
              .build();
      output = builder.addOutput("context", mixed);
      FusionRecipe recipe = builder.build();

      FusionPlan createdPlan = null;
      FusionExecutable createdExecutable = null;
      FusionSession createdSession = null;
      try {
        createdPlan = compiler.prepare(recipe, FusionCompileConfig.defaults());
        FusionCompilationReport report = createdPlan.getCompilationReport();
        if (!report.isNativeOnly()) {
          throw new IllegalStateException("player tile context plan is not native-only");
        }
        FusionConstantBindings bindings =
            FusionConstantBindings.builder(recipe)
                .bind(combinedWeightConstant, combinedWeight)
                .bind(combinedBiasConstant, combinedBias)
                .bind(outputWeightConstant, outputWeight)
                .build();
        createdExecutable = createdPlan.bind(bindings);
        createdSession =
            createdExecutable.newSession(
                resourceManager,
                FusionSessionConfig.builder().optOutputSlotCount(executionSlots).build());
        LOGGER.info(
            "Decision player tile context fusion prepared: dtype={}, contextWidth={}, "
                + "backend={}, commands={}, outputSlots={}, executableStorageBytes={}, "
                + "retainedSessionStorageBytes={}, requiredExecutionLaneStorageBytes={}",
            dataType,
            contextWidth,
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

    private FusionInvocation acquire() {
      return session.acquire();
    }

    private FusionRecipe.Dimension batch() {
      return batch;
    }

    private FusionRecipe.Input input() {
      return input;
    }

    private FusionRecipe.Output output() {
      return output;
    }

    @Override
    public void close() {
      Throwable closeFailure = closeResource(null, session);
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

    private final DecisionPolicyFusionPlayerTileContextExecution owner;
    private final int slot;
    private NDManager workingManager;
    private NDArray output;
    private FusionOutputLease lease;
    private boolean executed;
    private boolean sealed;
    private boolean finished;

    private FusionForward(
        DecisionPolicyFusionPlayerTileContextExecution owner, int slot, NDManager workingManager) {
      this.owner = owner;
      this.slot = slot;
      this.workingManager = workingManager;
    }

    @Override
    public NDArray mix(ParameterStore parameterStore, NDArray byPlayer) {
      if (finished || sealed || executed) {
        throw new IllegalStateException("player tile context fusion cannot execute twice");
      }
      NDArray result = owner.submit(this, byPlayer);
      executed = true;
      return result;
    }

    private NDManager workingManager() {
      return workingManager;
    }

    private FusionOutputLease lease() {
      return lease;
    }

    private void bind(NDArray output, FusionOutputLease lease) {
      this.output = output;
      this.lease = lease;
    }

    @Override
    public AutoCloseable seal() {
      if (finished || sealed || !executed || lease == null) {
        throw new IllegalStateException("player tile context fusion has no completed submission");
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
      try {
        if (output != null) {
          output.close();
          output = null;
        }
        if (lease != null) {
          lease.close();
          lease = null;
        }
      } catch (Throwable failure) {
        owner.poison(failure, true);
        rethrow(failure);
      }
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
