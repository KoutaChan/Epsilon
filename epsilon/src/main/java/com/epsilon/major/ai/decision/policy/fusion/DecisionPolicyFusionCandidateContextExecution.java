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
import com.epsilon.core.Action;
import com.epsilon.major.ai.decision.input.DecisionInputSchema;
import com.epsilon.major.ai.decision.policy.DecisionAlternative;
import com.epsilon.major.ai.decision.policy.DecisionPolicyCandidateContextExecution;
import com.epsilon.major.ai.decision.policy.DecisionPolicyCandidates;
import com.epsilon.major.ai.decision.policy.DecisionPolicyContextPool;
import java.util.ArrayList;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * djl-rocmの対応付けを使うグループ別マスク付きソフトマックス集約を使い、候補文脈の集約とネットワーク列への写像を一度に実行する。
 *
 * <p>選択肢 10列、PASS 1列、後段Fusionが直接読むRON/KYUSHU/TSUMOの各1列を一つのネイティブ
 * 投入で作る。各集合はセッションが持つ二つの実体となる確保の連続同じ領域への参照であり、有効なバッチのビューだけを呼び出し側へ貸し出す。
 * 入力テンソルの所有権は移さず、出力利用権だけを方策順伝播の寿命へ結び付ける。
 *
 * <p>候補幅はDecision 容量区分ごとに固定である。スコア、マスク、候補表現のデータ型は独立なので、最初の投入時に実テンソルの組を
 * 記述情報へ固定し、同じ容量区分では実行計画を再利用する。出力を持つセッションは実行枠ごとに一つだけ保持し、
 * 前の順伝播の利用権を返却した枠で候補幅が変わったときだけ差し替える。容量区分ごとの最大出力は重複して保持しない。
 */
public final class DecisionPolicyFusionCandidateContextExecution
    extends DecisionPolicyCandidateContextExecution {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(DecisionPolicyFusionCandidateContextExecution.class);
  private static final int GROUPS = Action.Type.values().length;
  private static final int DIRECT_DESTINATIONS = 3;
  private static final int OUTPUT_DESTINATIONS =
      DecisionAlternative.NETWORK_SIZE + 1 + DIRECT_DESTINATIONS;
  private static final int REFERENCE_BATCH = 384;

  private final NDManager resourceManager;
  private final FusionCompiler compiler;
  private final NDArray alternativeMapping;
  private final NDArray passMapping;
  private final NDArray ronMapping;
  private final NDArray kyushuMapping;
  private final NDArray tsumoMapping;
  private final int hiddenSize;
  private final int maximumBatch;
  private final Program[] programsByCandidateCapacity;
  private final FusionSession[] sessionsBySlot;
  private final Program[] sessionProgramsBySlot;
  private final FusionForward[] activeForwards;
  private final ArrayList<FusionOutputLease> poisonedLeases = new ArrayList<>();
  private Throwable failure;
  private boolean unknownSubmission;
  private boolean incomplete;
  private boolean closed;

  public DecisionPolicyFusionCandidateContextExecution(
      NDManager manager, int hiddenSize, DataType dataType, int maximumBatch, int executionSlots) {
    if (executionSlots <= 0) {
      throw new IllegalArgumentException("executionSlots must be positive");
    }
    requireFloatingType(dataType);
    this.hiddenSize = hiddenSize;
    this.maximumBatch = maximumBatch;
    activeForwards = new FusionForward[executionSlots];
    sessionsBySlot = new FusionSession[executionSlots];
    sessionProgramsBySlot = new Program[executionSlots];
    programsByCandidateCapacity = new Program[DecisionInputSchema.MAX_LEGAL_ACTIONS + 1];
    resourceManager = manager.newSubManager();

    try {
      alternativeMapping =
          resourceManager.create(DecisionPolicyContextPool.alternativeTypeMapping());
      passMapping = resourceManager.create(DecisionPolicyContextPool.passTypeMapping());
      ronMapping =
          resourceManager.create(
              DecisionPolicyContextPool.directTypeMapping(DecisionAlternative.RON));
      kyushuMapping =
          resourceManager.create(
              DecisionPolicyContextPool.directTypeMapping(DecisionAlternative.KYUSHU));
      tsumoMapping =
          resourceManager.create(
              DecisionPolicyContextPool.directTypeMapping(DecisionAlternative.TSUMO));
      compiler = manager.getEngine().newFusionCompiler(manager.getDevice());
      int referenceRows = Math.min(maximumBatch, REFERENCE_BATCH);
      long expectedOutputBytesPerSlot = outputBytes(maximumBatch, hiddenSize, dataType);
      long maximumRetainedOutputBytes =
          Math.multiplyExact(
              outputBytes(maximumBatch, hiddenSize, DataType.FLOAT32), executionSlots);
      LOGGER.info(
          "Decision candidate context fusion: maximumBatch={}, referenceRows={}, buckets={},"
              + " groups={}, destinations={}, hiddenSize={}, slots={}, expectedDtype={},"
              + " programBinding=FIRST_ACTUAL_DTYPES,"
              + " sessionOwnership=EXECUTION_SLOT, modelDtypeEstimateBytesPerSlot={},"
              + " maximumRetainedOutputBytes={}, activeLogicalBytesAtReference={}",
          maximumBatch,
          referenceRows,
          DecisionInputSchema.LEGAL_ACTION_BUCKETS.length,
          GROUPS,
          OUTPUT_DESTINATIONS,
          hiddenSize,
          executionSlots,
          dataType,
          expectedOutputBytesPerSlot,
          maximumRetainedOutputBytes,
          outputBytes(referenceRows, hiddenSize, dataType));
    } catch (RuntimeException | Error constructionFailure) {
      closePrograms(constructionFailure);
      closeResource(constructionFailure, resourceManager);
      throw constructionFailure;
    }
  }

  public static long outputBytes(long rows, int hiddenSize, DataType presenceType) {
    long contextElements =
        Math.multiplyExact(Math.multiplyExact(rows, OUTPUT_DESTINATIONS), hiddenSize);
    long presenceElements = Math.multiplyExact(rows, OUTPUT_DESTINATIONS);
    return Math.addExact(
        Math.multiplyExact(contextElements, DataType.FLOAT32.getNumOfBytes()),
        Math.multiplyExact(presenceElements, presenceType.getNumOfBytes()));
  }

  @Override
  public Forward beginForward(NDManager workingManager) {
    if (closed) {
      throw new IllegalStateException("candidate context fusion is closed");
    }
    if (failure != null) {
      throw new IllegalStateException("candidate context fusion is poisoned", failure);
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
          "cannot close candidate context fusion with incomplete device work", failure);
    }
    if (hasActiveForward()) {
      throw new IllegalStateException(
          "cannot close candidate context fusion with an active forward");
    }
    Throwable closeFailure = closeSessions(null);
    if (closeFailure == null) {
      closeFailure = closePrograms(null);
    }
    if (closeFailure == null) {
      closeFailure = closeResource(null, resourceManager);
    }
    if (closeFailure == null) {
      closed = true;
    }
    rethrow(closeFailure);
  }

  private LeasedOutput submit(
      FusionForward owner,
      NDArray candidateEmbeddings,
      NDArray candidateScores,
      NDArray typeMasks) {
    long rowCount = candidateScores.getShape().get(0);
    int candidateCapacity = Math.toIntExact(candidateScores.getShape().get(1));
    Program program =
        program(
            candidateCapacity,
            candidateScores.getDataType(),
            typeMasks.getDataType(),
            candidateEmbeddings.getDataType());
    FusionOutputLease lease = null;
    NDArray[] active = new NDArray[8];
    boolean submitAttempted = false;
    try (FusionInvocation invocation = acquire(program, owner.slot)) {
      invocation.setInput(program.scoresInput(), candidateScores);
      invocation.setInput(program.masksInput(), typeMasks);
      invocation.setInput(program.valuesInput(), candidateEmbeddings);
      invocation.setDimension(program.batch(), rowCount);
      submitAttempted = true;
      lease = invocation.submit();
      active[0] =
          activeRows(
              lease.get(program.alternativeContextsOutput()), rowCount, owner.workingManager());
      active[1] =
          activeRows(
              lease.get(program.alternativePresenceOutput()), rowCount, owner.workingManager());
      active[2] =
          activeRows(lease.get(program.passContextsOutput()), rowCount, owner.workingManager());
      active[3] =
          activeRows(lease.get(program.passPresenceOutput()), rowCount, owner.workingManager());
      active[4] =
          activeRows(lease.get(program.ronContextsOutput()), rowCount, owner.workingManager());
      active[5] =
          activeRows(lease.get(program.kyushuContextsOutput()), rowCount, owner.workingManager());
      active[6] =
          activeRows(lease.get(program.kyushuPresenceOutput()), rowCount, owner.workingManager());
      active[7] =
          activeRows(lease.get(program.tsumoContextsOutput()), rowCount, owner.workingManager());
      DecisionPolicyContextPool.MappedCandidateContexts outputs =
          new DecisionPolicyContextPool.MappedCandidateContexts(
              active[0], active[1], active[2], active[3], active[4], active[5], active[6],
              active[7]);
      return new LeasedOutput(outputs, lease);
    } catch (RuntimeException | Error submitFailure) {
      closeActive(submitFailure, active);
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
      poison(submitFailure, submitAttempted);
      throw submitFailure;
    }
  }

  private Program program(
      int candidateCapacity, DataType scoreType, DataType maskType, DataType valueType) {
    if (candidateCapacity < 0 || candidateCapacity >= programsByCandidateCapacity.length) {
      throw new IllegalArgumentException("unsupported candidate capacity: " + candidateCapacity);
    }
    requireFloatingType(scoreType);
    requireFloatingType(maskType);
    requireFloatingType(valueType);
    Program program = programsByCandidateCapacity[candidateCapacity];
    if (program == null) {
      boolean supported = false;
      for (int configuredCapacity : DecisionInputSchema.LEGAL_ACTION_BUCKETS) {
        if (configuredCapacity == candidateCapacity) {
          supported = true;
          break;
        }
      }
      if (!supported) {
        throw new IllegalArgumentException("unsupported candidate capacity: " + candidateCapacity);
      }
      program =
          new Program(
              compiler,
              alternativeMapping,
              passMapping,
              ronMapping,
              kyushuMapping,
              tsumoMapping,
              candidateCapacity,
              hiddenSize,
              scoreType,
              maskType,
              valueType,
              maximumBatch);
      programsByCandidateCapacity[candidateCapacity] = program;
    } else if (!program.matches(scoreType, maskType, valueType)) {
      throw new IllegalStateException(
          "candidate context input dtypes changed after program binding: capacity="
              + candidateCapacity
              + ", scores="
              + scoreType
              + ", masks="
              + maskType
              + ", values="
              + valueType);
    }
    return program;
  }

  private FusionInvocation acquire(Program program, int slot) {
    if (sessionProgramsBySlot[slot] != program) {
      if (sessionsBySlot[slot] != null) {
        // この枠の前の利用権は、全消費処理の完了後にForward.close()で返却済み。
        sessionsBySlot[slot].close();
        sessionsBySlot[slot] = null;
        sessionProgramsBySlot[slot] = null;
      }
      sessionsBySlot[slot] =
          program.executable.newSession(resourceManager, FusionSessionConfig.defaults());
      sessionProgramsBySlot[slot] = program;
    }
    return sessionsBySlot[slot].acquire();
  }

  private static NDArray activeRows(NDArray storage, long rows, NDManager workingManager) {
    NDArray active = storage.get(NDIndex.sliceAxis(0, 0, rows));
    active.attach(workingManager);
    return active;
  }

  private int freeSlot() {
    for (int slot = 0; slot < activeForwards.length; slot++) {
      if (activeForwards[slot] == null) {
        return slot;
      }
    }
    throw new IllegalStateException("no candidate context fusion slot is available");
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
      throw new IllegalStateException("candidate context fusion slot ownership was lost");
    }
    activeForwards[slot] = null;
  }

  private void poison(Throwable cause, boolean submittedWork) {
    if (failure == null) {
      failure = cause;
    }
    incomplete |= submittedWork;
  }

  private Throwable closePrograms(Throwable closeFailure) {
    for (int index = programsByCandidateCapacity.length - 1; index >= 0; index--) {
      Program program = programsByCandidateCapacity[index];
      if (program == null) {
        continue;
      }
      try {
        program.close();
        programsByCandidateCapacity[index] = null;
      } catch (Throwable failure) {
        closeFailure = addFailure(closeFailure, failure);
      }
    }
    return closeFailure;
  }

  private Throwable closeSessions(Throwable closeFailure) {
    for (int slot = sessionsBySlot.length - 1; slot >= 0; slot--) {
      if (sessionsBySlot[slot] == null) {
        continue;
      }
      try {
        sessionsBySlot[slot].close();
        sessionsBySlot[slot] = null;
        sessionProgramsBySlot[slot] = null;
      } catch (Throwable failure) {
        closeFailure = addFailure(closeFailure, failure);
      }
    }
    return closeFailure;
  }

  private static void requireFloatingType(DataType dataType) {
    if (dataType != DataType.FLOAT16
        && dataType != DataType.BFLOAT16
        && dataType != DataType.FLOAT32) {
      throw new IllegalArgumentException(
          "candidate context inputs require a floating-point dtype: " + dataType);
    }
  }

  private static final class Program implements AutoCloseable {

    private final DataType scoreType;
    private final DataType maskType;
    private final DataType valueType;
    private final FusionRecipe.Dimension batch;
    private final FusionRecipe.Input scoresInput;
    private final FusionRecipe.Input masksInput;
    private final FusionRecipe.Input valuesInput;
    private final FusionRecipe.Output alternativeContextsOutput;
    private final FusionRecipe.Output alternativePresenceOutput;
    private final FusionRecipe.Output passContextsOutput;
    private final FusionRecipe.Output passPresenceOutput;
    private final FusionRecipe.Output ronContextsOutput;
    private final FusionRecipe.Output kyushuContextsOutput;
    private final FusionRecipe.Output kyushuPresenceOutput;
    private final FusionRecipe.Output tsumoContextsOutput;
    private FusionPlan plan;
    private FusionExecutable executable;

    private Program(
        FusionCompiler compiler,
        NDArray alternativeMappingValue,
        NDArray passMappingValue,
        NDArray ronMappingValue,
        NDArray kyushuMappingValue,
        NDArray tsumoMappingValue,
        int candidateCapacity,
        int hiddenSize,
        DataType scoreType,
        DataType maskType,
        DataType valueType,
        int maximumBatch) {
      this.scoreType = scoreType;
      this.maskType = maskType;
      this.valueType = valueType;
      FusionRecipe.Builder builder =
          FusionRecipe.builder("epsilon-policy-candidate-context-" + candidateCapacity);
      batch = builder.addDimension("batch", maximumBatch);
      scoresInput =
          builder.addInput(
              "scores", FusionRecipe.TensorSpec.of(scoreType, batch, candidateCapacity));
      masksInput =
          builder.addInput(
              "typeMasks", FusionRecipe.TensorSpec.of(maskType, batch, candidateCapacity, GROUPS));
      valuesInput =
          builder.addInput(
              "candidateEmbeddings",
              FusionRecipe.TensorSpec.of(valueType, batch, candidateCapacity, hiddenSize));
      FusionRecipe.Constant alternativeMapping =
          builder.addConstant(
              "alternativeTypeMapping",
              FusionRecipe.TensorSpec.fixed(DataType.INT32, DecisionAlternative.NETWORK_SIZE));
      FusionRecipe.Constant passMapping =
          builder.addConstant("passTypeMapping", FusionRecipe.TensorSpec.fixed(DataType.INT32, 1));
      FusionRecipe.Constant ronMapping =
          builder.addConstant("ronTypeMapping", FusionRecipe.TensorSpec.fixed(DataType.INT32, 1));
      FusionRecipe.Constant kyushuMapping =
          builder.addConstant(
              "kyushuTypeMapping", FusionRecipe.TensorSpec.fixed(DataType.INT32, 1));
      FusionRecipe.Constant tsumoMapping =
          builder.addConstant("tsumoTypeMapping", FusionRecipe.TensorSpec.fixed(DataType.INT32, 1));
      FusionRecipe.MappedGroupedMaskedSoftmaxPoolGroup mapped =
          builder
              .mappedGroupedMaskedSoftmaxPoolGroup(
                  "candidateTypes", scoresInput, masksInput, valuesInput)
              .addOutputSet("alternatives", alternativeMapping)
              .addOutputSet("pass", passMapping)
              .addOutputSet("ron", ronMapping)
              .addOutputSet("kyushu", kyushuMapping)
              .addOutputSet("tsumo", tsumoMapping)
              .build();
      alternativeContextsOutput =
          builder.addOutput("alternativeContexts", mapped.getOutputSet(0).getContexts());
      alternativePresenceOutput =
          builder.addOutput("alternativePresence", mapped.getOutputSet(0).getPresence());
      passContextsOutput = builder.addOutput("passContexts", mapped.getOutputSet(1).getContexts());
      passPresenceOutput = builder.addOutput("passPresence", mapped.getOutputSet(1).getPresence());
      ronContextsOutput = builder.addOutput("ronContexts", mapped.getOutputSet(2).getContexts());
      kyushuContextsOutput =
          builder.addOutput("kyushuContexts", mapped.getOutputSet(3).getContexts());
      kyushuPresenceOutput =
          builder.addOutput("kyushuPresence", mapped.getOutputSet(3).getPresence());
      tsumoContextsOutput =
          builder.addOutput("tsumoContexts", mapped.getOutputSet(4).getContexts());
      FusionRecipe recipe = builder.build();
      FusionPlan createdPlan = null;
      FusionExecutable createdExecutable = null;
      FusionCompilationReport createdReport;
      try {
        createdPlan = compiler.prepare(recipe, FusionCompileConfig.defaults());
        createdReport = createdPlan.getCompilationReport();
        if (!createdReport.isNativeOnly()) {
          throw new IllegalStateException("policy candidate context plan is not native-only");
        }
        createdExecutable =
            createdPlan.bind(
                FusionConstantBindings.builder(recipe)
                    .bind(alternativeMapping, alternativeMappingValue)
                    .bind(passMapping, passMappingValue)
                    .bind(ronMapping, ronMappingValue)
                    .bind(kyushuMapping, kyushuMappingValue)
                    .bind(tsumoMapping, tsumoMappingValue)
                    .build());
      } catch (RuntimeException | Error constructionFailure) {
        closeResource(constructionFailure, createdExecutable);
        closeResource(constructionFailure, createdPlan);
        throw constructionFailure;
      }
      plan = createdPlan;
      executable = createdExecutable;
      FusionCompilationReport report = createdReport;
      LOGGER.info(
          "Decision candidate context fusion bucket prepared: candidateCapacity={}, "
              + "scoresDtype={}, masksDtype={}, valuesDtype={}, backend={}, commands={}, "
              + "executableStorageBytes={}, executionStorageBytes={}, workspaceBytes={}, "
              + "retainedSessionStorageBytesPerSlot={}, requiredExecutionLaneStorageBytes={}",
          candidateCapacity,
          scoreType,
          maskType,
          valueType,
          report.getBackend(),
          report.getCommandCount(),
          report.getExecutableStorageBytes(),
          report.getExecutionStorageBytes(),
          report.getWorkspaceBytes(),
          report.getRetainedSessionStorageBytes(1),
          report.getRequiredExecutionLaneStorageBytes());
    }

    private boolean matches(
        DataType candidateScoreType, DataType candidateMaskType, DataType candidateValueType) {
      return scoreType == candidateScoreType
          && maskType == candidateMaskType
          && valueType == candidateValueType;
    }

    private FusionRecipe.Dimension batch() {
      return batch;
    }

    private FusionRecipe.Input scoresInput() {
      return scoresInput;
    }

    private FusionRecipe.Input masksInput() {
      return masksInput;
    }

    private FusionRecipe.Input valuesInput() {
      return valuesInput;
    }

    private FusionRecipe.Output alternativeContextsOutput() {
      return alternativeContextsOutput;
    }

    private FusionRecipe.Output alternativePresenceOutput() {
      return alternativePresenceOutput;
    }

    private FusionRecipe.Output passContextsOutput() {
      return passContextsOutput;
    }

    private FusionRecipe.Output passPresenceOutput() {
      return passPresenceOutput;
    }

    private FusionRecipe.Output ronContextsOutput() {
      return ronContextsOutput;
    }

    private FusionRecipe.Output kyushuContextsOutput() {
      return kyushuContextsOutput;
    }

    private FusionRecipe.Output kyushuPresenceOutput() {
      return kyushuPresenceOutput;
    }

    private FusionRecipe.Output tsumoContextsOutput() {
      return tsumoContextsOutput;
    }

    @Override
    public void close() {
      Throwable failure = null;
      if (executable != null) {
        try {
          executable.close();
          executable = null;
        } catch (Throwable closeFailure) {
          failure = addFailure(failure, closeFailure);
        }
      }
      if (executable == null && plan != null) {
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

    private final DecisionPolicyFusionCandidateContextExecution owner;
    private final int slot;
    private final NDArray[] inputs = new NDArray[3];
    private NDManager workingManager;
    private LeasedOutput output;
    private boolean executed;
    private boolean sealed;
    private boolean finished;

    private FusionForward(
        DecisionPolicyFusionCandidateContextExecution owner, int slot, NDManager workingManager) {
      this.owner = owner;
      this.slot = slot;
      this.workingManager = workingManager;
    }

    @Override
    public DecisionPolicyContextPool.MappedCandidateContexts pool(
        NDArray candidateEmbeddings,
        NDArray candidateScores,
        DecisionPolicyCandidates.CandidateMasks candidateMasks) {
      if (finished || sealed || executed) {
        throw new IllegalStateException("candidate contexts cannot execute twice");
      }
      inputs[0] = candidateEmbeddings;
      inputs[1] = candidateScores;
      inputs[2] = candidateMasks.typeMasks();
      output = owner.submit(this, inputs[0], inputs[1], inputs[2]);
      executed = true;
      return output.outputs();
    }

    private NDManager workingManager() {
      return workingManager;
    }

    @Override
    public AutoCloseable seal() {
      if (finished || sealed || !executed) {
        throw new IllegalStateException("candidate context forward did not execute");
      }
      sealed = true;
      return this;
    }

    @Override
    public void poison(Throwable failure) {
      if (finished) {
        return;
      }
      owner.poison(failure, output != null || owner.hasIncompleteWork());
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
      if (output != null) {
        try {
          output.close();
          output = null;
        } catch (Throwable failure) {
          owner.poison(failure, true);
          closeFailure = failure;
        }
      }
      if (output == null) {
        clear();
        finished = true;
        owner.finish(slot, this);
      }
      rethrow(closeFailure);
    }

    private void clear() {
      Arrays.fill(inputs, null);
      workingManager = null;
    }
  }

  private static final class LeasedOutput implements AutoCloseable {

    private final DecisionPolicyContextPool.MappedCandidateContexts outputs;
    private FusionOutputLease lease;
    private boolean arraysReleased;

    private LeasedOutput(
        DecisionPolicyContextPool.MappedCandidateContexts outputs, FusionOutputLease lease) {
      this.outputs = outputs;
      this.lease = lease;
    }

    private DecisionPolicyContextPool.MappedCandidateContexts outputs() {
      return outputs;
    }

    @Override
    public void close() {
      if (!arraysReleased) {
        outputs.tsumoContexts().close();
        outputs.kyushuPresence().close();
        outputs.kyushuContexts().close();
        outputs.ronContexts().close();
        outputs.passPresence().close();
        outputs.passContexts().close();
        outputs.alternativePresence().close();
        outputs.alternativeContexts().close();
        arraysReleased = true;
      }
      if (lease != null) {
        lease.close();
        lease = null;
      }
    }
  }

  private static void closeActive(Throwable failure, NDArray[] active) {
    for (int index = active.length - 1; index >= 0; index--) {
      NDArray array = active[index];
      if (array == null) {
        continue;
      }
      try {
        array.close();
      } catch (Throwable closeFailure) {
        failure.addSuppressed(closeFailure);
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
