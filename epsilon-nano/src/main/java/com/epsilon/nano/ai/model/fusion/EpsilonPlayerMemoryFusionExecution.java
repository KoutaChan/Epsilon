package com.epsilon.nano.ai.model.fusion;

import ai.djl.Device;
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
import com.epsilon.core.GameState;
import com.epsilon.nano.ai.decision.input.DecisionInputSchema;
import com.epsilon.nano.ai.model.EpsilonMahjongStateEncoder;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 疎なプレイヤーごとの履歴表現エンコーダーをdjl-rocmのネイティブ Fusion 実行計画として実行する。 */
public final class EpsilonPlayerMemoryFusionExecution implements AutoCloseable {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(EpsilonPlayerMemoryFusionExecution.class);

  private final FusionRecipe.Dimension batch;
  private final FusionRecipe.Dimension active;
  private final FusionRecipe.Input playerInput;
  private final FusionRecipe.Input riverInput;
  private final FusionRecipe.Input meldInput;
  private final FusionRecipe.Input indices;
  private final FusionRecipe.Output output;
  private final Forward[] activeForwards;
  private FusionPlan plan;
  private FusionExecutable executable;
  private FusionSession session;
  private Throwable failure;
  private boolean unknownSubmission;
  private boolean incomplete;
  private boolean closed;

  EpsilonPlayerMemoryFusionExecution(
      FusionRecipe.Dimension batch,
      FusionRecipe.Dimension active,
      FusionRecipe.Input playerInput,
      FusionRecipe.Input riverInput,
      FusionRecipe.Input meldInput,
      FusionRecipe.Input indices,
      FusionRecipe.Output output,
      FusionPlan plan,
      FusionExecutable executable,
      FusionSession session,
      int executionSlots) {
    this.batch = batch;
    this.active = active;
    this.playerInput = playerInput;
    this.riverInput = riverInput;
    this.meldInput = meldInput;
    this.indices = indices;
    this.output = output;
    this.plan = plan;
    this.executable = executable;
    this.session = session;
    activeForwards = new Forward[executionSlots];
  }

  /**
   * 凍結済みパラメーターを同じデバイスへ関連付け、疎なプレイヤーごとの履歴表現エンコーダーの永続実行計画を作る。
   *
   * <p>存在するインデックスは既存の連結したデバイス連続バッファ上のビューをそのまま関連付ける。バッチと存在する数を別の可変長さとして扱うため、 インデックスの複製や再転送は発生しない。
   *
   * @param manager パイプラインと同じGPUを持つ長寿命管理元
   * @param parameters 凍結済みエンコーダーパラメーターの名前付きビュー
   * @param expectedDataType 凍結パラメーターに要求するデータ型
   * @param inputDataType デバイス入力のデータ型
   * @param maximumBatch 実行計画が受理する最大バッチ
   * @param executionSlots 同時に保持できる未回収順伝播数
   * @return パイプライン専用のプレイヤーごとの履歴表現 Fusion実行境界
   */
  public static EpsilonPlayerMemoryFusionExecution create(
      NDManager manager,
      Parameters parameters,
      DataType expectedDataType,
      DataType inputDataType,
      int maximumBatch,
      int executionSlots) {
    Objects.requireNonNull(manager, "manager");
    Objects.requireNonNull(parameters, "parameters");
    Objects.requireNonNull(expectedDataType, "expectedDataType");
    Objects.requireNonNull(inputDataType, "inputDataType");
    if (!manager.getDevice().isGpu()) {
      throw new UnsupportedOperationException("playerMemory=FUSION requires a GPU device");
    }
    if (inputDataType != expectedDataType) {
      throw new IllegalArgumentException(
          "playerMemory=FUSION requires matching parameter and input data types: "
              + expectedDataType
              + "/"
              + inputDataType);
    }
    if (maximumBatch <= 0) {
      throw new IllegalArgumentException("maximumBatch must be positive");
    }
    if (executionSlots <= 0) {
      throw new IllegalArgumentException("executionSlots must be positive");
    }
    parameters.requireDataType(expectedDataType);

    long maximumPresentTokens =
        Math.multiplyExact(
            maximumBatch,
            Math.multiplyExact(
                GameState.NUM_PLAYERS, EpsilonMahjongStateEncoder.PLAYER_MEMORY_TOKEN_COUNT));
    List<NDArray> constants = new ArrayList<>();
    FusionRecipe.Builder builder = FusionRecipe.builder("epsilon-player-memory-transformer");
    FusionRecipe.Dimension batch = builder.addDimension("batch", maximumBatch);
    FusionRecipe.Dimension active = builder.addDimension("activeTokens", maximumPresentTokens);
    FusionRecipe.Input playerInput =
        builder.addInput(
            "playerTokens",
            FusionRecipe.TensorSpec.of(
                expectedDataType, batch, GameState.NUM_PLAYERS, 1, parameters.hiddenSize()));
    FusionRecipe.Input riverInput =
        builder.addInput(
            "riverTokens",
            FusionRecipe.TensorSpec.of(
                expectedDataType,
                batch,
                GameState.NUM_PLAYERS,
                DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER,
                parameters.hiddenSize()));
    FusionRecipe.Input meldInput =
        builder.addInput(
            "meldTokens",
            FusionRecipe.TensorSpec.of(
                expectedDataType,
                batch,
                GameState.NUM_PLAYERS,
                DecisionInputSchema.MAX_MELDS_PER_PLAYER,
                parameters.hiddenSize()));
    FusionRecipe.Input indices =
        builder.addInput("presentIndices", FusionRecipe.TensorSpec.of(DataType.INT32, active));
    FusionRecipe.Constant inputNormWeight =
        addConstant(builder, constants, "inputNormWeight", parameters.inputNormWeight());
    FusionRecipe.Constant inputNormBias =
        addConstant(builder, constants, "inputNormBias", parameters.inputNormBias());
    FusionRecipe.Constant attentionInputWeight =
        addConstant(builder, constants, "attentionInputWeight", parameters.attentionInputWeight());
    FusionRecipe.Constant attentionInputBias =
        addConstant(builder, constants, "attentionInputBias", parameters.attentionInputBias());
    FusionRecipe.Constant queryKeyValueWeight =
        addConstant(builder, constants, "queryKeyValueWeight", parameters.queryKeyValueWeight());
    FusionRecipe.Constant attentionOutputWeight =
        addConstant(
            builder, constants, "attentionOutputWeight", parameters.attentionOutputWeight());
    FusionRecipe.Constant attentionOutputBias =
        addConstant(builder, constants, "attentionOutputBias", parameters.attentionOutputBias());
    FusionRecipe.Constant feedForwardInputWeight =
        addConstant(
            builder, constants, "feedForwardInputWeight", parameters.feedForwardInputWeight());
    FusionRecipe.Constant feedForwardInputBias =
        addConstant(builder, constants, "feedForwardInputBias", parameters.feedForwardInputBias());
    FusionRecipe.Constant expansionWeight =
        addConstant(builder, constants, "expansionWeight", parameters.expansionWeight());
    FusionRecipe.Constant expansionBias =
        addConstant(builder, constants, "expansionBias", parameters.expansionBias());
    FusionRecipe.Constant projectionWeight =
        addConstant(builder, constants, "projectionWeight", parameters.projectionWeight());
    FusionRecipe.Constant projectionBias =
        addConstant(builder, constants, "projectionBias", parameters.projectionBias());
    FusionRecipe.Constant outputWeight =
        addConstant(builder, constants, "outputWeight", parameters.outputWeight());
    FusionRecipe.Constant outputBias =
        addConstant(builder, constants, "outputBias", parameters.outputBias());
    FusionRecipe.IndexedLocalTransformerEncoder encoded =
        builder
            .indexedLocalTransformerEncoder(
                "encoded",
                List.of(playerInput, riverInput, meldInput),
                indices,
                parameters.attentionHeads(),
                parameters.attentionWidth(),
                parameters.feedForwardWidth())
            .setInputNormalization(inputNormWeight, inputNormBias)
            .setBlock(
                attentionInputWeight,
                attentionInputBias,
                queryKeyValueWeight,
                attentionOutputWeight,
                attentionOutputBias,
                feedForwardInputWeight,
                feedForwardInputBias,
                expansionWeight,
                expansionBias,
                projectionWeight,
                projectionBias,
                outputWeight,
                outputBias)
            .build();
    FusionRecipe.Output output = builder.addOutput("encoded", encoded);
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
        throw new IllegalStateException("player-memory transformer plan is not native-only");
      }
      executable = plan.bind(bindings.build());
      session =
          executable.newSession(
              manager, FusionSessionConfig.builder().optOutputSlotCount(executionSlots).build());
      long avoidedTokenConcatBytes =
          Math.multiplyExact(
              maximumPresentTokens,
              Math.multiplyExact((long) parameters.hiddenSize(), expectedDataType.getNumOfBytes()));
      LOGGER.info(
          "Decision player-memory transformer fusion: maximumBatch={}, "
              + "maximumPresentTokens={}, slots={}, inputSegments=3, "
              + "materializedTokenConcatBytes=0, avoidedTokenConcatBytesAtMaximumBatch={}, "
              + "nativeCommands=1, "
              + "gemmStages/forward=4, customStages/forward=6, "
              + "executableStorageBytes={}, retainedSessionStorageBytes={}, "
              + "requiredExecutionLaneStorageBytes={}",
          maximumBatch,
          maximumPresentTokens,
          executionSlots,
          avoidedTokenConcatBytes,
          report.getExecutableStorageBytes(),
          report.getRetainedSessionStorageBytes(executionSlots),
          report.getRequiredExecutionLaneStorageBytes());
      return new EpsilonPlayerMemoryFusionExecution(
          batch,
          active,
          playerInput,
          riverInput,
          meldInput,
          indices,
          output,
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

  /** GPUかつ入力と凍結パラメーターが同じデータ型ならネイティブ実行計画を構築できる。 */
  static boolean supports(Device device, DataType parameterDataType, DataType inputDataType) {
    return device.isGpu() && inputDataType == parameterDataType;
  }

  /** 空き枠を予約し、一つのプレイヤーごとの履歴表現順伝播を開始する。 */
  public Forward beginForward(NDManager workingManager) {
    if (closed) {
      throw new IllegalStateException("player-memory inference execution is closed");
    }
    if (failure != null) {
      throw new IllegalStateException("player-memory inference execution is poisoned", failure);
    }
    for (int slot = 0; slot < activeForwards.length; slot++) {
      if (activeForwards[slot] == null) {
        Forward forward = new Forward(this, slot, workingManager);
        activeForwards[slot] = forward;
        return forward;
      }
    }
    throw new IllegalStateException("no player-memory inference slot is available");
  }

  /** 完了未確認のデバイス処理を保持しているなら{@code true}を返す。 */
  public boolean hasIncompleteWork() {
    return incomplete || unknownSubmission;
  }

  /** この実行コンテキストを利用不能にした最初の失敗を返す。 */
  public Throwable failure() {
    return failure;
  }

  /** 同じストリーム末尾の完了確認後、失敗順伝播が保持した利用権を解放する。 */
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
      throw new IllegalStateException("player-memory inference has incomplete device work");
    }
    for (Forward forward : activeForwards) {
      if (forward != null) {
        throw new IllegalStateException("player-memory inference has an active forward");
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
      throw new IllegalStateException("player-memory forward does not own its slot");
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

  /** 一つのネットワーク順伝播が保持するプレイヤーごとの履歴表現出力利用権。 */
  public static final class Forward {

    private final EpsilonPlayerMemoryFusionExecution owner;
    private final int slot;
    private final NDManager workingManager;
    private FusionOutputLease lease;
    private NDArray activeOutput;
    private boolean encoded;
    private boolean sealed;
    private boolean poisoned;
    private boolean finished;

    private Forward(EpsilonPlayerMemoryFusionExecution owner, int slot, NDManager workingManager) {
      this.owner = owner;
      this.slot = slot;
      this.workingManager = workingManager;
    }

    /** 論理的に連結したプレイヤーごとの履歴表現区間を、配列を実際に連結せずにエンコーダーへ渡す。 */
    public NDArray encode(
        NDArray playerTokens, NDArray riverTokens, NDArray meldTokens, NDArray presentIndices) {
      if (encoded || sealed || poisoned) {
        throw new IllegalStateException("player-memory forward can encode only once");
      }
      long rowCount = playerTokens.getShape().get(0);
      long presentCount = presentIndices.getShape().get(0);
      requireInputDataType(owner.playerInput, playerTokens);
      requireInputDataType(owner.riverInput, riverTokens);
      requireInputDataType(owner.meldInput, meldTokens);
      requireInputDataType(owner.indices, presentIndices);
      retainInput(playerTokens);
      retainInput(riverTokens);
      retainInput(meldTokens);
      boolean submitAttempted = false;
      try (FusionInvocation invocation = owner.session.acquire()) {
        invocation.setInput(owner.playerInput, playerTokens);
        invocation.setInput(owner.riverInput, riverTokens);
        invocation.setInput(owner.meldInput, meldTokens);
        // インデックスは長寿命デバイス側バッチから借り、反復回数管理元へ所有権を移さない。
        invocation.setInput(owner.indices, presentIndices);
        invocation.setDimension(owner.batch, rowCount);
        invocation.setDimension(owner.active, presentCount);
        submitAttempted = true;
        lease = invocation.submit();
        activeOutput = lease.get(owner.output).get(NDIndex.sliceAxis(0, 0, rowCount));
        activeOutput.attach(workingManager);
        encoded = true;
        return activeOutput;
      } catch (RuntimeException | Error submitFailure) {
        if (submitAttempted && lease == null) {
          poisonUnknownSubmission(submitFailure);
        } else {
          poison(submitFailure);
        }
        throw submitFailure;
      }
    }

    private void retainInput(NDArray input) {
      if (input.getManager() != workingManager) {
        input.attach(workingManager);
      }
    }

    /** 出力枠を最終D2H完了まで保持する利用権へ変換する。 */
    public AutoCloseable seal() {
      if (!encoded || sealed || poisoned) {
        throw new IllegalStateException("player-memory forward cannot be sealed");
      }
      sealed = true;
      return this::releaseAfterCompletion;
    }

    /** 最終出力まで到達しなかった失敗を記録し、実行単位を再利用不可に設定する。 */
    public void poison(Throwable failure) {
      if (poisoned) {
        return;
      }
      poisoned = true;
      if (owner.failure == null) {
        owner.failure = failure;
      }
      boolean submittedWork = lease != null || owner.unknownSubmission;
      owner.incomplete |= submittedWork;
      if (!submittedWork) {
        try {
          releaseAfterCompletion();
        } catch (Throwable releaseFailure) {
          failure.addSuppressed(releaseFailure);
        }
      }
    }

    /** 完了トークンを返さず失敗した投入を、ストリーム完了確認まで再利用禁止にする。 */
    private void poisonUnknownSubmission(Throwable failure) {
      owner.unknownSubmission = true;
      poison(failure);
    }

    private void releaseAfterCompletion() {
      if (finished) {
        return;
      }
      Throwable releaseFailure = null;
      if (activeOutput != null) {
        try {
          activeOutput.close();
          activeOutput = null;
        } catch (Throwable closeFailure) {
          releaseFailure = closeFailure;
        }
      }
      if (activeOutput == null && lease != null) {
        releaseFailure = closeResource(releaseFailure, lease);
        if (releaseFailure == null) {
          lease = null;
        }
      }
      if (releaseFailure == null && activeOutput == null) {
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

    private static void requireInputDataType(FusionRecipe.Input input, NDArray value) {
      DataType expected = input.getSpec().getDataType();
      DataType actual = value.getDataType();
      if (actual != expected) {
        throw new IllegalArgumentException(
            "player-memory input "
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

  /**
   * 凍結済みプレイヤーごとの履歴表現パラメーターの名前付きビュー。
   *
   * <p>NDArrayの所有権は移さず、実行処理の寿命中に値を更新しない呼び出し契約とする。
   */
  public record Parameters(
      int hiddenSize,
      int attentionHeads,
      int attentionWidth,
      int feedForwardWidth,
      NDArray inputNormWeight,
      NDArray inputNormBias,
      NDArray attentionInputWeight,
      NDArray attentionInputBias,
      NDArray queryKeyValueWeight,
      NDArray attentionOutputWeight,
      NDArray attentionOutputBias,
      NDArray feedForwardInputWeight,
      NDArray feedForwardInputBias,
      NDArray expansionWeight,
      NDArray expansionBias,
      NDArray projectionWeight,
      NDArray projectionBias,
      NDArray outputWeight,
      NDArray outputBias) {

    /** 構造値と全パラメーター参照を検証する。 */
    public Parameters {
      if (hiddenSize <= 0) {
        throw new IllegalArgumentException("hiddenSize must be positive");
      }
      if (attentionHeads <= 0 || attentionWidth <= 0 || attentionWidth % attentionHeads != 0) {
        throw new IllegalArgumentException(
            "attentionWidth must be positive and divisible by attentionHeads");
      }
      if (feedForwardWidth <= 0) {
        throw new IllegalArgumentException("feedForwardWidth must be positive");
      }
      Objects.requireNonNull(inputNormWeight, "inputNormWeight");
      Objects.requireNonNull(inputNormBias, "inputNormBias");
      Objects.requireNonNull(attentionInputWeight, "attentionInputWeight");
      Objects.requireNonNull(attentionInputBias, "attentionInputBias");
      Objects.requireNonNull(queryKeyValueWeight, "queryKeyValueWeight");
      Objects.requireNonNull(attentionOutputWeight, "attentionOutputWeight");
      Objects.requireNonNull(attentionOutputBias, "attentionOutputBias");
      Objects.requireNonNull(feedForwardInputWeight, "feedForwardInputWeight");
      Objects.requireNonNull(feedForwardInputBias, "feedForwardInputBias");
      Objects.requireNonNull(expansionWeight, "expansionWeight");
      Objects.requireNonNull(expansionBias, "expansionBias");
      Objects.requireNonNull(projectionWeight, "projectionWeight");
      Objects.requireNonNull(projectionBias, "projectionBias");
      Objects.requireNonNull(outputWeight, "outputWeight");
      Objects.requireNonNull(outputBias, "outputBias");
    }

    private void requireDataType(DataType expectedDataType) {
      requireParameterDataType("inputNormWeight", inputNormWeight, expectedDataType);
      requireParameterDataType("inputNormBias", inputNormBias, expectedDataType);
      requireParameterDataType("attentionInputWeight", attentionInputWeight, expectedDataType);
      requireParameterDataType("attentionInputBias", attentionInputBias, expectedDataType);
      requireParameterDataType("queryKeyValueWeight", queryKeyValueWeight, expectedDataType);
      requireParameterDataType("attentionOutputWeight", attentionOutputWeight, expectedDataType);
      requireParameterDataType("attentionOutputBias", attentionOutputBias, expectedDataType);
      requireParameterDataType("feedForwardInputWeight", feedForwardInputWeight, expectedDataType);
      requireParameterDataType("feedForwardInputBias", feedForwardInputBias, expectedDataType);
      requireParameterDataType("expansionWeight", expansionWeight, expectedDataType);
      requireParameterDataType("expansionBias", expansionBias, expectedDataType);
      requireParameterDataType("projectionWeight", projectionWeight, expectedDataType);
      requireParameterDataType("projectionBias", projectionBias, expectedDataType);
      requireParameterDataType("outputWeight", outputWeight, expectedDataType);
      requireParameterDataType("outputBias", outputBias, expectedDataType);
    }
  }

  private static FusionRecipe.Constant addConstant(
      FusionRecipe.Builder builder, List<NDArray> values, String name, NDArray value) {
    values.add(value);
    return builder.addConstant(
        name, FusionRecipe.TensorSpec.fixed(value.getDataType(), value.getShape().getShape()));
  }

  private static void requireParameterDataType(
      String name, NDArray parameter, DataType expectedDataType) {
    if (parameter.getDataType() != expectedDataType) {
      throw new IllegalArgumentException(
          "player-memory parameter "
              + name
              + " dtype differs from expectedDataType: "
              + parameter.getDataType()
              + "/"
              + expectedDataType);
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
