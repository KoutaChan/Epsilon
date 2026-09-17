package com.epsilon.major.ai.model.fusion;

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
import com.epsilon.major.ai.model.EpsilonMahjongStateEncoder;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 一つのデバイスパイプラインが所有する戦略文脈Transformerの永続Fusion 実行計画と出力循環バッファ。
 *
 * <p>モデル構成とは独立した名前付きのパラメータービューを定数へ結び付け、6 トークンの短系列を実行枠のバッファに保持した出力へ変換する。
 * 利用権は後段のデバイス処理が完了するまで順伝播単位で保持する。
 */
public final class EpsilonStrategicContextFusionExecution implements AutoCloseable {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(EpsilonStrategicContextFusionExecution.class);
  private static final int STRATEGIC_TOKEN_COUNT = 2 + GameState.NUM_PLAYERS;

  private final FusionRecipe.Dimension batch;
  private final FusionRecipe.Input input;
  private final FusionRecipe.Output output;
  private final Forward[] activeForwards;
  private FusionPlan plan;
  private FusionExecutable executable;
  private FusionSession session;
  private Throwable failure;
  private boolean unknownSubmission;
  private boolean incomplete;
  private boolean closed;

  EpsilonStrategicContextFusionExecution(
      FusionRecipe.Dimension batch,
      FusionRecipe.Input input,
      FusionRecipe.Output output,
      FusionPlan plan,
      FusionExecutable executable,
      FusionSession session,
      int executionSlots) {
    this.batch = batch;
    this.input = input;
    this.output = output;
    this.plan = plan;
    this.executable = executable;
    this.session = session;
    activeForwards = new Forward[executionSlots];
  }

  /**
   * 凍結済みパラメーター参照を結び付け、戦略文脈Transformerの永続実行計画を作る。
   *
   * @param manager 実行計画・セッション・出力循環バッファを所有するGPU 管理元
   * @param parameters モデル側で解決した名前付きパラメータービュー
   * @param expectedDataType 入力と凍結パラメーターに要求するデータ型
   * @param maximumBatch 実行計画が受理する最大バッチ
   * @param executionSlots 同時に保持する未回収順伝播数
   * @return パイプライン-局所的な戦略文脈の-コンテキスト Fusion実行境界
   */
  public static EpsilonStrategicContextFusionExecution create(
      NDManager manager,
      Parameters parameters,
      DataType expectedDataType,
      int maximumBatch,
      int executionSlots) {
    if (!manager.getDevice().isGpu()) {
      throw new UnsupportedOperationException("strategicContext=FUSION requires a GPU device");
    }

    BlockParameters firstBlock = parameters.blocks().get(0);
    int hiddenSize = (int) firstBlock.attentionInputWeight().getShape().get(0);
    int attentionWidth = (int) firstBlock.queryKeyValueWeight().getShape().get(0) / 3;
    int feedForwardWidth = (int) firstBlock.feedForwardExpansionWeight().getShape().get(0);

    List<NDArray> constants = new ArrayList<>(parameters.blocks().size() * 13);
    FusionRecipe.Builder builder = FusionRecipe.builder("epsilon-strategic-transformer");
    FusionRecipe.Dimension batch = builder.addDimension("batch", maximumBatch);
    FusionRecipe.Input input =
        builder.addInput(
            "tokens",
            FusionRecipe.TensorSpec.of(expectedDataType, batch, STRATEGIC_TOKEN_COUNT, hiddenSize));
    FusionRecipe.TransformerEncoderStackBuilder stackBuilder =
        builder.transformerEncoderStack(
            "encoded",
            input,
            EpsilonMahjongStateEncoder.ATTENTION_HEADS,
            attentionWidth,
            feedForwardWidth);
    for (int blockIndex = 0; blockIndex < parameters.blocks().size(); blockIndex++) {
      BlockParameters block = parameters.blocks().get(blockIndex);
      stackBuilder.addBlock(
          addConstant(
              builder,
              constants,
              "block" + blockIndex + "AttentionInputWeight",
              block.attentionInputWeight()),
          addConstant(
              builder,
              constants,
              "block" + blockIndex + "AttentionInputBias",
              block.attentionInputBias()),
          addConstant(
              builder,
              constants,
              "block" + blockIndex + "QueryKeyValueWeight",
              block.queryKeyValueWeight()),
          addConstant(
              builder,
              constants,
              "block" + blockIndex + "AttentionOutputWeight",
              block.attentionOutputWeight()),
          addConstant(
              builder,
              constants,
              "block" + blockIndex + "AttentionOutputBias",
              block.attentionOutputBias()),
          addConstant(
              builder,
              constants,
              "block" + blockIndex + "FeedForwardInputWeight",
              block.feedForwardInputWeight()),
          addConstant(
              builder,
              constants,
              "block" + blockIndex + "FeedForwardInputBias",
              block.feedForwardInputBias()),
          addConstant(
              builder,
              constants,
              "block" + blockIndex + "FeedForwardExpansionWeight",
              block.feedForwardExpansionWeight()),
          addConstant(
              builder,
              constants,
              "block" + blockIndex + "FeedForwardExpansionBias",
              block.feedForwardExpansionBias()),
          addConstant(
              builder,
              constants,
              "block" + blockIndex + "FeedForwardProjectionWeight",
              block.feedForwardProjectionWeight()),
          addConstant(
              builder,
              constants,
              "block" + blockIndex + "FeedForwardProjectionBias",
              block.feedForwardProjectionBias()),
          addConstant(
              builder, constants, "block" + blockIndex + "OutputWeight", block.outputWeight()),
          addConstant(builder, constants, "block" + blockIndex + "OutputBias", block.outputBias()));
    }
    FusionRecipe.TransformerEncoderStack encoded = stackBuilder.build();
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
        throw new IllegalStateException("strategic transformer plan is not native-only");
      }
      executable = plan.bind(bindings.build());
      session =
          executable.newSession(
              manager, FusionSessionConfig.builder().optOutputSlotCount(executionSlots).build());
      LOGGER.info(
          "Decision strategic transformer fusion: blocks={}, maximumBatch={}, slots={}, "
              + "scheduledStages/forward={}, executableStorageBytes={}, "
              + "retainedSessionStorageBytes={}, requiredExecutionLaneStorageBytes={}",
          parameters.blocks().size(),
          maximumBatch,
          executionSlots,
          parameters.blocks().size() * 9,
          report.getExecutableStorageBytes(),
          report.getRetainedSessionStorageBytes(executionSlots),
          report.getRequiredExecutionLaneStorageBytes());
      return new EpsilonStrategicContextFusionExecution(
          batch, input, output, plan, executable, session, executionSlots);
    } catch (RuntimeException | Error constructionFailure) {
      closeResource(constructionFailure, session);
      closeResource(constructionFailure, executable);
      closeResource(constructionFailure, plan);
      throw constructionFailure;
    }
  }

  /** 空き循環バッファ実行枠を一つ予約し、当該バッチの戦略文脈の順伝播を開始する。 */
  public Forward beginForward(NDManager workingManager) {
    if (closed) {
      throw new IllegalStateException("strategic inference execution is closed");
    }
    if (failure != null) {
      throw new IllegalStateException("strategic inference execution is poisoned", failure);
    }
    for (int slot = 0; slot < activeForwards.length; slot++) {
      if (activeForwards[slot] == null) {
        Forward forward = new Forward(this, slot, workingManager);
        activeForwards[slot] = forward;
        return forward;
      }
    }
    throw new IllegalStateException("no strategic inference slot is available");
  }

  /** デバイス完了待ちが必要な失敗済み順伝播が残っているかを返す。 */
  public boolean hasIncompleteWork() {
    return incomplete || unknownSubmission;
  }

  /** 最初に記録した非同期投入失敗を返す。 */
  public Throwable failure() {
    return failure;
  }

  /** デバイス完了確認後に、失敗済み順伝播の利用権と実行枠を解放する。 */
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
      throw new IllegalStateException("strategic inference has incomplete device work");
    }
    for (Forward forward : activeForwards) {
      if (forward != null) {
        throw new IllegalStateException("strategic inference has an active forward");
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
      throw new IllegalStateException("strategic forward does not own its slot");
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

  /** 一つのネットワーク順伝播が保持する実行枠のバッファに保持した戦略文脈出力。 */
  public static final class Forward {

    private final EpsilonStrategicContextFusionExecution owner;
    private final int slot;
    private final NDManager workingManager;
    private FusionOutputLease lease;
    private boolean encoded;
    private boolean sealed;
    private boolean poisoned;
    private boolean finished;

    private Forward(
        EpsilonStrategicContextFusionExecution owner, int slot, NDManager workingManager) {
      this.owner = owner;
      this.slot = slot;
      this.workingManager = workingManager;
    }

    /** 6 トークン入力を一度だけ投入し、有効なバッチに切った利用権で管理する出力を返す。 */
    public NDArray encode(NDArray tokens) {
      if (encoded || sealed || poisoned) {
        throw new IllegalStateException("strategic forward can encode only once");
      }
      long rowCount = tokens.getShape().get(0);
      DataType inputDataType = owner.input.getSpec().getDataType();
      NDArray planInput =
          tokens.getDataType() == inputDataType ? tokens : tokens.toType(inputDataType, false);
      if (planInput.getManager() != workingManager) {
        planInput.attach(workingManager);
      }
      boolean submitAttempted = false;
      try (FusionInvocation invocation = owner.session.acquire()) {
        invocation.setInput(owner.input, planInput);
        invocation.setDimension(owner.batch, rowCount);
        submitAttempted = true;
        lease = invocation.submit();
        NDArray activeOutput = lease.get(owner.output).get(NDIndex.sliceAxis(0, 0, rowCount));
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

    /** 成功した順伝播を封印し、完了後に呼ぶ利用権解放参照を返す。 */
    public AutoCloseable seal() {
      if (!encoded || sealed || poisoned) {
        throw new IllegalStateException("strategic forward cannot be sealed");
      }
      sealed = true;
      return this::releaseAfterCompletion;
    }

    /** 投入失敗を実行処理へ記録し、未投入なら実行枠を直ちに返す。 */
    public void poison(Throwable submitFailure) {
      if (poisoned) {
        return;
      }
      poisoned = true;
      if (owner.failure == null) {
        owner.failure = submitFailure;
      }
      boolean submittedWork = lease != null || owner.unknownSubmission;
      owner.incomplete |= submittedWork;
      if (!submittedWork) {
        try {
          releaseAfterCompletion();
        } catch (Throwable releaseFailure) {
          submitFailure.addSuppressed(releaseFailure);
        }
      }
    }

    /** 完了トークンを返さず失敗した投入を、ストリーム完了確認まで再利用禁止にする。 */
    private void poisonUnknownSubmission(Throwable submitFailure) {
      owner.unknownSubmission = true;
      poison(submitFailure);
    }

    private void releaseAfterCompletion() {
      if (finished) {
        return;
      }
      Throwable closeFailure = null;
      if (lease != null) {
        closeFailure = closeResource(closeFailure, lease);
        if (closeFailure == null) {
          lease = null;
        }
      }
      if (closeFailure == null) {
        finished = true;
        owner.finish(slot, this);
      } else {
        poisoned = true;
        owner.incomplete = true;
        if (owner.failure == null) {
          owner.failure = closeFailure;
        }
      }
      rethrow(closeFailure);
    }
  }

  /** エンコーダーが公開する、ブロック実体を含まない凍結済みパラメーター参照。 */
  public record Parameters(List<BlockParameters> blocks) {

    /** パラメーター順を固定し、呼び出し元の可変Listから切り離す。 */
    public Parameters {
      blocks = List.copyOf(blocks);
    }
  }

  /** 一つのTransformer ブロックを構成する名前付きの NDArray参照。 */
  public record BlockParameters(
      NDArray attentionInputWeight,
      NDArray attentionInputBias,
      NDArray queryKeyValueWeight,
      NDArray attentionOutputWeight,
      NDArray attentionOutputBias,
      NDArray feedForwardInputWeight,
      NDArray feedForwardInputBias,
      NDArray feedForwardExpansionWeight,
      NDArray feedForwardExpansionBias,
      NDArray feedForwardProjectionWeight,
      NDArray feedForwardProjectionBias,
      NDArray outputWeight,
      NDArray outputBias) {}

  /** ブロック所有のNDArray参照を複製せず定数対応付け順へ記録する。 */
  private static FusionRecipe.Constant addConstant(
      FusionRecipe.Builder builder, List<NDArray> constants, String name, NDArray value) {
    constants.add(value);
    return builder.addConstant(
        name, FusionRecipe.TensorSpec.fixed(value.getDataType(), value.getShape().getShape()));
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
