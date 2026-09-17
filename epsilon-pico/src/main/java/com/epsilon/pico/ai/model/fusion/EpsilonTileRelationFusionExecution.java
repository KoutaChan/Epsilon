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
import ai.djl.engine.fusion.IndexedRelationAttention;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import com.epsilon.core.Tile;
import com.epsilon.pico.ai.model.EpsilonTileRelationEncoder;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 一つのデバイスパイプラインが所有する牌関係Transformerの永続Fusion 実行計画と出力循環バッファ。
 *
 * <p>モデルブロックを参照せず、構築時に受け取った名前付きのパラメータービューだけを定数へ結び付ける。呼び出し元が所有する入力は借用し、
 * 出力利用権は後段のデバイス処理が終わるまで位置単位で保持する。
 */
public final class EpsilonTileRelationFusionExecution implements AutoCloseable {

  private static final Logger LOGGER =
      LoggerFactory.getLogger(EpsilonTileRelationFusionExecution.class);

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

  EpsilonTileRelationFusionExecution(
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
   * 重みを固定したパラメーター参照を結び付け、牌関係Transformerの永続実行計画を作る。
   *
   * @param manager 実行計画・セッション・出力循環バッファを所有するGPU 管理元
   * @param parameters モデル側で解決した名前付きパラメータービュー
   * @param expectedDataType 入力と凍結パラメーターに要求するデータ型
   * @param maximumBatch 実行計画が受理する最大バッチ
   * @param executionSlots 同時に保持する未回収順伝播数
   * @return パイプライン-局所的な牌-関係 Fusion実行境界
   */
  public static EpsilonTileRelationFusionExecution create(
      NDManager manager,
      Parameters parameters,
      DataType expectedDataType,
      int maximumBatch,
      int executionSlots) {
    if (!manager.getDevice().isGpu()) {
      throw new UnsupportedOperationException("tileRelation=FUSION requires a GPU device");
    }

    int hiddenSize = (int) parameters.finalOutputWeight().getShape().get(0);
    BlockParameters firstBlock = parameters.blocks().get(0);
    int attentionWidth = (int) firstBlock.queryKeyValueWeight().getShape().get(0) / 3;
    int feedForwardWidth = (int) firstBlock.feedForwardExpansionWeight().getShape().get(0);
    int relationCount = (int) firstBlock.relationKey().getShape().get(0);

    List<NDArray> constants = new ArrayList<>(parameters.blocks().size() * 13 + 3);
    FusionRecipe.Builder builder = FusionRecipe.builder("epsilon-tile-relation-transformer");
    FusionRecipe.Dimension batch = builder.addDimension("batch", maximumBatch);
    FusionRecipe.Input input =
        builder.addInput(
            "tokens",
            FusionRecipe.TensorSpec.of(expectedDataType, batch, Tile.NUM_TILE_TYPES, hiddenSize));
    FusionRecipe.Constant relationIds =
        addConstant(builder, constants, "relationIds", parameters.relationIds());

    int blockCount = parameters.blocks().size();
    FusionRecipe.Constant[] attentionInputWeights = new FusionRecipe.Constant[blockCount];
    FusionRecipe.Constant[] attentionInputBiases = new FusionRecipe.Constant[blockCount];
    for (int blockIndex = 0; blockIndex < blockCount; blockIndex++) {
      BlockParameters block = parameters.blocks().get(blockIndex);
      attentionInputWeights[blockIndex] =
          addConstant(
              builder,
              constants,
              "block" + blockIndex + "AttentionInputWeight",
              block.attentionInputWeight());
      attentionInputBiases[blockIndex] =
          addConstant(
              builder,
              constants,
              "block" + blockIndex + "AttentionInputBias",
              block.attentionInputBias());
    }
    FusionRecipe.Constant finalOutputWeight =
        addConstant(builder, constants, "finalOutputWeight", parameters.finalOutputWeight());
    FusionRecipe.Constant finalOutputBias =
        addConstant(builder, constants, "finalOutputBias", parameters.finalOutputBias());

    FusionRecipe.TransformerEncoderStackBuilder stackBuilder =
        builder.transformerEncoderStack(
            "encoded",
            input,
            EpsilonTileRelationEncoder.ATTENTION_HEADS,
            attentionWidth,
            feedForwardWidth);
    for (int blockIndex = 0; blockIndex < blockCount; blockIndex++) {
      BlockParameters block = parameters.blocks().get(blockIndex);
      IndexedRelationAttention relation =
          builder.indexedRelationAttention(
              relationIds,
              addConstant(
                  builder, constants, "block" + blockIndex + "RelationKey", block.relationKey()),
              addConstant(
                  builder, constants, "block" + blockIndex + "RelationBias", block.relationBias()));
      FusionRecipe.Constant outputWeight =
          blockIndex + 1 < blockCount ? attentionInputWeights[blockIndex + 1] : finalOutputWeight;
      FusionRecipe.Constant outputBias =
          blockIndex + 1 < blockCount ? attentionInputBiases[blockIndex + 1] : finalOutputBias;
      stackBuilder.addBlock(
          attentionInputWeights[blockIndex],
          attentionInputBiases[blockIndex],
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
          outputWeight,
          outputBias,
          relation);
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
        throw new IllegalStateException("tile relation transformer plan is not native-only");
      }
      executable = plan.bind(bindings.build());
      session =
          executable.newSession(
              manager, FusionSessionConfig.builder().optOutputSlotCount(executionSlots).build());
      long activeBatch = Math.min(384, maximumBatch);
      long activeLogicalBytes =
          Math.multiplyExact(
              activeBatch,
              Math.multiplyExact(
                  expectedDataType.getNumOfBytes(),
                  Tile.NUM_TILE_TYPES
                          * (2L * hiddenSize
                              + 3L * attentionWidth
                              + Math.max(
                                  feedForwardWidth,
                                  EpsilonTileRelationEncoder.ATTENTION_HEADS
                                      * (long) relationCount))
                      + EpsilonTileRelationEncoder.ATTENTION_HEADS
                          * (long) Tile.NUM_TILE_TYPES
                          * roundUp(Tile.NUM_TILE_TYPES, 8)));
      int expectedEagerStages = blockCount * 12 + 1;
      int scheduledStages = blockCount * 10 + 1;
      LOGGER.info(
          "Decision tile relation fusion: blocks={}, maximumBatch={}, slots={}, "
              + "expectedEagerStages/forward={}, scheduledStages/forward={}, "
              + "expectedDispatchReduction/forward={}, executableStorageBytes={}, "
              + "retainedSessionStorageBytes={}, requiredExecutionLaneStorageBytes={}, "
              + "activeLogicalBytesAtBatch{}={}",
          blockCount,
          maximumBatch,
          executionSlots,
          expectedEagerStages,
          scheduledStages,
          expectedEagerStages - scheduledStages,
          report.getExecutableStorageBytes(),
          report.getRetainedSessionStorageBytes(executionSlots),
          report.getRequiredExecutionLaneStorageBytes(),
          activeBatch,
          activeLogicalBytes);
      return new EpsilonTileRelationFusionExecution(
          batch, input, output, plan, executable, session, executionSlots);
    } catch (RuntimeException | Error constructionFailure) {
      closeResource(constructionFailure, session);
      closeResource(constructionFailure, executable);
      closeResource(constructionFailure, plan);
      throw constructionFailure;
    }
  }

  /** 空き循環バッファ位置を予約する。入力NDArrayの管理元所有権は移譲しない。 */
  public Forward beginForward(NDManager workingManager) {
    if (closed) {
      throw new IllegalStateException("tile relation inference execution is closed");
    }
    if (failure != null) {
      throw new IllegalStateException("tile relation inference execution is poisoned", failure);
    }
    for (int slot = 0; slot < activeForwards.length; slot++) {
      if (activeForwards[slot] == null) {
        Forward forward = new Forward(this, slot, workingManager);
        activeForwards[slot] = forward;
        return forward;
      }
    }
    throw new IllegalStateException("no tile relation inference slot is available");
  }

  /** デバイス完了待ちが必要な失敗済み順伝播が残っているかを返す。 */
  public boolean hasIncompleteWork() {
    return incomplete || unknownSubmission;
  }

  /** 最初に記録した非同期投入失敗を返す。 */
  public Throwable failure() {
    return failure;
  }

  /** デバイス完了確認後に、失敗済み順伝播の利用権と位置を解放する。 */
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
      throw new IllegalStateException("tile relation inference has incomplete device work");
    }
    for (Forward forward : activeForwards) {
      if (forward != null) {
        throw new IllegalStateException("tile relation inference has an active forward");
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
      throw new IllegalStateException("tile relation forward does not own its slot");
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

  /** 一つのネットワーク順伝播が保持する実行枠のバッファに保持した牌関係出力。 */
  public static final class Forward {

    private final EpsilonTileRelationFusionExecution owner;
    private final int slot;
    private final NDManager workingManager;
    private FusionOutputLease lease;
    private boolean encoded;
    private boolean sealed;
    private boolean poisoned;
    private boolean finished;

    private Forward(EpsilonTileRelationFusionExecution owner, int slot, NDManager workingManager) {
      this.owner = owner;
      this.slot = slot;
      this.workingManager = workingManager;
    }

    /** 借用した入力を一度だけ投入し、有効なバッチに切った利用権で管理する出力を返す。 */
    public NDArray encodeBorrowed(NDArray tokens) {
      if (encoded || sealed || poisoned) {
        throw new IllegalStateException("tile relation forward can encode only once");
      }
      if (tokens.getDataType() != owner.input.getSpec().getDataType()) {
        throw new IllegalArgumentException("tile relation fusion input type mismatch");
      }
      long rowCount = tokens.getShape().get(0);
      boolean submitAttempted = false;
      try (FusionInvocation invocation = owner.session.acquire()) {
        invocation.setInput(owner.input, tokens);
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
        throw new IllegalStateException("tile relation forward cannot be sealed");
      }
      sealed = true;
      return this::releaseAfterCompletion;
    }

    /** 投入失敗を実行へ記録し、未投入なら位置を直ちに返す。 */
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

  /** エンコーダーが公開する、ブロック実体を含まない重みを固定したパラメーター参照。 */
  public record Parameters(
      NDArray relationIds,
      NDArray finalOutputWeight,
      NDArray finalOutputBias,
      List<BlockParameters> blocks) {

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
      NDArray relationKey,
      NDArray relationBias,
      NDArray attentionOutputWeight,
      NDArray attentionOutputBias,
      NDArray feedForwardInputWeight,
      NDArray feedForwardInputBias,
      NDArray feedForwardExpansionWeight,
      NDArray feedForwardExpansionBias,
      NDArray feedForwardProjectionWeight,
      NDArray feedForwardProjectionBias) {}

  /** ブロック所有のNDArray参照を複製せず定数対応付け順へ記録する。 */
  private static FusionRecipe.Constant addConstant(
      FusionRecipe.Builder builder, List<NDArray> constants, String name, NDArray value) {
    constants.add(value);
    return builder.addConstant(
        name, FusionRecipe.TensorSpec.fixed(value.getDataType(), value.getShape().getShape()));
  }

  private static long roundUp(long value, long multiple) {
    return Math.floorDiv(Math.addExact(value, multiple - 1), multiple) * multiple;
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
