package com.epsilon.nano.ai.decision.fusion;

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
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import com.epsilon.config.settings.DecisionInferenceFusionMode;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 複数の推論スコアを、行優先の連続した FLOAT32 テンソルへまとめる。
 *
 * <p>FUSION経路は入力元数・各データ型・各幅・最大行数を含む構成識別情報ごとに実行計画を遅延生成する。通常は直前に使った構成の一致確認だけで再利用し、バッチごとのキー生成や
 * 実行計画構築を行わない。各プログラムはパイプラインの実行枠数と同じ数の再利用可能な出力バッファを持ち、投入 スレッドから直列に作成された複数利用権を独立して保持できる。
 */
public final class DecisionInferenceOutputPacker implements AutoCloseable {

  private static final Logger LOGGER = LoggerFactory.getLogger(DecisionInferenceOutputPacker.class);

  private final NDManager manager;
  private final DecisionInferenceFusionMode packing;
  private final int executionSlots;
  private final FusionCompiler compiler;
  private final ArrayList<FusionProgram> programs;
  private FusionProgram lastProgram;
  private boolean closed;
  private boolean resourcesReleased;

  /**
   * 指定した出力連結方式とパイプラインの実行枠数に合わせて構築する。
   *
   * <p>{@code executionSlots}は各Fusion プログラムの再利用する出力バッファ数である。{@link #pack}、利用権返却、{@link
   * #close()}は同じ投入スレッドから直列に呼ぶ。
   *
   * @param manager ネットワークと同じデバイスを所有する長寿命管理元
   * @param packing 出力連結方式
   * @param executionSlots 同時に保持できる未回収出力数
   */
  public DecisionInferenceOutputPacker(
      NDManager manager, DecisionInferenceFusionMode packing, int executionSlots) {
    if (executionSlots <= 0) {
      throw new IllegalArgumentException("executionSlots must be positive");
    }
    this.manager = Objects.requireNonNull(manager, "manager");
    this.packing = Objects.requireNonNull(packing, "packing");
    this.executionSlots = executionSlots;
    programs = new ArrayList<>();
    if (packing == DecisionInferenceFusionMode.FUSION) {
      if (!manager.getDevice().isGpu()) {
        throw new UnsupportedOperationException(
            "FUSION output packing requires a GPU inference device");
      }
      compiler = manager.getEngine().newFusionCompiler(manager.getDevice());
    } else {
      compiler = null;
    }
  }

  /**
   * スコアテンソルを有効行だけの連続FLOAT32 テンソルへ詰める。
   *
   * @param workingManager EAGER結果と入力スコアをバッチ完了まで所有する管理元
   * @param scores 形状がそれぞれ {@code [activeRows, width]} の入力元テンソル
   * @param activeRows このバッチの有効行数
   * @param maximumRows この容量区分を収容する出力記憶領域の最大行数
   * @return 連結したテンソルと必要なFusion 利用権
   */
  public DecisionPackedScores pack(
      NDManager workingManager, NDList scores, int activeRows, int maximumRows) {
    return pack(workingManager, scores, activeRows, maximumRows, null);
  }

  /** スコアテンソルと、それらが消費した上流デバイス利用権を一つの完了境界へ束ねる。 */
  public DecisionPackedScores pack(
      NDManager workingManager,
      NDList scores,
      int activeRows,
      int maximumRows,
      AutoCloseable upstream) {
    requireOpen();
    validateRows(scores, activeRows, maximumRows);
    if (packing == DecisionInferenceFusionMode.EAGER) {
      NDArray packed = NDArrays.concat(scores, 1).toType(DataType.FLOAT32, false);
      workingManager.attachAll(new NDList(packed));
      return DecisionPackedScores.borrowed(packed, upstream);
    }
    return program(scores, maximumRows).pack(scores, activeRows, upstream);
  }

  /**
   * 現在キャッシュしているFusion 構成識別情報数を返す。
   *
   * @return 必要になった時点に構築済みのプログラム数
   */
  int cachedProgramCount() {
    return programs.size();
  }

  private FusionProgram program(NDList scores, int maximumRows) {
    if (lastProgram != null && lastProgram.matches(scores, maximumRows)) {
      return lastProgram;
    }
    for (FusionProgram candidate : programs) {
      if (candidate.matches(scores, maximumRows)) {
        lastProgram = candidate;
        return candidate;
      }
    }
    FusionProgram created = createProgram(OutputSignature.from(scores, maximumRows));
    programs.add(created);
    lastProgram = created;
    return created;
  }

  private FusionProgram createProgram(OutputSignature signature) {
    FusionRecipe.Builder builder =
        FusionRecipe.builder("epsilon-decision-output-pack-" + programs.size());
    FusionRecipe.Dimension rows = builder.addDimension("rows", signature.maximumRows());
    FusionRecipe.Input[] inputs = new FusionRecipe.Input[signature.sourceCount()];
    FusionRecipe.Value[] values = new FusionRecipe.Value[signature.sourceCount()];
    for (int source = 0; source < inputs.length; source++) {
      FusionRecipe.Input input =
          builder.addInput(
              "source" + source,
              FusionRecipe.TensorSpec.of(
                  signature.dataType(source), rows, signature.width(source)));
      inputs[source] = input;
      values[source] = input;
    }
    FusionRecipe.OutputPack packed = builder.outputPack("packed", values);
    FusionRecipe.Output output = builder.addOutput("scores", packed);
    FusionRecipe recipe = builder.build();

    FusionPlan plan = null;
    FusionExecutable executable = null;
    FusionSession session = null;
    try {
      plan = compiler.prepare(recipe, FusionCompileConfig.defaults());
      FusionCompilationReport report = plan.getCompilationReport();
      requireNativeOnly(report.isNativeOnly(), recipe.getName());
      executable = plan.bind(FusionConstantBindings.builder(recipe).build());
      session =
          executable.newSession(
              manager, FusionSessionConfig.builder().optOutputSlotCount(executionSlots).build());
      logCompilation(signature, report, executionSlots);
      return new FusionProgram(signature, rows, inputs, output, plan, executable, session);
    } catch (RuntimeException | Error failure) {
      closeResource(failure, session);
      closeResource(failure, executable);
      closeResource(failure, plan);
      throw failure;
    }
  }

  private static void logCompilation(
      OutputSignature signature, FusionCompilationReport report, int executionSlots) {
    LOGGER.info(
        "Decision output fusion prepared: maxRows={}, dtypes={}, widths={}, backend={}, "
            + "commands={}, outputSlots={}, retainedSessionStorageBytes={}, "
            + "requiredExecutionLaneStorageBytes={}, nativeOnly={}",
        signature.maximumRows(),
        Arrays.toString(signature.dataTypes()),
        Arrays.toString(signature.widths()),
        report.getBackend(),
        report.getCommandCount(),
        executionSlots,
        report.getRetainedSessionStorageBytes(executionSlots),
        report.getRequiredExecutionLaneStorageBytes(),
        report.isNativeOnly());
  }

  static void requireNativeOnly(boolean nativeOnly, String recipeName) {
    if (!nativeOnly) {
      throw new IllegalStateException(
          "Decision output packing plan is not native-only: " + recipeName);
    }
  }

  private static void validateRows(NDList scores, int activeRows, int maximumRows) {
    Objects.requireNonNull(scores, "scores");
    if (scores.isEmpty()) {
      throw new IllegalArgumentException("scores must not be empty");
    }
    if (activeRows <= 0 || activeRows > maximumRows) {
      throw new IllegalArgumentException("activeRows must be between one and maximumRows");
    }
    for (NDArray score : scores) {
      Shape shape = score.getShape();
      if (shape.dimension() != 2 || shape.get(0) != activeRows || shape.get(1) <= 0) {
        throw new IllegalArgumentException(
            "each score must have shape [activeRows, positiveWidth]");
      }
    }
  }

  private void requireOpen() {
    if (closed) {
      throw new IllegalStateException("output packer is closed");
    }
  }

  /**
   * 例外後に保持した投入済み処理の完了を再確認する。
   *
   * <p>完了を確認できないプログラムは利用権とネイティブ資源を保持する。全プログラムを必ず試行し、最初の失敗に後続失敗を追加する。
   */
  void recoverIncompleteSubmissions() {
    Throwable failure = null;
    for (FusionProgram program : programs) {
      try {
        program.recoverPoisonedSubmission();
      } catch (Throwable recoveryFailure) {
        failure = addFailure(failure, recoveryFailure);
      }
    }
    rethrow(failure);
  }

  /** 呼び出し側が最終ストリームイベントを待った後、完了不明だった出力投入済み処理を依存解放する。 */
  public void releaseFailedSubmissionsAfterCompletion() {
    Throwable failure = null;
    for (FusionProgram program : programs) {
      failure = program.releaseFailedSubmissionAfterCompletion(failure);
    }
    rethrow(failure);
  }

  /** 利用権取得前の失敗を含む完了不明投入済み処理を保持している場合だけ{@code true}を返す。 */
  public boolean hasIncompleteWork() {
    for (FusionProgram program : programs) {
      if (program.hasIncompleteWork()) {
        return true;
      }
    }
    return false;
  }

  /** Fusion セッション、紐付けた実行可能オブジェクト、実行計画の順で実行コンテキスト資源を解放する。 */
  @Override
  public void close() {
    if (resourcesReleased) {
      return;
    }
    closed = true;
    lastProgram = null;
    recoverIncompleteSubmissions();
    Throwable failure = null;
    for (int index = programs.size() - 1; index >= 0; index--) {
      try {
        programs.get(index).close();
        programs.remove(index);
      } catch (Throwable closeFailure) {
        failure = addFailure(failure, closeFailure);
      }
    }
    resourcesReleased = programs.isEmpty();
    rethrow(failure);
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

  private record OutputSignature(int maximumRows, DataType[] dataTypes, long[] widths) {

    private static OutputSignature from(NDList scores, int maximumRows) {
      DataType[] dataTypes = new DataType[scores.size()];
      long[] widths = new long[scores.size()];
      for (int source = 0; source < scores.size(); source++) {
        NDArray score = scores.get(source);
        dataTypes[source] = score.getDataType();
        widths[source] = score.getShape().get(1);
      }
      return new OutputSignature(maximumRows, dataTypes, widths);
    }

    private int sourceCount() {
      return dataTypes.length;
    }

    private DataType dataType(int source) {
      return dataTypes[source];
    }

    private long width(int source) {
      return widths[source];
    }

    private boolean matches(NDList scores, int candidateMaximumRows) {
      if (maximumRows != candidateMaximumRows || scores.size() != dataTypes.length) {
        return false;
      }
      for (int source = 0; source < dataTypes.length; source++) {
        NDArray score = scores.get(source);
        if (score.getDataType() != dataTypes[source]
            || score.getShape().dimension() != 2
            || score.getShape().get(1) != widths[source]) {
          return false;
        }
      }
      return true;
    }
  }

  private static final class FusionProgram implements AutoCloseable {

    private final OutputSignature signature;
    private final FusionRecipe.Dimension rows;
    private final FusionRecipe.Input[] inputs;
    private final FusionRecipe.Output output;
    private FusionPlan plan;
    private FusionExecutable executable;
    private FusionSession session;
    private FusionOutputLease poisonedLease;
    private Throwable poisonCause;
    private boolean unknownSubmission;

    private FusionProgram(
        OutputSignature signature,
        FusionRecipe.Dimension rows,
        FusionRecipe.Input[] inputs,
        FusionRecipe.Output output,
        FusionPlan plan,
        FusionExecutable executable,
        FusionSession session) {
      this.signature = signature;
      this.rows = rows;
      this.inputs = inputs;
      this.output = output;
      this.plan = plan;
      this.executable = executable;
      this.session = session;
    }

    private boolean matches(NDList scores, int maximumRows) {
      return signature.matches(scores, maximumRows);
    }

    private DecisionPackedScores pack(NDList scores, int activeRows, AutoCloseable upstream) {
      if (hasIncompleteWork()) {
        throw new IllegalStateException(
            "fusion output program is poisoned by an incomplete submission", poisonCause);
      }
      FusionOutputLease lease = null;
      NDArray storage = null;
      boolean submitAttempted = false;
      try (FusionInvocation invocation = session.acquire()) {
        for (int source = 0; source < inputs.length; source++) {
          invocation.setInput(inputs[source], scores.get(source));
        }
        invocation.setDimension(rows, activeRows);
        submitAttempted = true;
        lease = invocation.submit();
        storage = lease.get(output);
        return DecisionPackedScores.leasedActiveRows(
            storage, activeRows, signature.maximumRows(), lease, upstream);
      } catch (RuntimeException | Error failure) {
        if (lease != null) {
          releaseFailedSubmission(lease, failure);
        } else if (submitAttempted) {
          unknownSubmission = true;
          poisonCause = failure;
        }
        throw failure;
      }
    }

    /**
     * 投入後の構築失敗ではデバイス完了を確認してから枠を返す。
     *
     * <p>通常経路には同期を加えない。例外時だけ利用権が記憶する投入済み処理
     * ストリームの完了を待ち、その後に枠を返す。完了を確認できない場合だけプログラムを再利用不可に設定し、利用権とネイティブ資源を保持したまま再利用を禁止する。
     */
    private void releaseFailedSubmission(FusionOutputLease lease, Throwable failure) {
      try {
        lease.synchronize();
        lease.close();
      } catch (RuntimeException | Error completionFailure) {
        failure.addSuppressed(completionFailure);
        poisonedLease = lease;
        poisonCause = failure;
      }
    }

    /** 解放時に再利用不可に設定済み投入済み処理の完了を再確認する。完了を確認できなければネイティブ資源を解放しない。 */
    private void recoverPoisonedSubmission() {
      if (unknownSubmission) {
        throw new IllegalStateException(
            "fusion output submission has no completion token", poisonCause);
      }
      if (poisonedLease == null) {
        return;
      }
      poisonedLease.synchronize();
      poisonedLease.close();
      poisonedLease = null;
      poisonCause = null;
    }

    private boolean hasIncompleteWork() {
      return poisonedLease != null || unknownSubmission;
    }

    private Throwable releaseFailedSubmissionAfterCompletion(Throwable failure) {
      unknownSubmission = false;
      if (poisonedLease != null) {
        try {
          poisonedLease.close();
          poisonedLease = null;
        } catch (Throwable closeFailure) {
          return addFailure(failure, closeFailure);
        }
      }
      poisonCause = null;
      return failure;
    }

    @Override
    public void close() {
      recoverPoisonedSubmission();
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
}
