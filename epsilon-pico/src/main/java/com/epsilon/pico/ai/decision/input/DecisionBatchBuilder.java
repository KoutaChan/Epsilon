package com.epsilon.pico.ai.decision.input;

import com.epsilon.ai.decision.DecisionBranchTarget;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.RoundPublicStateIndex;
import com.epsilon.engine.EngineDecisionBuffer;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.GameEngine;
import java.util.ArrayDeque;
import java.util.List;
import java.util.Objects;

/**
 * Decision の入力を行順に書き込み、バッチを構築する。
 *
 * <p>推論では実在する遷移だけを保存し、学習では固定長の連続バッファを使う。両者で同じ入力エンコーダーを使用し、各行は一度だけ確定する。宣言した容量をすべて埋めると構築を完了でき、その後は再利用しない。学習時には選択行動、確率分布、教師値、学習重みを入力と同じ行順で保存する。
 */
public final class DecisionBatchBuilder {

  /** 並列エンコーダー実行処理が所有し、バッチをまたいで再利用する行encoding 作業領域再利用領域。 */
  public static final class EncodingWorkspacePool {
    private final ArrayDeque<EncodingWorkspace> available;

    public EncodingWorkspacePool(int workspaceCount) {
      if (workspaceCount <= 0) {
        throw new IllegalArgumentException("workspaceCount must be positive");
      }
      available = new ArrayDeque<>(workspaceCount);
      for (int index = 0; index < workspaceCount; index++) {
        available.addLast(new EncodingWorkspace());
      }
    }

    private EncodingWorkspace borrow() {
      synchronized (available) {
        EncodingWorkspace workspace = available.pollFirst();
        return workspace != null ? workspace : new EncodingWorkspace();
      }
    }

    private void release(EncodingWorkspace workspace) {
      synchronized (available) {
        available.addFirst(workspace);
      }
    }
  }

  private static final class EncodingWorkspace {
    private final EngineDecisionBuffer decision = new EngineDecisionBuffer();
    private final DecisionFeatureEncoder.Scratch encoding = new DecisionFeatureEncoder.Scratch();
  }

  /** 一つの行範囲を同じワーカーで符号化し、終了時に作業領域を返す借用単位。 */
  public final class EncodingSession implements AutoCloseable {
    private EncodingWorkspace workspace;

    private EncodingSession(EncodingWorkspace workspace) {
      this.workspace = workspace;
    }

    /** 現在処理待ちのエンジン判断を予約済みの行へ直接書きます。 */
    public void encodeInferenceRow(
        int rowIndex,
        GameEngine engine,
        EngineDecisionPoint point,
        DecisionBoundaryContext boundaryContext) {
      EncodingWorkspace current = requireWorkspace();
      engine.analyzeDecision(point, current.decision);
      DecisionRowEncoder.encodeDecisionRow(
          current.decision, boundaryContext, hostBatch, rowIndex, current.encoding);
    }

    /** 借用観測を解析し、予約済みの行へ直接書きます。 */
    public void encodeObservedInferenceRow(
        int rowIndex, PublicObservation observation, List<Action> legalActions) {
      EncodingWorkspace current = requireWorkspace();
      observation.analyze(legalActions, current.decision);
      DecisionRowEncoder.encodeDecisionRow(
          current.decision,
          DecisionBoundaryContext.uniform(),
          hostBatch,
          rowIndex,
          current.encoding);
    }

    private EncodingWorkspace requireWorkspace() {
      if (workspace == null) {
        throw new IllegalStateException("Encoding session is closed");
      }
      requireOpen();
      return workspace;
    }

    /** 同じワーカーで書き込みを終えてから呼び、借用中の作業領域を一度だけ返します。 */
    @Override
    public void close() {
      if (workspace != null) {
        workspacePool.release(workspace);
        workspace = null;
      }
    }
  }

  private final DecisionHostBatch hostBatch;
  private final EncodingWorkspacePool workspacePool;
  private boolean sealed;

  private DecisionBatchBuilder(
      int rowCapacity,
      DecisionBucket bucket,
      boolean includeTrainingTargets,
      EncodingWorkspacePool workspacePool) {
    hostBatch =
        includeTrainingTargets
            ? new DecisionHostBatch(rowCapacity, bucket, true)
            : DecisionHostBatch.inference(rowCapacity, bucket);
    this.workspacePool = Objects.requireNonNull(workspacePool, "workspacePool");
  }

  /**
   * 教師値を持たない推論バッチビルダーを作る。
   *
   * @param rowCapacity バッチへ格納する行数
   * @param bucket 行動数と遷移数を収容する密な形状
   * @return 推論入力専用ビルダー
   */
  public static DecisionBatchBuilder inference(int rowCapacity, DecisionBucket bucket) {
    return inference(rowCapacity, bucket, new EncodingWorkspacePool(1));
  }

  /** 実行処理所有作業領域再利用領域を借用する推論バッチビルダーを作る。 */
  public static DecisionBatchBuilder inference(
      int rowCapacity, DecisionBucket bucket, EncodingWorkspacePool workspacePool) {
    return new DecisionBatchBuilder(rowCapacity, bucket, false, workspacePool);
  }

  /**
   * 入力と学習教師値を持つバッチビルダーを作る。
   *
   * @param rowCapacity バッチへ格納する行数
   * @param bucket 行動数と遷移数を収容する密な形状
   * @return 学習入力・教師値用ビルダー
   */
  public static DecisionBatchBuilder training(int rowCapacity, DecisionBucket bucket) {
    return new DecisionBatchBuilder(rowCapacity, bucket, true, new EncodingWorkspacePool(1));
  }

  /**
   * 合法行動集合を推論向けの最小密な容量区分へ収容する。
   *
   * @param legalActions 候補位置順の合法行動一覧
   * @return 行動数と鳴き後遷移有無を収容できる容量区分
   */
  public static DecisionBucket selectInferenceBucket(List<Action> legalActions) {
    return DecisionBucket.forInferenceCounts(
        legalActions.size(), hasImmediateDiscardTransitions(legalActions) ? 2 : 1);
  }

  /**
   * 合法手を伴わないafter-状態の状態/価値表現用1行バッチを構築する。
   *
   * @param state 符号化する局状態
   * @param player 観測主体の席番号（0-3）
   * @return 行動軸をdummy 1 位置にした状態だけを使うバッチ
   */
  public static DecisionHostBatch stateOnlyBatch(GameState state, int player) {
    DecisionBatchBuilder batchBuilder =
        new DecisionBatchBuilder(1, new DecisionBucket(1, 1), false, new EncodingWorkspacePool(1));
    EncodingWorkspace workspace = batchBuilder.workspacePool.borrow();
    try {
      workspace.decision.analyze(state, player, List.of(Action.pass()), state.publicState());
      DecisionRowEncoder.encodeStateOnlyRow(workspace.decision, batchBuilder.hostBatch, 0);
      batchBuilder.hostBatch.commitStateOnlyRow(0);
      batchBuilder.sealed = true;
      return batchBuilder.hostBatch;
    } finally {
      batchBuilder.workspacePool.release(workspace);
    }
  }

  /** 現在保留中の中の対局エンジン判断を次の推論行へ直接書く。 */
  public int addInferenceRow(
      GameEngine engine, EngineDecisionPoint point, DecisionBoundaryContext boundaryContext) {
    requireOpen();
    int rowIndex = hostBatch.size();
    EncodingWorkspace workspace = workspacePool.borrow();
    try {
      engine.analyzeDecision(point, workspace.decision);
      DecisionRowEncoder.encodeDecisionRow(
          workspace.decision, boundaryContext, hostBatch, rowIndex, workspace.encoding);
      hostBatch.commitActionRow(rowIndex);
      return rowIndex;
    } finally {
      workspacePool.release(workspace);
    }
  }

  /** 呼出しワーカーの行範囲で再利用する作業領域を借ります。 */
  public EncodingSession openEncoding() {
    requireOpen();
    return new EncodingSession(workspacePool.borrow());
  }

  /** 現在保留中の中の対局エンジン判断を予約済み推論行へ直接書く。 */
  public void encodeInferenceRow(
      int rowIndex,
      GameEngine engine,
      EngineDecisionPoint point,
      DecisionBoundaryContext boundaryContext) {
    try (var session = openEncoding()) {
      session.encodeInferenceRow(rowIndex, engine, point, boundaryContext);
    }
  }

  /** 対局実行環境の借用観測を一度だけ解析し、系列固有の入力行へ書き込みます。 */
  public void encodeObservedInferenceRow(
      int rowIndex, PublicObservation observation, List<Action> legalActions) {
    try (var session = openEncoding()) {
      session.encodeObservedInferenceRow(rowIndex, observation, legalActions);
    }
  }

  /** GameEngine外で復元された局面を次の推論行へ書く。 */
  public int addDetachedInferenceRow(
      GameState state,
      int player,
      List<Action> legalActions,
      RoundPublicStateIndex publicState,
      DecisionBoundaryContext boundaryContext) {
    requireOpen();
    int rowIndex = hostBatch.size();
    EncodingWorkspace workspace = workspacePool.borrow();
    try {
      workspace.decision.analyze(state, player, legalActions, publicState);
      DecisionRowEncoder.encodeDecisionRow(
          workspace.decision, boundaryContext, hostBatch, rowIndex, workspace.encoding);
      hostBatch.commitActionRow(rowIndex);
      return rowIndex;
    } finally {
      workspacePool.release(workspace);
    }
  }

  /** GameEngine外で復元された局面を予約済み推論行へ書く。 */
  public void encodeDetachedInferenceRow(
      int rowIndex,
      GameState state,
      int player,
      List<Action> legalActions,
      RoundPublicStateIndex publicState,
      DecisionBoundaryContext boundaryContext) {
    requireOpen();
    EncodingWorkspace workspace = workspacePool.borrow();
    try {
      workspace.decision.analyze(state, player, legalActions, publicState);
      DecisionRowEncoder.encodeDecisionRow(
          workspace.decision, boundaryContext, hostBatch, rowIndex, workspace.encoding);
    } finally {
      workspacePool.release(workspace);
    }
  }

  /** 復元局面に必要な最小容量区分を返す。 */
  public static DecisionBucket minimumDetachedBucket(
      GameState state, int player, List<Action> legalActions, RoundPublicStateIndex publicState) {
    EngineDecisionBuffer decision = new EngineDecisionBuffer();
    decision.analyze(state, player, legalActions, publicState);
    return DecisionBucket.forRequiredCounts(
        decision.actionCount(), decision.maximumTransitionsPerAction());
  }

  /**
   * 並列に書き込み済みの全推論行を行順に確定してバッチを返す。
   *
   * @return 書き込みの確定済みホスト側バッチ
   */
  public DecisionHostBatch buildEncodedInferenceRows() {
    requireOpen();
    for (int row = 0; row < hostBatch.capacity(); row++) {
      hostBatch.commitActionRow(row);
    }
    sealed = true;
    return hostBatch;
  }

  /**
   * 準備済みの入力計画と選択行動学習教師値を次の行へ追加する。
   *
   * @param state 復元局面
   * @param player 観測席
   * @param legalActions 合法手
   * @param publicState 意思決定時点の公開インデックス
   * @param boundaryContext 判断境界文脈
   * @param chosenSlot 実際に選択された候補位置
   * @param behaviorPolicy 候補位置順の探索適用後の確率
   * @param rolloutPolicy 探索適用前の更新の基準となるモデル確率
   * @param valueTarget 観測主体から見た期待効用教師値
   * @param advantage 選択行動へだけ適用する効用アドバンテージ
   * @param actorWeight 方策損失へ掛けるサンプル重み
   * @param sampleWeight 方策・価値共通のサンプル重み
   * @return 追加したバッチ行インデックス
   */
  public int addDetachedTrainingRow(
      GameState state,
      int player,
      List<Action> legalActions,
      RoundPublicStateIndex publicState,
      DecisionBoundaryContext boundaryContext,
      int chosenSlot,
      float[] behaviorPolicy,
      float[] rolloutPolicy,
      float valueTarget,
      float advantage,
      float actorWeight,
      float sampleWeight) {
    requireOpen();
    requireTrainingBuilder();
    int rowIndex = hostBatch.size();
    int legalActionCount = legalActions.size();
    validateTargets(
        rowIndex,
        legalActionCount,
        chosenSlot,
        behaviorPolicy,
        rolloutPolicy,
        valueTarget,
        advantage,
        actorWeight,
        sampleWeight);
    EncodingWorkspace workspace = workspacePool.borrow();
    try {
      workspace.decision.analyze(state, player, legalActions, publicState);
      DecisionRowEncoder.encodeDecisionRow(
          workspace.decision, boundaryContext, hostBatch, rowIndex, workspace.encoding);
      writeTrustedTargets(
          rowIndex,
          legalActionCount,
          chosenSlot,
          behaviorPolicy,
          rolloutPolicy,
          valueTarget,
          advantage,
          actorWeight,
          sampleWeight);
      hostBatch.commitActionRow(rowIndex);
      return rowIndex;
    } finally {
      workspacePool.release(workspace);
    }
  }

  /**
   * 検証済みサンプル教師値を、既に符号化済みの入力行へ再走査せず組み合わせる。
   *
   * @param sourceBatch 入力連続バッファを保持する符号化済みバッチ
   * @param sourceRowIndex 複製元の行インデックス
   * @param chosenSlot 実際に選択された候補位置
   * @param behaviorPolicy 候補位置順の探索適用後の確率
   * @param rolloutPolicy 探索適用前の更新の基準となるモデル確率
   * @param valueTarget 観測主体から見た期待効用教師値
   * @param advantage 選択行動へだけ適用する効用アドバンテージ
   * @param actorWeight 方策損失へ掛けるサンプル重み
   * @param sampleWeight 方策・価値共通のサンプル重み
   * @return 追加したバッチ行インデックス
   */
  public int addTrustedEncodedTrainingRow(
      DecisionHostBatch sourceBatch,
      int sourceRowIndex,
      int chosenSlot,
      float[] behaviorPolicy,
      float[] rolloutPolicy,
      float valueTarget,
      float advantage,
      float actorWeight,
      float sampleWeight) {
    requireOpen();
    requireTrainingBuilder();
    int legalActionCount = sourceBatch.legalActionCount(sourceRowIndex);
    int rowIndex = hostBatch.size();
    hostBatch.copyNextInputRowFrom(sourceBatch, sourceRowIndex, rowIndex);
    writeTrustedTargets(
        rowIndex,
        legalActionCount,
        chosenSlot,
        behaviorPolicy,
        rolloutPolicy,
        valueTarget,
        advantage,
        actorWeight,
        sampleWeight);
    hostBatch.commitActionRow(rowIndex);
    return rowIndex;
  }

  /**
   * 収容上限まで埋まったバッチを書き込みの確定して返す。
   *
   * @return 全行確定済みのホスト側バッチ
   */
  public DecisionHostBatch build() {
    requireOpen();
    if (hostBatch.size() != hostBatch.capacity()) {
      throw new IllegalStateException(
          "batch must be filled exactly before build: "
              + hostBatch.size()
              + "/"
              + hostBatch.capacity());
    }
    sealed = true;
    return hostBatch;
  }

  /** 指定行の分岐比較教師値を書き込む。 */
  public void writeBranchTarget(int row, DecisionBranchTarget target) {
    hostBatch.trainingTargets().writeBranchTarget(row, target);
  }

  private void requireOpen() {
    if (sealed) {
      throw new IllegalStateException("batch builder is sealed");
    }
  }

  private static boolean hasImmediateDiscardTransitions(List<Action> legalActions) {
    for (Action action : legalActions) {
      if (action.type() == Action.Type.CHI || action.type() == Action.Type.PON) {
        return true;
      }
    }
    return false;
  }

  private void requireTrainingBuilder() {
    if (!hostBatch.hasTrainingTargets()) {
      throw new IllegalStateException("operation requires a training builder");
    }
  }

  private void writeTrustedTargets(
      int row,
      int legalActionCount,
      int chosenSlot,
      float[] behaviorPolicy,
      float[] rolloutPolicy,
      float valueTarget,
      float advantage,
      float actorWeight,
      float sampleWeight) {
    hostBatch
        .trainingTargets()
        .writeTrustedRow(
            row,
            legalActionCount,
            chosenSlot,
            behaviorPolicy,
            rolloutPolicy,
            valueTarget,
            advantage,
            actorWeight,
            sampleWeight);
  }

  private void validateTargets(
      int row,
      int legalActionCount,
      int chosenSlot,
      float[] behaviorPolicy,
      float[] rolloutPolicy,
      float valueTarget,
      float advantage,
      float actorWeight,
      float sampleWeight) {
    hostBatch
        .trainingTargets()
        .validateRow(
            row,
            legalActionCount,
            chosenSlot,
            behaviorPolicy,
            rolloutPolicy,
            valueTarget,
            advantage,
            actorWeight,
            sampleWeight);
  }
}
