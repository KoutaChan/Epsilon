package com.epsilon.pico.ai.decision.training;

import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.pico.ai.decision.arena.DecisionSnapshotEvaluatorProvider;
import com.epsilon.pico.ai.decision.runtime.EpsilonDecisionEvaluator;
import com.epsilon.pico.ai.decision.runtime.EpsilonDecisionEvaluatorFactory;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * 自己対局に使う保存済みモデルの評価器を、必要になった時点で生成してまとめて解放する。
 *
 * <p>現在の候補については呼び出し側が管理する評価器を共有する。過去の採用モデルはチェックポイントから個別に読み込み、保存済みモデルの ID ごとに同じ評価器を再利用する。
 */
final class DecisionSnapshotEvaluatorSet
    implements DecisionSnapshotEvaluatorProvider, AutoCloseable {

  private final EpsilonDecisionSnapshotPool snapshotPool;
  private final EpsilonDecisionEvaluator candidateEvaluator;
  private final long candidateVersion;
  private final DecisionExecutionContext executionContext;
  private final SettingsLoader config;
  private final Map<Long, EpsilonDecisionEvaluatorFactory.Handle> handles = new HashMap<>();

  DecisionSnapshotEvaluatorSet(
      EpsilonDecisionSnapshotPool snapshotPool,
      EpsilonDecisionEvaluator candidateEvaluator,
      long candidateVersion,
      DecisionExecutionContext executionContext,
      SettingsLoader config) {
    this.config = config;
    this.snapshotPool = snapshotPool;
    this.candidateEvaluator = candidateEvaluator;
    this.candidateVersion = candidateVersion;
    this.executionContext = java.util.Objects.requireNonNull(executionContext, "executionContext");
  }

  @Override
  public EpsilonDecisionEvaluator evaluatorFor(long snapshotId) throws Exception {
    if (snapshotId == candidateVersion) {
      return candidateEvaluator;
    }
    EpsilonDecisionEvaluatorFactory.Handle existing = handles.get(snapshotId);
    if (existing != null) {
      return existing.evaluator();
    }
    EpsilonDecisionSnapshotPool.SnapshotEntry snapshot = snapshotPool.snapshot(snapshotId);
    Path checkpoint = resolveSnapshotCheckpoint(snapshotId, snapshot);
    EpsilonDecisionEvaluatorFactory.Handle handle =
        EpsilonDecisionEvaluatorFactory.openPolicyCheckpointEvaluator(
            checkpoint, executionContext, config);
    handles.put(snapshotId, handle);
    return handle.evaluator();
  }

  /**
   * 複数モデルによるが選んだ過去対局収集に使う採用モデルスナップショットを解決します。
   *
   * <p>解決不能時に候補へ戻すと、スナップショット対戦に見える同一方策対戦になるため不整合を検出して直ちに例外を送出するします。
   */
  static Path resolveSnapshotCheckpoint(
      long snapshotId, EpsilonDecisionSnapshotPool.SnapshotEntry snapshot) {
    if (snapshot == null) {
      throw new IllegalStateException(
          "Decision snapshot pool selected a missing snapshot: snapshotId=" + snapshotId);
    }
    try {
      return EpsilonDecisionSnapshotPool.requireArenaChampionCheckpoint(snapshot);
    } catch (IOException | RuntimeException e) {
      throw new IllegalStateException(
          "Decision snapshot checkpoint is not usable: snapshotId="
              + snapshotId
              + " path="
              + snapshot.path()
              + " reason="
              + e.getMessage(),
          e);
    }
  }

  @Override
  public void close() {
    for (EpsilonDecisionEvaluatorFactory.Handle handle : handles.values()) {
      handle.close();
    }
  }
}
