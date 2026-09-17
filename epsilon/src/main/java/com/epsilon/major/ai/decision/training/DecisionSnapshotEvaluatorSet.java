package com.epsilon.major.ai.decision.training;

import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.major.ai.decision.arena.DecisionSnapshotEvaluatorProvider;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionEvaluator;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionEvaluatorFactory;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;

/**
 * 複数モデルによる自己対局で使う固定スナップショット評価器を遅延生成し、一括して解放します。
 *
 * <p>現在の候補だけは呼出し側が所有する評価器を共有します。過去スナップショットは対局収集用の採用モデルチェックポイントから個別に開き、スナップショット IDごとに同じ参照を再利用します。
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
   * 対戦相手として選ばれた過去対局収集用の採用モデルスナップショットを解決します。
   *
   * <p>解決不能時に候補へ戻すと、スナップショット対戦に見える同一方策対戦になるため直ちに例外を送出する。
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
