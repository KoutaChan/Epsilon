package com.epsilon.major.ai.decision.runtime;

import com.epsilon.core.Action;
import com.epsilon.major.ai.decision.input.DecisionBucket;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

/**
 * 通常の局面を候補モデルへ、見送り・チー・ポンだけを選べる応答局面を比較元モデルへ振り分ける。
 *
 * <p>RONや槓を同時に含む応答は候補へ残す。これにより、鳴き選択以外の応答判断能力を巻き戻さず、PASS対CHI/PONの寄与だけを固定牌山
 * 対戦評価で分離する。委譲先の資源所有権は呼び出し元にあり、この包むオブジェクトは解放しない。
 */
final class EpsilonDecisionReactionHybridEvaluator implements EpsilonDecisionEvaluator {

  static final String REACTION_DEFINITION =
      "legal actions contain PASS and CHI/PON, and contain no action outside PASS/CHI/PON";
  private static final Executor SUBMISSION_EXECUTOR =
      command -> Thread.ofVirtual().name("epsilon-decision-reaction-hybrid-submit").start(command);

  private final EpsilonDecisionEvaluator candidateEvaluator;
  private final EpsilonDecisionEvaluator parentEvaluator;
  private long candidateRows;
  private long parentReactionRows;

  EpsilonDecisionReactionHybridEvaluator(
      EpsilonDecisionEvaluator candidateEvaluator, EpsilonDecisionEvaluator parentEvaluator) {
    this.candidateEvaluator = candidateEvaluator;
    this.parentEvaluator = parentEvaluator;
  }

  @Override
  public int preferredBatchSize() {
    return Math.min(candidateEvaluator.preferredBatchSize(), parentEvaluator.preferredBatchSize());
  }

  @Override
  public int preferredBatchSize(DecisionBucket bucket) {
    return Math.min(
        candidateEvaluator.preferredBatchSize(bucket), parentEvaluator.preferredBatchSize(bucket));
  }

  @Override
  public int preferredStreamingBatchSize() {
    return Math.min(
        candidateEvaluator.preferredStreamingBatchSize(),
        parentEvaluator.preferredStreamingBatchSize());
  }

  @Override
  public int preferredStreamingBatchSize(DecisionBucket bucket) {
    return Math.min(
        candidateEvaluator.preferredStreamingBatchSize(bucket),
        parentEvaluator.preferredStreamingBatchSize(bucket));
  }

  @Override
  public synchronized EpsilonDecisionEvaluator resolveInferenceEvaluator(
      List<Action> legalActions) {
    if (routesToParent(legalActions)) {
      parentReactionRows = Math.addExact(parentReactionRows, 1L);
      return parentEvaluator;
    }
    candidateRows = Math.addExact(candidateRows, 1L);
    return candidateEvaluator;
  }

  @Override
  public List<EpsilonDecisionInferenceServer.Prediction> evaluateBatch(DecisionHostBatch batch) {
    Partition partition = partition(batch);
    if (partition.parentCount() == 0) {
      return candidateEvaluator.evaluateBatch(batch);
    }
    if (partition.candidateCount() == 0) {
      return parentEvaluator.evaluateBatch(batch);
    }

    ArrayList<EpsilonDecisionInferenceServer.Prediction> out =
        new ArrayList<>(Collections.nCopies(batch.size(), null));
    evaluateRows(
        candidateEvaluator,
        batch,
        partition.candidateRows(),
        partition.candidateCount(),
        out,
        "candidate");
    evaluateRows(
        parentEvaluator, batch, partition.parentRows(), partition.parentCount(), out, "parent");
    return out;
  }

  @Override
  public CompletableFuture<List<EpsilonDecisionInferenceServer.Prediction>> submitBatch(
      DecisionHostBatch batch) {
    return CompletableFuture.supplyAsync(() -> evaluateBatch(batch), SUBMISSION_EXECUTOR);
  }

  synchronized RoutingStats routingStats() {
    return new RoutingStats(candidateRows, parentReactionRows);
  }

  synchronized void resetRoutingStats() {
    candidateRows = 0L;
    parentReactionRows = 0L;
  }

  static boolean routesToParent(DecisionHostBatch batch, int row) {
    boolean pass = false;
    boolean call = false;
    int legalCount = batch.legalActionCount(row);
    for (int slot = 0; slot < legalCount; slot++) {
      int actionId = batch.legalActionId(row, slot);
      switch (Action.fromIndex(actionId).type()) {
        case PASS -> pass = true;
        case CHI, PON -> call = true;
        default -> {
          return false;
        }
      }
    }
    return pass && call;
  }

  static boolean routesToParent(List<Action> legalActions) {
    boolean pass = false;
    boolean call = false;
    for (Action action : legalActions) {
      switch (action.type()) {
        case PASS -> pass = true;
        case CHI, PON -> call = true;
        default -> {
          return false;
        }
      }
    }
    return pass && call;
  }

  private static void evaluateRows(
      EpsilonDecisionEvaluator evaluator,
      DecisionHostBatch source,
      int[] sourceRows,
      int rowCount,
      List<EpsilonDecisionInferenceServer.Prediction> destination,
      String label) {
    if (rowCount == 0) {
      return;
    }
    DecisionHostBatch selected =
        rowCount == source.size() ? source : source.selectRows(sourceRows, rowCount);
    List<EpsilonDecisionInferenceServer.Prediction> predictions = evaluator.evaluateBatch(selected);
    if (predictions.size() != rowCount) {
      throw new IllegalStateException(
          label
              + " prediction count mismatch: requested="
              + rowCount
              + " got="
              + predictions.size());
    }
    for (int row = 0; row < rowCount; row++) {
      destination.set(sourceRows[row], predictions.get(row));
    }
  }

  private synchronized Partition partition(DecisionHostBatch batch) {
    boolean[] parentRoute = new boolean[batch.size()];
    int parentCount = 0;
    for (int row = 0; row < batch.size(); row++) {
      if (routesToParent(batch, row)) {
        parentRoute[row] = true;
        parentCount++;
      }
    }
    int candidateCount = batch.size() - parentCount;
    candidateRows = Math.addExact(candidateRows, candidateCount);
    parentReactionRows = Math.addExact(parentReactionRows, parentCount);
    if (candidateCount == 0 || parentCount == 0) {
      return new Partition(null, candidateCount, null, parentCount);
    }
    int[] candidateSourceRows = new int[candidateCount];
    int[] parentSourceRows = new int[parentCount];
    int candidateIndex = 0;
    int parentIndex = 0;
    for (int row = 0; row < parentRoute.length; row++) {
      if (parentRoute[row]) {
        parentSourceRows[parentIndex++] = row;
      } else {
        candidateSourceRows[candidateIndex++] = row;
      }
    }
    return new Partition(candidateSourceRows, candidateCount, parentSourceRows, parentCount);
  }

  private record Partition(
      int[] candidateRows, int candidateCount, int[] parentRows, int parentCount) {}

  record RoutingStats(long candidateRows, long parentReactionRows) {
    long totalRows() {
      return Math.addExact(candidateRows, parentReactionRows);
    }

    double parentReactionRate() {
      long total = totalRows();
      return total == 0L ? 0.0 : parentReactionRows / (double) total;
    }
  }
}
