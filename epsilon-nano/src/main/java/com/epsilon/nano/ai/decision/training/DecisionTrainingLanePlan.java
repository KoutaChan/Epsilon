package com.epsilon.nano.ai.decision.training;

import java.util.ArrayList;
import java.util.List;

/**
 * 1回のパラメーター更新に使う小バッチを、推定計算量に応じて学習ワーカーへ配分する。
 *
 * <p>ワーカーごとのリストと割り当て順だけを保持し、サンプルの参照情報は複製しない。返したリストと配列は、計画の実行が終わるまで変更してはならない。
 */
final class DecisionTrainingLanePlan {

  private final ArrayList<ArrayList<DecisionTrainingMicroBatch>> batchesByLane;
  private final int[] activeLanes;
  private final int microBatchCount;

  private DecisionTrainingLanePlan(
      ArrayList<ArrayList<DecisionTrainingMicroBatch>> batchesByLane,
      int[] activeLanes,
      int microBatchCount) {
    this.batchesByLane = batchesByLane;
    this.activeLanes = activeLanes;
    this.microBatchCount = microBatchCount;
  }

  /**
   * 計算量降順のLPTで、累積計算量が最小の実行単位へ小バッチを一度だけ割り当てる。
   *
   * <p>{@code batches}と記述情報は並べ替えもコピーもしない。同計算量時の探索開始実行単位は更新段階ごとに回転し、短い末尾を実行単位 0へ固定しない。
   */
  static DecisionTrainingLanePlan create(
      ArrayList<DecisionTrainingMicroBatch> batches, int laneCount, int globalStepIndex) {
    if (batches.isEmpty() || laneCount <= 0) {
      throw new IllegalArgumentException("Decision lane plan requires batches and lanes");
    }
    long[] batchCosts = new long[batches.size()];
    for (int index = 0; index < batches.size(); index++) {
      batchCosts[index] = batches.get(index).estimatedCost();
    }
    int[] assignedLanes = assignEstimatedCosts(batchCosts, laneCount, globalStepIndex);
    ArrayList<ArrayList<DecisionTrainingMicroBatch>> batchesByLane = new ArrayList<>(laneCount);
    for (int lane = 0; lane < laneCount; lane++) {
      batchesByLane.add(new ArrayList<>());
    }
    for (int index = 0; index < batches.size(); index++) {
      DecisionTrainingMicroBatch batch = batches.get(index);
      int lane = assignedLanes[index];
      batchesByLane.get(lane).add(batch);
    }
    int activeCount = 0;
    for (List<DecisionTrainingMicroBatch> lane : batchesByLane) {
      if (!lane.isEmpty()) {
        activeCount++;
      }
    }
    int[] activeLanes = new int[activeCount];
    int active = 0;
    for (int lane = 0; lane < laneCount; lane++) {
      if (!batchesByLane.get(lane).isEmpty()) {
        activeLanes[active++] = lane;
      }
    }
    return new DecisionTrainingLanePlan(batchesByLane, activeLanes, batches.size());
  }

  /** 記述情報を必要としない純粋LPT。戻り値は元計算量インデックスごとの実行単位番号である。 */
  static int[] assignEstimatedCosts(long[] costs, int laneCount, int globalStepIndex) {
    if (costs.length == 0 || laneCount <= 0) {
      throw new IllegalArgumentException("Decision LPT requires costs and lanes");
    }
    int[] descending = new int[costs.length];
    for (int index = 0; index < descending.length; index++) {
      if (costs[index] <= 0L) {
        throw new IllegalArgumentException("Decision LPT cost must be positive");
      }
      descending[index] = index;
    }
    for (int index = 1; index < descending.length; index++) {
      int candidate = descending[index];
      int insertion = index;
      while (insertion > 0 && precedes(candidate, descending[insertion - 1], costs)) {
        descending[insertion] = descending[insertion - 1];
        insertion--;
      }
      descending[insertion] = candidate;
    }

    int[] laneByCost = new int[costs.length];
    long[] costByLane = new long[laneCount];
    int tieStart = Math.floorMod(globalStepIndex, laneCount);
    for (int index : descending) {
      int lane = leastLoadedLane(costByLane, tieStart);
      laneByCost[index] = lane;
      costByLane[lane] += costs[index];
      tieStart = Math.floorMod(lane + 1, laneCount);
    }
    return laneByCost;
  }

  private static boolean precedes(int candidate, int current, long[] costs) {
    return costs[candidate] > costs[current]
        || (costs[candidate] == costs[current] && candidate < current);
  }

  private static int leastLoadedLane(long[] costs, int tieStart) {
    int selected = tieStart;
    long minimum = costs[selected];
    for (int offset = 1; offset < costs.length; offset++) {
      int lane = Math.floorMod(tieStart + offset, costs.length);
      if (costs[lane] < minimum) {
        selected = lane;
        minimum = costs[lane];
      }
    }
    return selected;
  }

  int[] activeLanes() {
    return activeLanes;
  }

  List<DecisionTrainingMicroBatch> batches(int lane) {
    return batchesByLane.get(lane);
  }

  int microBatchCount() {
    return microBatchCount;
  }
}
