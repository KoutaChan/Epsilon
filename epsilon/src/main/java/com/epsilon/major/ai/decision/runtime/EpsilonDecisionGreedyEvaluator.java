package com.epsilon.major.ai.decision.runtime;

import com.epsilon.core.Action;
import com.epsilon.major.ai.decision.input.DecisionBucket;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.runtime.DecisionInferenceIngress;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * 各Decision行について最大方策確率の合法行動候補の位置だけを返せる評価器。
 *
 * <p>固定対戦評価のように無作為抽出、対局中の行動履歴、価値を必要としない呼び出し元が、全候補確率と行ごとの {@link
 * EpsilonDecisionInferenceServer.Prediction}を構築せずに同じ最大確率の行動選択を得るための能力境界である。
 */
public interface EpsilonDecisionGreedyEvaluator extends EpsilonDecisionEvaluator {

  @Override
  default EpsilonDecisionGreedyEvaluator resolveInferenceEvaluator(List<Action> legalActions) {
    return this;
  }

  /**
   * 通常評価器を最大確率の行動を選ぶ方式能力へ適応する。コンパクト出力を実装済みならそのまま返す。
   *
   * @param evaluator 既存のDecision 評価器
   * @return 同じ方策のargmaxを返す評価器
   */
  static EpsilonDecisionGreedyEvaluator adapt(EpsilonDecisionEvaluator evaluator) {
    if (evaluator instanceof EpsilonDecisionGreedyEvaluator greedyEvaluator) {
      return greedyEvaluator;
    }
    return new PredictionBackedGreedyEvaluator(evaluator);
  }

  /**
   * 同じ型付き容量区分に属するDecision行を評価し、最大方策確率の候補位置を返す。
   *
   * @param batch ホスト上に構築済みの型付きバッチ
   * @return 入力行と同じ順序の0始まりの合法行動候補の位置
   */
  int[] evaluateGreedyActionSlots(DecisionHostBatch batch);

  /** 型付きバッチをパイプラインへ投入し、各行の最大確率の候補位置を非同期に返します。 */
  default CompletableFuture<int[]> submitGreedyActionSlots(DecisionHostBatch batch) {
    return CompletableFuture.completedFuture(evaluateGreedyActionSlots(batch));
  }

  /** 取得済みホスト入力受付枠と型付きバッチを移譲し、各行の最大確率の候補位置を非同期に返します。 */
  default CompletableFuture<int[]> submitGreedyActionSlots(
      DecisionHostBatch batch, DecisionInferenceIngress ingress) {
    DecisionInferenceIngress.Ownership ownership = ingress.handoff(this);
    ownership.release();
    return submitGreedyActionSlots(batch);
  }
}

/** コンパクト出力を持たない特殊評価器について、通常Predictionのargmaxを使う変換用の実装。 */
class PredictionBackedGreedyEvaluator implements EpsilonDecisionGreedyEvaluator {

  protected final EpsilonDecisionEvaluator delegate;
  private final IdentityHashMap<EpsilonDecisionEvaluator, EpsilonDecisionGreedyEvaluator>
      routedEvaluators = new IdentityHashMap<>();

  PredictionBackedGreedyEvaluator(EpsilonDecisionEvaluator delegate) {
    this.delegate = delegate;
  }

  @Override
  public synchronized EpsilonDecisionGreedyEvaluator resolveInferenceEvaluator(
      List<Action> legalActions) {
    EpsilonDecisionEvaluator resolved = delegate.resolveInferenceEvaluator(legalActions);
    if (resolved == delegate) {
      return this;
    }
    return routedEvaluators.computeIfAbsent(resolved, EpsilonDecisionGreedyEvaluator::adapt);
  }

  @Override
  public List<EpsilonDecisionInferenceServer.Prediction> evaluateBatch(DecisionHostBatch batch) {
    return delegate.evaluateBatch(batch);
  }

  @Override
  public boolean needsInferenceWork() {
    return delegate.needsInferenceWork();
  }

  @Override
  public CompletableFuture<List<EpsilonDecisionInferenceServer.Prediction>> submitBatch(
      DecisionHostBatch batch) {
    return delegate.submitBatch(batch);
  }

  @Override
  public CompletableFuture<List<EpsilonDecisionInferenceServer.Prediction>> submitBatch(
      DecisionHostBatch batch, DecisionInferenceIngress ingress) {
    return delegate.submitBatch(batch, ingress);
  }

  @Override
  public int[] evaluateGreedyActionSlots(DecisionHostBatch batch) {
    return greedyActionSlots(delegate.evaluateBatch(batch));
  }

  @Override
  public CompletableFuture<int[]> submitGreedyActionSlots(DecisionHostBatch batch) {
    return delegate
        .submitBatch(batch)
        .thenApply(PredictionBackedGreedyEvaluator::greedyActionSlots);
  }

  @Override
  public CompletableFuture<int[]> submitGreedyActionSlots(
      DecisionHostBatch batch, DecisionInferenceIngress ingress) {
    return delegate
        .submitBatch(batch, ingress)
        .thenApply(PredictionBackedGreedyEvaluator::greedyActionSlots);
  }

  @Override
  public int preferredBatchSize() {
    return delegate.preferredBatchSize();
  }

  @Override
  public int preferredBatchSize(DecisionBucket bucket) {
    return delegate.preferredBatchSize(bucket);
  }

  @Override
  public int preferredStreamingBatchSize() {
    return delegate.preferredStreamingBatchSize();
  }

  @Override
  public int preferredStreamingBatchSize(DecisionBucket bucket) {
    return delegate.preferredStreamingBatchSize(bucket);
  }

  @Override
  public DecisionInferenceIngress.Attempt tryAcquireInferenceIngress() {
    return delegate.tryAcquireInferenceIngress();
  }

  static int[] greedyActionSlots(List<EpsilonDecisionInferenceServer.Prediction> predictions) {
    int[] selectedSlots = new int[predictions.size()];
    for (int row = 0; row < predictions.size(); row++) {
      selectedSlots[row] = predictions.get(row).greedyActionSlot();
    }
    return selectedSlots;
  }
}
