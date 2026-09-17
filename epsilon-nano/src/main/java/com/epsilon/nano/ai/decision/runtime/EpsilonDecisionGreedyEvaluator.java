package com.epsilon.nano.ai.decision.runtime;

import com.epsilon.core.Action;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.runtime.DecisionInferenceIngress;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * 各入力行で最も確率が高い合法手の位置を返す推論器。
 *
 * <p>対戦比較などで行動の抽選や価値予測を必要としない場合に使う。全候補の確率配列を返す推論器と同じ選択結果を、行ごとの出力オブジェクトを作らず取得できる。
 */
public interface EpsilonDecisionGreedyEvaluator extends EpsilonDecisionEvaluator {

  @Override
  default EpsilonDecisionGreedyEvaluator resolveInferenceEvaluator(List<Action> legalActions) {
    return this;
  }

  /**
   * 通常評価器を最大確率の行動を選ぶ方式能力へ適応する。選択位置だけの出力を実装済みならそのまま返す。
   *
   * @param evaluator 既存のDecision 評価器
   * @return 同じ方策で最大確率となる行動を返す評価器
   */
  static EpsilonDecisionGreedyEvaluator adapt(EpsilonDecisionEvaluator evaluator) {
    if (evaluator instanceof EpsilonDecisionGreedyEvaluator greedyEvaluator) {
      return greedyEvaluator;
    }
    return new PredictionBackedGreedyEvaluator(evaluator);
  }

  /**
   * 同じ型付き容量区分に属するDecision行を評価し、最大方策確率の枠を返す。
   *
   * @param batch ホスト上に構築済みの型付きバッチ
   * @return 入力行と同じ順序の0始まりの合法行動候補の位置
   */
  int[] evaluateGreedyActionSlots(DecisionHostBatch batch);

  /** 型付きバッチをパイプラインへ投入し、各行の最大確率の行動位置を非同期に返します。 */
  default CompletableFuture<int[]> submitGreedyActionSlots(DecisionHostBatch batch) {
    return CompletableFuture.completedFuture(evaluateGreedyActionSlots(batch));
  }

  /** 取得済みホスト入力受付枠と型付きバッチを移譲し、各行の最大確率の行動位置を非同期に返します。 */
  default CompletableFuture<int[]> submitGreedyActionSlots(
      DecisionHostBatch batch, DecisionInferenceIngress ingress) {
    DecisionInferenceIngress.Ownership ownership = ingress.handoff(this);
    ownership.release();
    return submitGreedyActionSlots(batch);
  }
}

/** 選択位置だけの出力を持たない特殊評価器について、通常Prediction の最大確率の行動を使う変換処理。 */
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
