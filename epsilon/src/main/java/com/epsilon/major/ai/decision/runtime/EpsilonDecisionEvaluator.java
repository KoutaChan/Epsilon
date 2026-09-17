package com.epsilon.major.ai.decision.runtime;

import com.epsilon.core.Action;
import com.epsilon.major.ai.decision.input.DecisionBucket;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.major.config.settings.DecisionInferenceSettings;
import com.epsilon.runtime.DecisionInferenceIngress;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/** 型付きの Decision バッチを評価するためのインターフェース。 */
public interface EpsilonDecisionEvaluator extends AutoCloseable {

  /**
   * 一行の合法手集合を実際に評価する実際に推論を行う評価器を返します。
   *
   * <p>通常の評価器は自身を返す。複数方策を振り分ける振り分け処理は、受付前に安定した実際に推論を行う評価器の識別情報へ解決することで、 一つの論理的な
   * バッチが複数の入力を受け付けられるバッチ枠を同時予約する状態を作らない。
   */
  default EpsilonDecisionEvaluator resolveInferenceEvaluator(List<Action> legalActions) {
    return this;
  }

  /**
   * 同じ型付き容量区分に属する Decision 行を一括評価する。
   *
   * @param batch ホスト上に構築済みの型付きバッチ
   * @return 入力行と同じ順序の推論結果
   */
  List<EpsilonDecisionInferenceServer.Prediction> evaluateBatch(DecisionHostBatch batch);

  /**
   * 型付きバッチをパイプラインへ投入します。
   *
   * <p>GPU実装はこのメソッドをoverrideします。小さなCPU テスト評価器は同期評価を完了済みFutureへ包みます。
   *
   * @param batch ホスト上に構築済みの型付きバッチ
   * @return 入力行順の推論結果を完了するFuture
   */
  default CompletableFuture<List<EpsilonDecisionInferenceServer.Prediction>> submitBatch(
      DecisionHostBatch batch) {
    return CompletableFuture.completedFuture(evaluateBatch(batch));
  }

  /** ホスト符号化用格納枠を非待機で取得し、失敗時は正確な解放通知を返します。 */
  default DecisionInferenceIngress.Attempt tryAcquireInferenceIngress() {
    return DecisionInferenceIngress.Attempt.acquired(DecisionInferenceIngress.direct(this));
  }

  /** この評価器が利用できるデバイスへ、次のバッチを供給する必要があるかを返す。 */
  default boolean needsInferenceWork() {
    return true;
  }

  /**
   * 取得済みホスト入力受付枠と型付きバッチを同時に移譲します。
   *
   * <p>このメソッドを呼んだ後、呼び出し側は入力受付枠を解放しません。同期失敗を含む全終了処理は受け取った側が行います。
   */
  default CompletableFuture<List<EpsilonDecisionInferenceServer.Prediction>> submitBatch(
      DecisionHostBatch batch, DecisionInferenceIngress ingress) {
    DecisionInferenceIngress.Ownership ownership = ingress.handoff(this);
    ownership.release();
    return submitBatch(batch);
  }

  /**
   * 容量区分非依存の標準デバイス側バッチ上限を返す。
   *
   * @return 一回の順伝播に入れる最大行数
   */
  default int preferredBatchSize() {
    return DecisionInferenceSettings.defaults().maxBatch();
  }

  /**
   * 指定型付き容量区分を一回のデバイス順伝播へ入れる最大行数を返す。
   *
   * @param bucket 行動・遷移容量を表す型付き容量区分
   * @return 一回の順伝播に入れる最大行数
   */
  default int preferredBatchSize(DecisionBucket bucket) {
    return preferredBatchSize();
  }

  /**
   * Streaming 対局実行処理が一度に構築する標準論理バッチ行数を返す。
   *
   * @return ホスト側で一度に構築する行数
   */
  default int preferredStreamingBatchSize() {
    return preferredBatchSize();
  }

  /**
   * Streaming 対局実行処理が指定型付き容量区分について一度に構築する論理バッチ行数を返す。
   *
   * @param bucket 行動・遷移容量を表す型付き容量区分
   * @return ホスト側で一度に構築する行数
   */
  default int preferredStreamingBatchSize(DecisionBucket bucket) {
    return preferredStreamingBatchSize();
  }

  @Override
  default void close() {}
}
