package com.epsilon.agent.play;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.runtime.DecisionInferenceIngress;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * 卓の席に座る打ち手。読むのは問い合わせの記録 O だけ。その場で答える打ち手は、進行のスレッドで答える。そうでない打ち手には、卓が
 * 受付の区分ごとにまとめた問い合わせに非同期で答えさせる。同じ O には、呼ぶ順や同時に呼ばれるかに依らず同じ答えを返す。
 *
 * @param <Q> 問い合わせの型
 */
public interface TableSeat<Q extends Query> extends AutoCloseable {

  /** 進行のスレッドで {@link #evaluateInline} を呼べるなら true。 */
  boolean inline();

  /** 答えをその場で書く。読むのは問い合わせの O だけ。 */
  default void evaluateInline(Q query) {
    throw new UnsupportedOperationException(getClass().getSimpleName() + " is not inline");
  }

  /** O を入れる受付の区分（0 以上）。同じ区分の問い合わせだけを 1 つのバッチにまとめる。問い合わせを作る進行のスレッドが呼ぶ。 */
  int admission(Observation observation);

  /** 受付の区分ごとに、1 つのバッチに入れられる問い合わせの上限。 */
  default int maxBatch(int admission) {
    return 4096;
  }

  /** 実行先が空いていて、満たないバッチでも出すべきなら true。 */
  default boolean needsWork() {
    return true;
  }

  /** 受付の区分のバッチ 1 つ分の容量を、待たずに予約する。 */
  default DecisionInferenceIngress.Attempt tryAcquireIngress(int admission) {
    return DecisionInferenceIngress.Attempt.acquired(DecisionInferenceIngress.direct(this));
  }

  /** バッチの全ての問い合わせに答えを書き、書き終えたら完了する。予約の所有権は、呼び出しと同時に打ち手へ移る。 */
  default CompletableFuture<Void> submit(List<Q> batch, DecisionInferenceIngress ingress) {
    throw new UnsupportedOperationException(getClass().getSimpleName() + " is inline");
  }

  @Override
  default void close() {}
}
