package com.epsilon.agent.policy;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.Variant;
import java.util.OptionalDouble;

/**
 * 記録 O だけを読んで手を選ぶ方策。同じ O には、呼ぶ順や同時に呼ぶかどうかに関わらず同じ答えを返す。その場で答える
 * {@link InlinePolicy} と、まとめて推論する {@link BatchedPolicy} の 2 種類がある。
 */
public sealed interface Policy extends AutoCloseable permits InlinePolicy, BatchedPolicy {

  /** 読める O のルール。ほかのルールの O を渡すと IllegalArgumentException を投げる。 */
  Variant variant();

  /** 評価結果の行に付ける名前（{@link Series#label}）。 */
  String label();

  /** O の判断を評価する。呼ぶたびにメモリを確保するので、使うのは Reviewer とクライアントだけで、ホットパスでは使わない。 */
  Evaluation evaluate(Observation observation);

  /**
   * 価値から評価の確率を作るときの温度（効用の単位）。確率は softmax(価値 / 温度)。
   *
   * @return 温度。価値を出さない方策は空
   */
  default OptionalDouble temperature() {
    return OptionalDouble.empty();
  }

  @Override
  void close();
}
