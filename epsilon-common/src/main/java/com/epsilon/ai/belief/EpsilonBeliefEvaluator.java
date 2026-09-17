package com.epsilon.ai.belief;

import java.util.List;

/** 公開情報を入力に、他家の手牌・受け入れ牌・シャンテン数・テンパイ確率を予測するインターフェース。受け入れ牌は、テンパイ時には待ち牌を表す。 */
public interface EpsilonBeliefEvaluator<I> extends AutoCloseable {

  /**
   * 一局面から3人の他家について手牌、受け入れ牌、シャンテン数、テンパイ確率を予測する。
   *
   * @param input 1局面の型付き状態入力
   * @return 3人の他家について、手牌と受け入れ牌のロジット、シャンテン数、テンパイのロジットを格納した出力
   */
  EpsilonBeliefPrior evaluate(I input);

  /**
   * 複数入力を順番に評価する既定実装。
   *
   * @param inputs 評価する1局面ずつの状態入力群
   * @return 入力順に並べた他家状態の予測結果
   */
  default List<EpsilonBeliefPrior> evaluateBatch(List<I> inputs) {
    return inputs.stream().map(this::evaluate).toList();
  }

  /** 評価器が所有する推論資源を解放する。資源を持たない実装では何もしない。 */
  @Override
  default void close() {}
}
