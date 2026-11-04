package com.epsilon.reviewer.engine.review;

/**
 * 評価した 1 判断。配列は表示の順（最善が先）で、作った後は変えない。
 *
 * @param actionIds 候補の行動 ID（Action.toIndex）
 * @param values 価値のあるモデルは最善との価値差（損失、最善が 0）、無ければ確率
 * @param tiles 候補の鳴き・槓の牌。他の手は null
 * @param chosen 牌譜の手の位置。記録に無い判断は -1
 * @param probabilities 損失と温度から作れない表示の確率。作れる判断と価値の無いモデルは null
 * @param valued values が損失か
 */
public record EvaluatedDecision(
    int[] actionIds,
    float[] values,
    String[][] tiles,
    int chosen,
    float[] probabilities,
    boolean valued) {
  public int size() {
    return actionIds.length;
  }
}
