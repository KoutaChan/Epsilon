package com.epsilon.reviewer.engine.review;

/**
 * 評価した 1 つの判断。配列は表示の順（最善が先頭）に並ぶ。作った後は中身を変えない。
 *
 * @param actionIds 候補の行動の番号（{@code Action.code}）
 * @param values 価値を返すモデルなら最善との価値の差（損失。最善が 0）、返さないモデルなら確率
 * @param tiles 候補の鳴き・槓に使う牌。それ以外の手は null
 * @param chosen 牌譜の手が何番目か。牌譜に無い判断は -1
 * @param valued {@code values} が損失なら true、確率なら false
 */
public record EvaluatedDecision(
    int[] actionIds,
    float[] values,
    String[][] tiles,
    int chosen,
    boolean valued) {
  public int size() {
    return actionIds.length;
  }
}
