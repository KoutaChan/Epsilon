package com.epsilon.prism.learn;

/**
 * 1 回の当てはめの記録（reports/iterations.jsonl に入る）。
 *
 * @param name 当てはめの名前（"A"・"B"・"distill"）
 * @param rows 使える行の数（並べ替えて巡回する行）
 * @param steps 歩数
 * @param clippedSteps 勾配のノルムを上限で切った歩数の見積もり
 * @param loss 回収した歩の損失の平均（行あたり。回収した歩がなければ NaN）
 * @param firstLoss 最初に回収した歩の損失
 * @param lastLoss 最後に回収した歩の損失
 * @param stageSeconds {行の結合（先読みスレッドの時間）, 転送, 順伝播と逆伝播, 最適化}
 */
public record FitResult(
    String name,
    int rows,
    int steps,
    int clippedSteps,
    double loss,
    double firstLoss,
    double lastLoss,
    double seconds,
    double[] stageSeconds) {
  /** 行も歩もない当てはめ。 */
  public static FitResult empty(String name, int rows) {
    return new FitResult(name, rows, 0, 0, Double.NaN, Double.NaN, Double.NaN, 0, new double[4]);
  }
}
