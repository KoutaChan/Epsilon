package com.epsilon.prism.learn;

/**
 * 1 回の当てはめの記録（reports/iterations.jsonl に入る）。
 *
 * @param name 当てはめの名前（{@link Fit#NAME}）
 * @param rows 使える行の数（並べ替えて巡回する行）
 * @param steps 歩数
 * @param clippedSteps 勾配のノルムを上限で切った歩数の見積もり
 * @param loss 回収した歩の損失の平均（行あたり。回収した歩がなければ NaN）
 * @param firstLoss 最初に回収した歩の損失
 * @param lastLoss 最後に回収した歩の損失
 * @param stageSeconds {歩を作る（先読みスレッドの時間）, 入力と損失の教師の転送, 順伝播と逆伝播, 最適化}
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

  /** ログの 1 行（反復 t の当てはめ）。 */
  public String summary(int t) {
    return String.format(
        "t=%d fit %s rows=%d steps=%d clipped=%d loss=%.4e first=%.4e last=%.4e %.1fs"
            + " stages=[pack %.0fs transfer %.0fs forward/backward %.0fs optimizer %.0fs]",
        t,
        name,
        rows,
        steps,
        clippedSteps,
        loss,
        firstLoss,
        lastLoss,
        seconds,
        stage(0),
        stage(1),
        stage(2),
        stage(3));
  }

  private double stage(int index) {
    return stageSeconds != null && index < stageSeconds.length ? stageSeconds[index] : Double.NaN;
  }
}
