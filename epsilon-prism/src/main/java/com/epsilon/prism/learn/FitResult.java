package com.epsilon.prism.learn;

/**
 * 1 回の学習の記録（reports/iterations.jsonl に入る）。
 *
 * @param rows 使える行の数（並べ替えて順に回る行）
 * @param steps ステップ数
 * @param clippedSteps 勾配のノルムを上限で切ったステップ数の見積もり
 * @param loss 回収したステップの損失の平均（行あたり。回収したステップがなければ NaN）
 * @param firstLoss 最初に回収したステップの損失
 * @param lastLoss 最後に回収したステップの損失
 * @param seconds 学習にかかった時間（秒）
 * @param stageSeconds 段ごとの秒数 {ステップを作る（先読みスレッドの時間）, 入力と損失の教師の転送, 順伝播と逆伝播, 最適化}
 */
public record FitResult(
    int rows,
    int steps,
    int clippedSteps,
    double loss,
    double firstLoss,
    double lastLoss,
    double seconds,
    double[] stageSeconds) {
  /** 行もステップもない学習の記録。 */
  public static FitResult empty(int rows) {
    return new FitResult(rows, 0, 0, Double.NaN, Double.NaN, Double.NaN, 0, new double[4]);
  }

  /** 反復 {@code t} の学習を表すログの 1 行。 */
  public String summary(int t) {
    return String.format(
        "t=%d fit rows=%d steps=%d clipped=%d loss=%.4e first=%.4e last=%.4e %.1fs"
            + " stages=[pack %.0fs transfer %.0fs forward/backward %.0fs optimizer %.0fs]",
        t,
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
