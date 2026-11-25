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
 * @param stageSeconds 段（{@link Stage}）ごとの秒数。添字は段の番号
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

  /** 学習の段。{@link #stageSeconds} の添字は段の番号。 */
  public enum Stage {
    /** ステップを作る（先読みスレッドの時間）。 */
    PACK("pack"),
    /** 入力と損失の教師の転送。 */
    TRANSFER("transfer"),
    /** 順伝播と逆伝播。 */
    COMPUTE("forward/backward"),
    /** 最適化。 */
    OPTIMIZER("optimizer");

    /** 段の数。 */
    public static final int COUNT = values().length;

    /** ログに書く名前。 */
    private final String label;

    Stage(String label) {
      this.label = label;
    }
  }

  /** 行もステップもない学習の記録。 */
  public static FitResult empty(int rows) {
    return new FitResult(
        rows, 0, 0, Double.NaN, Double.NaN, Double.NaN, 0, new double[Stage.COUNT]);
  }

  /** 反復 {@code t} の学習を表すログの 1 行。 */
  public String summary(int t) {
    StringBuilder stages = new StringBuilder();
    for (Stage stage : Stage.values()) {
      if (stage.ordinal() > 0) stages.append(' ');
      stages.append(stage.label).append(' ').append(String.format("%.0fs", seconds(stage)));
    }
    return String.format(
        "t=%d fit rows=%d steps=%d clipped=%d loss=%.4e first=%.4e last=%.4e %.1fs stages=[%s]",
        t, rows, steps, clippedSteps, loss, firstLoss, lastLoss, seconds, stages);
  }

  /** 段 {@code stage} の秒数（記録に無ければ NaN）。 */
  public double seconds(Stage stage) {
    int index = stage.ordinal();
    return stageSeconds != null && index < stageSeconds.length ? stageSeconds[index] : Double.NaN;
  }
}
