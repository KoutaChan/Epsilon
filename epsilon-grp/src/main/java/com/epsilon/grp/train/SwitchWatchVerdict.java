package com.epsilon.grp.train;

import java.util.List;

/**
 * GRP を切り替えた後の監視の判定。監視の行を重ならない 8 行ずつに分けて判定する。8 行の差の平均 d̄ = Σ d_h / 8 が +2 SE
 * （SE = √Σ SE_h² / 8）を超えたらアラートを出す。切り替えに損が無ければ、誤ってアラートが出るのは判定 1 回あたり約 2.3%。
 * どの行でも 1 SE ずつ悪ければ、判定の約 8 割でアラートが出る。
 */
public enum SwitchWatchVerdict {
  /** アラートなし。重みの半減期を 2 倍にしていれば元に戻す。 */
  QUIET,
  /** アラートあり。次の反復の始めに、直近に使った 8 個のモデルのうち最新のコホートで最も良いものへ戻し、重みの半減期を 2 倍にする。 */
  ROLLBACK,
  /** 重みの半減期を 2 倍にしている間に、またアラートが出た。再学習を止めて今のモデルを使い続け、人に知らせる。 */
  HALT;

  /** 判定 1 回に使う行の数。 */
  public static final int LINES = 8;

  /** アラートを出すしきい値。8 行の平均の SE の何倍か。 */
  static final double STANDARD_ERRORS = 2;

  /**
   * 重ならない 8 行を判定する。
   *
   * @param lines コホートの昇順に並べた 8 行
   * @param widened 今、重みの半減期を 2 倍にしていれば true
   * @throws IllegalArgumentException 行が 8 つでないとき、またはコホートが昇順でないとき
   */
  public static SwitchWatchVerdict judge(List<SwitchWatchLine> lines, boolean widened) {
    if (lines.size() != LINES) {
      throw new IllegalArgumentException("A switch watch verdict needs " + LINES + " lines, got " + lines.size());
    }
    double sum = 0;
    double variance = 0;
    for (int index = 0; index < LINES; index++) {
      SwitchWatchLine line = lines.get(index);
      if (index > 0 && line.cohort() <= lines.get(index - 1).cohort()) {
        throw new IllegalArgumentException("Switch watch lines must have ascending cohorts: " + lines);
      }
      sum += line.mseDifference();
      variance += line.standardError() * line.standardError();
    }
    double mean = sum / LINES;
    double standardError = Math.sqrt(variance) / LINES;
    if (!(mean > STANDARD_ERRORS * standardError)) {
      return QUIET;
    }
    return widened ? HALT : ROLLBACK;
  }
}
